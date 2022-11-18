from collections import defaultdict
from typing import Optional, Any, Dict, List, Tuple, Union

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from functorch import combine_state_for_ensemble, vmap
from pytorch_lightning import LightningModule
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import normalize
from torch import Tensor
from torch.nn import ModuleList
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader

from continual_learning.models.autoencoder.omniglot import OmniglotAutoencoder
from continual_learning.models.custom_layers.differentiable_knn_layer import DKNN
from continual_learning.optimizers.naive_optimizer import NaiveOptimizer


class WeakLearner(nn.Module):
    def __init__(self, input_size: int, output_size: int, tanh_factor: float):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear = torch.nn.Linear(input_size, output_size)
        raw_weights_init = tf.keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='truncated_normal')(shape=(128, input_size, output_size)).numpy()
        single_weak_learner_weights = raw_weights_init[0].transpose(1, 0)
        self.linear.weight.data = torch.Tensor(single_weak_learner_weights)
        # self.linear.weight.data = torch.normal(mean=0, std=np.sqrt(1.0 / input_size), size=(output_size, input_size))  # TODO weights initialization scale
        self.tanh_factor = tanh_factor

    def forward(self, x: torch.Tensor):
        raw_input = x.view(x.size(0), -1)
        raw_output = self.linear(raw_input)
        return torch.tanh(raw_output / self.tanh_factor) * self.tanh_factor


class VanillaClassifier(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
        raw_weights_init = tf.keras.initializers.VarianceScaling(scale=0.1, mode='fan_in', distribution='truncated_normal')(shape=(input_size, output_size)).numpy()
        self.linear.weight.data = torch.Tensor(raw_weights_init.transpose(1, 0))

    def forward(self, x: torch.Tensor):
        raw_input = x.view(x.size(0), -1)
        raw_output = self.linear(raw_input)
        return torch.log_softmax(raw_output, dim=1)


class TanhClassifier(nn.Module):
    def __init__(self, input_size: int, output_size: int, tanh_factor: float):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
        raw_weights_init = tf.keras.initializers.VarianceScaling(scale=0.1, mode='fan_in', distribution='truncated_normal')(shape=(input_size, output_size)).numpy()
        self.linear.weight.data = torch.Tensor(raw_weights_init.transpose(1, 0))
        self.tanh_factor = tanh_factor

    def forward(self, x: torch.Tensor):
        raw_input = x.view(x.size(0), -1)
        raw_output = self.linear(raw_input)
        return torch.tanh(raw_output / self.tanh_factor) * self.tanh_factor


class EnsembleE2EModule(nn.Module):
    def __init__(
        self,
        encoder: OmniglotAutoencoder,
        input_size: int,
        num_classifiers: int,
        k_neighbors: int,
        keys: torch.Tensor,
        classes: List[int],
        learning_rate: float,
        weight_decay: float,
        tanh_factor: float,
        hard_voting: bool,
        freeze_encoder: bool,
        trainable_keys: bool,
    ):
        """Adaptation of https://arxiv.org/pdf/2105.13327.pdf for end-to-end continual learning.
         :param keys: memory size x key size
        """
        super().__init__()
        self.encoder = encoder
        self.input_size = input_size
        self.num_classifiers = num_classifiers
        self.k_neighbors = k_neighbors
        self.trainable_keys = trainable_keys
        self.keys = nn.Parameter(nn.functional.normalize(keys, p=2, dim=1, eps=1e-12), requires_grad=trainable_keys)
        self.classes = classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.tanh_factor = tanh_factor
        self.hard_voting = hard_voting

        assert len(keys) == self.num_classifiers

        self.knn_num_samples = 128  # TODO unused
        self.knn_use_manual_grad = True
        self.knn_epsilon = 5e-4
        self.knn_inner_iter = 400
        self.dknn = DKNN(
            k=k_neighbors,
            num_samples=self.knn_num_samples,
            num_neighbors=self.num_classifiers,
            use_manual_grad=self.knn_use_manual_grad,
            epsilon=self.knn_epsilon,
            max_iter=self.knn_inner_iter,
        )

        self.num_classes = len(classes)

        self.distribution_tracker = GaussianNB()

        if freeze_encoder:
            self.encoder.freeze()

        self.models = self.configure_weak_learners()
        self.vanilla_classifier = VanillaClassifier(
            input_size=self.input_size, output_size=self.num_classes
        )
        self.tanh_classifier = TanhClassifier(
            input_size=self.input_size, output_size=self.num_classes, tanh_factor=self.tanh_factor
        )
        self.optimizer = self.configure_optimizer()
        self.keys_optimizer = torch.optim.Adam(
            params=[{'params': self.keys}], lr=0.00005,
        )
        self.models_stats = defaultdict(lambda: defaultdict(int))
        self.seen_classes = set()
        self.train_dataset = None
        self.train_loader = None

        self.vanilla_losses = []
        self.tanh_losses = []
        self.memory_losses = []
        self.key_losses = []
        self.ensemble_losses = []

    def configure_optimizer(self) -> Optimizer:
        return NaiveOptimizer(
            [
                {'params': self.vanilla_classifier.parameters()},
                {'params': self.tanh_classifier.parameters()},
                {'params': self.models.parameters()},
                # {'params': self.keys},
            ],
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay
        )

    def configure_weak_learners(self) -> ModuleList:
        return ModuleList([WeakLearner(input_size=self.input_size, output_size=self.num_classes, tanh_factor=self.tanh_factor) for _ in range(self.num_classifiers)])

    def lookup_memory(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input = torch.nn.functional.normalize(input, p=2, dim=1, eps=1e-12)
        knn_similarity, cos_similarity, cos_distance = self.dknn(
            query=input, neighbors=self.keys, cosine_distance=True, return_distances=True
        )
        return knn_similarity, cos_similarity, cos_distance

    def encode(self, example: Tensor) -> Tensor:
        example = example.reshape((-1, 1, 28, 28))
        tanh_encoding, _ = self.encoder.model.encode(example)
        return tanh_encoding

    def get_loss(
        self,
        vanilla_outputs: torch.Tensor,
        tanh_outputs: torch.Tensor,
        ensemble_outputs: torch.Tensor,
        cos_distance: torch.Tensor,
        knn_similarity: torch.Tensor,
        examples: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Label to one hot
        labels_one_hot = nn.functional.one_hot(labels, num_classes=self.num_classes)
        # Classifier loss
        vanilla_loss = -torch.mean(torch.sum(vanilla_outputs * labels_one_hot, dim=1))
        tanh_loss = -torch.mean(torch.sum(tanh_outputs * labels_one_hot, dim=1))
        # Memory loss
        memory_loss = -torch.mean(torch.sum(ensemble_outputs * labels_one_hot, dim=1))
        # Keys loss
        key_loss = torch.mean(torch.sum(cos_distance * knn_similarity, dim=1))
        # Total loss
        loss = vanilla_loss + tanh_loss + memory_loss + key_loss
        return loss, vanilla_loss, tanh_loss, memory_loss, key_loss

    def ensemble_forward(self, x: torch.Tensor, knn_similarity: torch.Tensor, cos_similarity: torch.Tensor) -> torch.Tensor:
        ensemble_outputs_batch = []
        for single_input, single_knn_sim, single_cos_sim in zip(x, knn_similarity, cos_similarity):
            ensemble_outputs = [model.forward(single_input.view(1, -1)) for index, model in enumerate(self.models)]
            ensemble_outputs_stacked = torch.cat(ensemble_outputs, dim=0).unsqueeze(0)
            ensemble_outputs_batch.append(ensemble_outputs_stacked)
        ensemble_outputs_batch = torch.cat(ensemble_outputs_batch, dim=0)
        cosine_similarities = torch.tensor(data=cos_similarity).unsqueeze(dim=2)
        knn_similarities = knn_similarity.unsqueeze(dim=2)
        ensemble_outputs_batch = ensemble_outputs_batch * knn_similarities
        cosine_similarities = cosine_similarities * knn_similarities
        ensemble_outputs = torch.sum(ensemble_outputs_batch * cosine_similarities, dim=1) / torch.sum(cosine_similarities, dim=1)
        return ensemble_outputs

    def fit(self, x: Tensor, y: Tensor, x_is_encoded: bool = False):
        if not x_is_encoded:
            x = self.encode(x)

        self.distribution_tracker.partial_fit(x.detach().numpy(), y.detach().numpy(), classes=self.classes)

        self.optimizer.zero_grad()
        self.keys_optimizer.zero_grad()

        knn_similarity, cos_similarity, cos_distance = self.lookup_memory(x)

        # self.update_model_stats(indexes, y.tolist())

        vanilla_output = self.vanilla_classifier.forward(x)
        tanh_output = self.tanh_classifier.forward(x)
        ensemble_outputs = self.ensemble_forward(x, knn_similarity, cos_similarity)

        loss, vanilla_loss, tanh_loss, memory_loss, key_loss = self.get_loss(vanilla_output, tanh_output, ensemble_outputs, cos_distance, knn_similarity, x, y)
        loss.backward()

        # from torchviz import make_dot
        # make_dot(loss, params=dict(list(self.models.named_parameters()) + [('keys', self.keys)])).render("attached", format="png")
        # print(self.keys)

        self.optimizer.step()
        self.keys_optimizer.step()

        for new_label in torch.unique(y):
            self.update_seen_classes(new_label.item())

        self.vanilla_losses.append(vanilla_loss.item())
        self.tanh_losses.append(tanh_loss.item())
        self.memory_losses.append(memory_loss.item())
        self.ensemble_losses.append(loss.item())
        self.key_losses.append(key_loss.item())

    def predict(self, x: Tensor, return_dict: bool = False) -> Union[Tensor, Dict[str, Tensor]]:
        x = self.encode(x)

        knn_similarity, cos_similarity, cos_distance = self.lookup_memory(x)

        with torch.no_grad():
            vanilla_outputs = self.vanilla_classifier.forward(x)
            tanh_outputs = self.tanh_classifier.forward(x)
            ensemble_outputs = self.ensemble_forward(x, knn_similarity, cos_similarity)

            y_pred_vanilla = torch.argmax(vanilla_outputs, dim=1)
            y_pred_tanh = torch.argmax(tanh_outputs, dim=1)
            y_pred_ensemble = torch.argmax(ensemble_outputs, dim=1)

            if return_dict:
                return {
                    'vanilla': y_pred_vanilla,
                    'tanh': y_pred_tanh,
                    'ensemble': y_pred_ensemble,
                }

            return y_pred_ensemble

    def is_known_class(self, label: Any) -> bool:
        return label in self.seen_classes

    def update_seen_classes(self, label: Any) -> None:
        self.seen_classes.add(label)

    def update_model_stats(self, indexes: np.ndarray, labels: Any) -> None:
        for index_group, label in zip(indexes, labels):
            for index in index_group:
                self.models_stats[index][label] += 1

    def plot_loss(self):
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
        df_loss = pd.DataFrame({
            'Batch': list(range(len(self.ensemble_losses))),
            'Ensemble': self.ensemble_losses,
            'Memory': self.memory_losses,
            'Vanilla': self.vanilla_losses,
            'Tanh': self.tanh_losses,
            'Key': self.key_losses,
        })
        sns.lineplot(x='Batch', y='Loss', hue='Loss type',
                     data=pd.melt(df_loss, ['Batch'], var_name='Loss type', value_name='Loss'))
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.savefig('Loss.png')


def generate_example_from_nb(model, class_to_sample: int, examples_count: int = 1):
    class_mean = model.theta_[class_to_sample]
    class_std = np.sqrt(model.var_[class_to_sample])
    sampled = np.random.normal(loc=class_mean, scale=class_std,
                               size=(examples_count, len(class_mean))).reshape(examples_count, -1)

    sampled = np.clip(sampled, -1, 1)
    return sampled


def generate_batch(nb_model, classes_to_sample, shuffle: bool = True) -> torch.tensor:
    generated_examples = []
    generated_labels = []
    for class_index, examples_count in classes_to_sample.items():
        generated_example = generate_example_from_nb(nb_model, class_index, examples_count)
        generated_tensor = torch.from_numpy(generated_example)
        generated_examples.append(generated_tensor)
        generated_labels.extend([class_index] * examples_count)

    batch = torch.cat(generated_examples, dim=0).float()
    labels = torch.tensor(generated_labels)

    if shuffle:
        indexes_shuffled = torch.randperm(len(labels))
        batch = batch[indexes_shuffled].view(batch.size())
        labels = labels[indexes_shuffled]

    return batch, labels
