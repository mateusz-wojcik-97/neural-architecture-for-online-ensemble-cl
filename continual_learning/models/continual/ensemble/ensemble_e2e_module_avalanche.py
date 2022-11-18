from collections import defaultdict
from typing import Optional, Any, Dict, List, Tuple, Union

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.naive_bayes import GaussianNB
from torch import Tensor
from torch.nn import ModuleList

from continual_learning.models.custom_layers.differentiable_knn_layer import DKNN
from continual_learning.models.feature_extractors.base import FeatureExtractor


class WeakLearner(nn.Module):
    def __init__(self, input_size: int, output_size: int, tanh_factor: float, device: torch.device):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear = torch.nn.Linear(input_size, output_size)
        raw_weights_init = tf.keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='truncated_normal')(shape=(128, input_size, output_size)).numpy()
        single_weak_learner_weights = raw_weights_init[0].transpose(1, 0)
        self.linear.weight.data = torch.Tensor(single_weak_learner_weights)
        self.tanh_factor = tanh_factor
        self.linear.to(device)

    def forward(self, x: torch.Tensor):
        raw_input = x.view(x.size(0), -1)
        raw_output = self.linear(raw_input)
        return torch.tanh(raw_output / self.tanh_factor) * self.tanh_factor


class VanillaClassifier(nn.Module):
    def __init__(self, input_size: int, output_size: int, device: torch.device):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
        raw_weights_init = tf.keras.initializers.VarianceScaling(scale=0.1, mode='fan_in', distribution='truncated_normal')(shape=(input_size, output_size)).numpy()
        self.linear.weight.data = torch.Tensor(raw_weights_init.transpose(1, 0))
        self.linear.to(device)

    def forward(self, x: torch.Tensor):
        raw_input = x.view(x.size(0), -1)
        raw_output = self.linear(raw_input)
        return torch.log_softmax(raw_output, dim=1)


class TanhClassifier(nn.Module):
    def __init__(self, input_size: int, output_size: int, tanh_factor: float, device: torch.device):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
        raw_weights_init = tf.keras.initializers.VarianceScaling(scale=0.1, mode='fan_in', distribution='truncated_normal')(shape=(input_size, output_size)).numpy()
        self.linear.weight.data = torch.Tensor(raw_weights_init.transpose(1, 0))
        self.tanh_factor = tanh_factor
        self.linear.to(device)

    def forward(self, x: torch.Tensor):
        raw_input = x.view(x.size(0), -1)
        raw_output = self.linear(raw_input)
        return torch.tanh(raw_output / self.tanh_factor) * self.tanh_factor


class EnsembleE2EModule(nn.Module):
    def __init__(
        self,
        encoder: FeatureExtractor,
        input_size: int,
        num_classifiers: int,
        k_neighbors: int,
        learning_rate: float,
        weight_decay: float,
        tanh_factor: float,
        hard_voting: bool,
        trainable_keys: bool,
        num_classes: int,
        classes: Optional[List[int]] = None,
        device: torch.device = 'cpu',
    ):
        """Adaptation of https://arxiv.org/pdf/2105.13327.pdf for end-to-end continual learning.
         :param keys: memory size x key size
        """
        super().__init__()
        self.encoder = encoder
        self.input_size = input_size
        self.num_classes = num_classes
        self.num_classifiers = num_classifiers
        self.k_neighbors = k_neighbors
        self.device = device
        self.keys = self._init_keys(trainable=trainable_keys)
        self.classes = classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.tanh_factor = tanh_factor
        self.hard_voting = hard_voting

        assert len(self.keys) == self.num_classifiers

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
            device=self.device,
        )

        self.distribution_tracker = GaussianNB()

        self.models = ModuleList([WeakLearner(input_size=self.input_size, output_size=self.num_classes, tanh_factor=self.tanh_factor, device=self.device) for _ in range(self.num_classifiers)])
        self.vanilla_classifier = VanillaClassifier(
            input_size=self.input_size, output_size=self.num_classes, device=self.device
        ).to(self.device)
        self.tanh_classifier = TanhClassifier(
            input_size=self.input_size, output_size=self.num_classes, tanh_factor=self.tanh_factor, device=self.device
        ).to(self.device)
        self.keys_optimizer = torch.optim.Adam(
            params=[{'params': self.keys}], lr=0.00005,
        )
        self.models_stats = defaultdict(lambda: defaultdict(int))
        self.seen_classes = set()
        self.train_dataset = None
        self.train_loader = None

    def _init_keys(self, trainable: bool = False):
        keys = np.random.normal(size=(self.num_classifiers, self.input_size))
        keys = torch.from_numpy(keys).to(self.device)
        keys = keys.type(torch.FloatTensor)
        return nn.Parameter(nn.functional.normalize(keys, p=2, dim=1, eps=1e-12), requires_grad=trainable)

    def lookup_memory(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input = torch.nn.functional.normalize(input, p=2, dim=1, eps=1e-12).to(self.device)
        knn_similarity, cos_similarity, cos_distance = self.dknn(
            query=input, neighbors=self.keys, cosine_distance=True, return_distances=True
        )
        return knn_similarity, cos_similarity, cos_distance

    def encode(self, example: Tensor) -> Tensor:
        return self.encoder.get_features(example)

    def ensemble_forward(self, x: torch.Tensor, knn_similarity: torch.Tensor, cos_similarity: torch.Tensor) -> torch.Tensor:
        ensemble_outputs_batch = []
        for single_input, single_knn_sim, single_cos_sim in zip(x, knn_similarity, cos_similarity):
            ensemble_outputs = [model.forward(single_input.view(1, -1)) for index, model in enumerate(self.models)]
            ensemble_outputs_stacked = torch.cat(ensemble_outputs, dim=0).unsqueeze(0)
            ensemble_outputs_batch.append(ensemble_outputs_stacked)
        ensemble_outputs_batch = torch.cat(ensemble_outputs_batch, dim=0).to(self.device)
        cosine_similarities = torch.tensor(data=cos_similarity).unsqueeze(dim=2).to(self.device)
        knn_similarities = knn_similarity.unsqueeze(dim=2).to(self.device)
        ensemble_outputs_batch = ensemble_outputs_batch * knn_similarities
        cosine_similarities = cosine_similarities * knn_similarities
        ensemble_outputs = torch.sum(ensemble_outputs_batch * cosine_similarities, dim=1) / torch.sum(cosine_similarities, dim=1)
        return ensemble_outputs

    def forward(self, x: Tensor, x_is_encoded: bool = False):
        if not x_is_encoded:
            x = self.encode(x)
        x = x.to(self.device)
        # self.distribution_tracker.partial_fit(x.detach().numpy(), y.detach().numpy(), classes=self.classes)

        # self.keys_optimizer.zero_grad()

        knn_similarity, cos_similarity, cos_distance = self.lookup_memory(x)

        # self.update_model_stats(indexes, y.tolist())

        vanilla_output = self.vanilla_classifier.forward(x)
        tanh_output = self.tanh_classifier.forward(x)
        ensemble_outputs = self.ensemble_forward(x, knn_similarity, cos_similarity)

        # self.keys_optimizer.step()  # TODO

        # for new_label in torch.unique(y):
        #     self.update_seen_classes(new_label.item())

        return ensemble_outputs, tanh_output, vanilla_output, cos_distance, knn_similarity

    def predict(self, x: Tensor, return_dict: bool = False) -> Union[Tensor, Dict[str, Tensor]]:
        x = self.encode(x)  # TODO is_encoded
        x = x.to(self.device)
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
