from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union, Optional

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import normalize
from torch import Tensor
from torch.nn import ModuleList
from tqdm import tqdm

from continual_learning.models.feature_extractors.base import FeatureExtractor


def init_weights(input_size: int, output_size: int):
    raw_weights_init = tf.keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='truncated_normal')(
        shape=(128, input_size, output_size)).numpy()
    return raw_weights_init[0].transpose(1, 0)


class WeakLearner(nn.Module):
    def __init__(self, input_size: int, output_size: int, tanh_factor: float, device: torch.device):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear = torch.nn.Linear(input_size, output_size)
        raw_weights_init = tf.keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='truncated_normal')(shape=(60, input_size, output_size)).numpy()
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
        super().__init__()
        self.encoder = encoder
        self.input_size = input_size
        self.num_classes = num_classes
        self.num_classifiers = num_classifiers
        self.k_neighbors = k_neighbors
        self.keys = self._init_keys(trainable=trainable_keys)
        self.classes = classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.tanh_factor = tanh_factor
        self.hard_voting = hard_voting
        self.device = device

        assert len(self.keys) == self.num_classifiers

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

    def _init_keys(self, trainable: bool = False) -> nn.Parameter:
        keys = np.random.normal(size=(self.num_classifiers, self.input_size))
        keys = torch.from_numpy(keys)
        keys = keys.type(torch.FloatTensor)
        return nn.Parameter(nn.functional.normalize(keys, p=2, dim=1, eps=1e-12), requires_grad=trainable)

    def lookup_memory(self, input: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        input = normalize(input, norm='l2', copy=False)
        cos_similarity = np.transpose(self.keys.cpu().detach().numpy() @ input.T)
        arg_sorted_indexes = np.argsort(cos_similarity, axis=1)
        indexes = np.flip(arg_sorted_indexes, axis=1)
        cos_similarity = np.take_along_axis(cos_similarity, indices=indexes, axis=1)
        indexes = indexes[:, :self.k_neighbors]
        cos_similarity = cos_similarity[:, :self.k_neighbors]
        return indexes, cos_similarity

    def encode(self, example: Tensor) -> Tensor:
        return self.encoder.get_features(example)  # TODO self.encoder.get_features(example.to(torch.device('cpu')))

    def ensemble_forward(self, x: torch.Tensor, indexes, cos_similarity) -> torch.Tensor:
        ensemble_outputs_batch = []
        for single_input, single_indexes, single_cos_sim in zip(x, indexes, cos_similarity):
            ensemble_outputs = [model.forward(single_input.view(1, -1)) for index, model in enumerate(self.models) if index in single_indexes]
            ensemble_outputs_stacked = torch.cat(ensemble_outputs, dim=0).unsqueeze(0)
            ensemble_outputs_batch.append(ensemble_outputs_stacked)
        ensemble_outputs_batch = torch.cat(ensemble_outputs_batch, dim=0).to(self.device)
        cosine_similarities = torch.tensor(data=cos_similarity).unsqueeze(dim=2).to(self.device)
        ensemble_outputs = torch.sum(ensemble_outputs_batch * cosine_similarities, dim=1) / torch.sum(cosine_similarities, dim=1)
        return ensemble_outputs

    def forward(self, x: Tensor, x_is_encoded: bool = False):
        if not x_is_encoded:
            x = self.encode(x)

        indexes, cos_similarity = self.lookup_memory(x.cpu().detach().numpy())

        x = x.to(self.device)
        vanilla_output = self.vanilla_classifier.forward(x)
        tanh_output = self.tanh_classifier.forward(x)
        ensemble_outputs = self.ensemble_forward(x, indexes, cos_similarity)

        return ensemble_outputs, tanh_output, vanilla_output

    def predict(self, x: Tensor, return_dict: bool = False) -> Union[Tensor, Dict[str, Tensor]]:
        x = self.encode(x)

        indexes, cos_similarity = self.lookup_memory(x.cpu().detach().numpy())

        x = x.to(self.device)
        with torch.no_grad():
            vanilla_outputs = self.vanilla_classifier.forward(x)
            tanh_outputs = self.tanh_classifier.forward(x)
            ensemble_outputs = self.ensemble_forward(x, indexes, cos_similarity)

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
