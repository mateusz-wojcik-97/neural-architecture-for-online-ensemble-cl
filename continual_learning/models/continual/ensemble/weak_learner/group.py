from collections import defaultdict
from typing import Optional, Type, Any, Union, Dict, List

import numpy as np
import torch
from functorch import combine_state_for_ensemble, vmap
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, IterableDataset

from continual_learning.models.continual.ensemble.weak_learner.base import WeakLearner
from continual_learning.models.continual.ensemble.weak_learner.neural_network_sklearn import \
    NeuralNetworkSklearn


class WeakLearnersGroup:
    def __init__(self, size: int, weak_learner: Type[WeakLearner], **kwargs):
        self.weak_learner = weak_learner
        self.weak_learner_kwargs = kwargs
        self.size = size
        self.models = self.construct_weak_learners()
        self.models_stats = defaultdict(lambda: defaultdict(int))
        self.seen_classes = set()
        self.train_dataset = None
        self.train_loader = None

    def construct_weak_learners(self) -> NDArray[WeakLearner]:
        return np.array([self.weak_learner(**self.weak_learner_kwargs) for _ in range(self.size)])

    def fit(self, x: Tensor, y: Tensor, indexes_to_fit: Optional[np.ndarray] = None, clf_meta: Optional[Dict[str, Any]] = None):
        if indexes_to_fit is None:
            indexes_to_fit = np.array(range(self.size))

        if self.train_dataset is None:
            self.train_dataset = MNISTTrainDataset(x, y)
            self.train_loader = DataLoader(self.train_dataset, batch_size=1, num_workers=0)
        else:
            self.train_dataset.update_buffer(x, y)

        # data_loader = data_loader_from_tensor(x, y)

        for index, model in enumerate(self.models[indexes_to_fit]):
            model.fit(self.train_loader, clf_meta={**clf_meta, 'index': index})

    def predict(self, x: Tensor, indexes_to_predict: Optional[np.ndarray] = None, clf_meta: Optional[Dict[str, Any]] = None) -> Tensor:
        if indexes_to_predict is None:
            indexes_to_predict = np.array(range(self.size))

        # for index, model in enumerate(self.models[indexes_to_predict]):
        #     model.set

        # models_to_predict = [m.model for m in self.models[indexes_to_predict]]
        # fmodel, params, buffers = combine_state_for_ensemble(models_to_predict)
        # y_pred = vmap(fmodel, in_dims=(0, 0, None))(params, buffers, x)
        # y_pred = y_pred.squeeze(dim=1)
        # y_pred = torch.argmax(y_pred, dim=1)
        # y_pred = y_pred.detach()

        models_to_predict = self.models[indexes_to_predict]
        data_loader = data_loader_from_tensor(x)
        y_pred = []
        for model in models_to_predict:
            y_pred_model = model.predict(data_loader, clf_meta=clf_meta)
            y_pred.extend(y_pred_model)
        y_pred = torch.vstack(y_pred)

        return y_pred

    def is_known_class(self, label: Any) -> bool:
        return label in self.seen_classes

    def update_seen_classes(self, label: Any) -> None:
        self.seen_classes.add(label)

    def update_model_stats(self, indexes: np.ndarray, label: Any) -> None:
        for index in indexes:
            self.models_stats[index][label] += 1


def data_loader_from_tensor(x: Tensor, y: Optional[Tensor] = None) -> DataLoader:
    if y is None:
        dataset = MNISTTestDataset(x)
    else:
        dataset = MNISTTrainDataset(x, y)
    return DataLoader(dataset, batch_size=1, num_workers=6)


class MNISTTrainDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x, self.y

    def __len__(self):
        return 1

    def update_buffer(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y


class MNISTTestDataset(Dataset):
    def __init__(self, x: torch.Tensor):
        self.x = x

    def __getitem__(self, index):
        return self.x

    def __len__(self):
        return 1


if __name__ == '__main__':
    sample_x = Tensor([[0.5, 0.6, 0.7], [0.1, 0.2, 0.3]])
    sample_y = Tensor([0, 1])
    classes = [0, 1]
    group = WeakLearnersGroup(
        size=10, weak_learner=NeuralNetworkSklearn, classes=classes, learning_rate_init=0.001
    )
    group.fit(sample_x, sample_y)
    y_pred = group.predict(sample_x)
    print(y_pred)
