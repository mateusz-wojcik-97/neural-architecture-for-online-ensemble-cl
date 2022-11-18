from typing import List

from sklearn.neural_network import MLPClassifier
from torch import Tensor, from_numpy

from continual_learning.models.continual.ensemble.weak_learner.base import WeakLearner


class NeuralNetworkSklearn(WeakLearner):
    def __init__(self, classes: List[int], **kwargs):
        self.classes = classes
        self.model = MLPClassifier(**kwargs)

    def fit(self, x: Tensor, y: Tensor):
        x_numpy, y_numpy = x.numpy(), y.tolist()
        self.model.partial_fit(x_numpy, y_numpy, self.classes)

    def predict(self, x: Tensor) -> Tensor:
        x_numpy = x.numpy()
        y_pred = self.model.predict(x_numpy)
        return from_numpy(y_pred)
