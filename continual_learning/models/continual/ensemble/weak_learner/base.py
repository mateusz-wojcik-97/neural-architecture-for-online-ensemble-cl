from abc import ABC, abstractmethod

from torch import Tensor


class WeakLearner(ABC):
    @abstractmethod
    def fit(self, x: Tensor, y: Tensor):
        pass

    @abstractmethod
    def predict(self, x: Tensor) -> Tensor:
        pass
