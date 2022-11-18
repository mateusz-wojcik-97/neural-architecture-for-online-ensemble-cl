from abc import ABC, abstractmethod
from typing import Any

import torch


class FeatureExtractor(ABC):

    @abstractmethod
    def preprocess(self, x: Any) -> Any:
        pass

    @abstractmethod
    def get_features(self, x: Any) -> torch.Tensor:
        pass
