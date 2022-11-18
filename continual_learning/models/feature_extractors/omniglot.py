from typing import Any

import torch

from continual_learning.models.autoencoder.omniglot import OmniglotAutoencoder
from continual_learning.models.feature_extractors.base import FeatureExtractor


class OmniglotFeatureEncoder(FeatureExtractor):
    def __init__(self, model: OmniglotAutoencoder, freeze: bool = True, device: torch.device = 'cpu'):
        self.model = model
        self.model.to(device)

        if freeze:
            self.model.freeze()

    def preprocess(self, x: Any) -> Any:
        return x.reshape((-1, 1, 28, 28))

    def get_features(self, x: Any) -> torch.Tensor:
        x = self.preprocess(x)
        tanh_encoding, _ = self.model.model.encode(x)
        return tanh_encoding
