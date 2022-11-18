import torch.nn
from omegaconf import DictConfig

from continual_learning.models.autoencoder.omniglot import OmniglotAutoencoder
from continual_learning.models.feature_extractors.base import FeatureExtractor
from continual_learning.models.feature_extractors.byol import load_byol_resnet_from_checkpoint, \
    ByolFeatureEncoder
from continual_learning.models.feature_extractors.omniglot import OmniglotFeatureEncoder
from settings import MODELS_DIR

DEFAULT_BYOL_ENCODER_PATH = MODELS_DIR / 'resnet50_byol' / 'resnet50_byol_imagenet2012.pth.tar'
DEFAULT_OMNIGLOT_ENCODER_PATH = MODELS_DIR / 'ensemble_omniglot_autoencoder' / 'encoder.ckpt'


def load_encoder(encoder_name: str, config: DictConfig) -> FeatureExtractor:
    if encoder_name == 'omniglot':
        autoencoder = OmniglotAutoencoder.load_from_checkpoint(
            checkpoint_path=DEFAULT_OMNIGLOT_ENCODER_PATH,
            # map_location=torch.device('cuda'),
            # Params only for compatibility
            input_size=28,
            encoder_size=config.sample_size,
            learning_rate=0.001,
        )
        encoder = OmniglotFeatureEncoder(model=autoencoder)
        return encoder
    elif encoder_name == 'byol':
        return ByolFeatureEncoder(DEFAULT_BYOL_ENCODER_PATH, device=config.device)
    else:
        raise ValueError(f"Dataset {encoder_name} is not available")
