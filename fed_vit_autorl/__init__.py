"""
fed-vit-autorl: Federated Learning of Vision Transformers with AutoRL hyperparameter tuning.
"""

from .models import TinyViT
from .data import SyntheticImageDataset
from .federated import FederatedViTClient, FederatedViTServer
from .rl_tuner import RLHyperparamTuner

__version__ = "0.1.0"
__all__ = [
    "TinyViT",
    "SyntheticImageDataset",
    "FederatedViTClient",
    "FederatedViTServer",
    "RLHyperparamTuner",
]
