"""Fed-ViT-AutoRL: Federated Vision Transformers for Autonomous Driving.

A federated reinforcement learning framework where edge vehicles jointly
fine-tune Vision Transformer based perception stacks while respecting
latency and privacy constraints.
"""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

__all__ = ["__version__"]