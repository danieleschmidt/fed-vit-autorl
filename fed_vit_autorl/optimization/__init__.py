"""Performance optimization utilities."""

# Import non-torch dependent components first
from .caching_system import AdaptiveCache, CacheManager, cached
from .auto_scaler import ResourceOptimizer, PredictiveScaler
from .load_balancer import AdvancedLoadBalancer, BalancingAlgorithm

# Try to import torch-dependent components
try:
    from .performance import (
        ModelOptimizer,
        DistributedTrainingManager,
        MemoryManager,
        AsyncDataLoader,
        GradientCompressor,
        AdaptiveScheduler,
        PerformanceProfile,
    )
    _TORCH_COMPONENTS_AVAILABLE = True
except ImportError:
    _TORCH_COMPONENTS_AVAILABLE = False

    # Placeholder classes for torch-dependent components
    class _TorchDependentPlaceholder:
        def __init__(self, *args, **kwargs):
            raise ImportError("This component requires torch. Install with: pip install torch")

    ModelOptimizer = _TorchDependentPlaceholder
    DistributedTrainingManager = _TorchDependentPlaceholder
    MemoryManager = _TorchDependentPlaceholder
    AsyncDataLoader = _TorchDependentPlaceholder
    GradientCompressor = _TorchDependentPlaceholder
    AdaptiveScheduler = _TorchDependentPlaceholder
    PerformanceProfile = _TorchDependentPlaceholder

__all__ = [
    # Always available
    "AdaptiveCache",
    "CacheManager",
    "cached",
    "ResourceOptimizer",
    "PredictiveScaler",
    "AdvancedLoadBalancer",
    "BalancingAlgorithm",

    # May require torch
    "ModelOptimizer",
    "DistributedTrainingManager",
    "MemoryManager",
    "AsyncDataLoader",
    "GradientCompressor",
    "AdaptiveScheduler",
    "PerformanceProfile",
]
