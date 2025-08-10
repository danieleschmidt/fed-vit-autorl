"""Vision Transformer models for federated autonomous driving."""

try:
    from .vit_perception import ViTPerception
    from .detection_head import DetectionHead
    from .segmentation_head import SegmentationHead
    
    __all__ = ["ViTPerception", "DetectionHead", "SegmentationHead"]
except ImportError:
    # Graceful degradation when torch is not available
    class _MissingDependency:
        def __init__(self, *args, **kwargs):
            raise ImportError("This component requires torch. Install with: pip install torch")
    
    ViTPerception = _MissingDependency
    DetectionHead = _MissingDependency
    SegmentationHead = _MissingDependency
    
    __all__ = ["ViTPerception", "DetectionHead", "SegmentationHead"]