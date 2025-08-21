"""Vision Transformer models for federated autonomous driving."""

try:
    from .vit_perception import ViTPerception
    from .detection_head import DetectionHead
    from .segmentation_head import SegmentationHead

    __all__ = ["ViTPerception", "DetectionHead", "SegmentationHead"]
except ImportError:
    # Graceful degradation when torch is not available
    import warnings

    class _MockModel:
        def __init__(self, *args, **kwargs):
            warnings.warn("torch not available, using mock implementation")

    ViTPerception = _MockModel
    DetectionHead = _MockModel
    SegmentationHead = _MockModel

    __all__ = ["ViTPerception", "DetectionHead", "SegmentationHead"]
