"""Tests for vision transformer models."""

import pytest
import torch

from fed_vit_autorl.models import ViTPerception, DetectionHead, SegmentationHead


class TestViTPerception:
    """Test cases for ViTPerception model."""

    def test_vit_perception_init(self):
        """Test ViTPerception initialization."""
        model = ViTPerception(
            img_size=224,
            patch_size=16,
            num_classes=100,
            embed_dim=768,
            depth=12,
            num_heads=12,
        )

        assert model.img_size == 224
        assert model.patch_size == 16
        assert model.num_classes == 100

    def test_vit_perception_forward(self):
        """Test ViTPerception forward pass."""
        model = ViTPerception(img_size=224, num_classes=10)
        x = torch.randn(2, 3, 224, 224)

        output = model(x)

        assert output.shape == (2, 10)

    def test_vit_perception_get_features(self):
        """Test ViTPerception feature extraction."""
        model = ViTPerception(img_size=224, embed_dim=768)
        x = torch.randn(2, 3, 224, 224)

        features = model.get_features(x)

        assert features.shape == (2, 768)


class TestDetectionHead:
    """Test cases for DetectionHead."""

    def test_detection_head_init(self):
        """Test DetectionHead initialization."""
        head = DetectionHead(
            input_dim=768,
            num_classes=20,
            num_anchors=9,
        )

        assert head.input_dim == 768
        assert head.num_classes == 20
        assert head.num_anchors == 9

    def test_detection_head_forward(self):
        """Test DetectionHead forward pass."""
        head = DetectionHead(input_dim=768, num_classes=20, num_anchors=9)
        features = torch.randn(2, 768)

        output = head(features)

        assert 'classes' in output
        assert 'boxes' in output
        assert 'objectness' in output
        assert output['classes'].shape == (2, 9, 20)
        assert output['boxes'].shape == (2, 9, 4)
        assert output['objectness'].shape == (2, 9)


class TestSegmentationHead:
    """Test cases for SegmentationHead."""

    def test_segmentation_head_init(self):
        """Test SegmentationHead initialization."""
        head = SegmentationHead(
            input_dim=768,
            num_classes=10,
            img_size=224,
        )

        assert head.input_dim == 768
        assert head.num_classes == 10
        assert head.img_size == 224

    def test_segmentation_head_forward(self):
        """Test SegmentationHead forward pass."""
        head = SegmentationHead(input_dim=768, num_classes=10, img_size=224)
        features = torch.randn(2, 768)

        output = head(features)

        assert 'segmentation' in output
        assert output['segmentation'].shape == (2, 10, 224, 224)
