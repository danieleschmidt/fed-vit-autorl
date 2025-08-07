"""Comprehensive tests for enhanced model components."""

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock

from fed_vit_autorl.models.vit_perception import ViTPerception
from fed_vit_autorl.models.detection_head import DetectionHead
from fed_vit_autorl.models.segmentation_head import SegmentationHead


class TestEnhancedViTPerception:
    """Test enhanced ViT perception model."""
    
    @pytest.fixture
    def vit_model(self):
        """Create ViT model for testing."""
        return ViTPerception(
            img_size=224,
            patch_size=16,
            num_classes=10,
            embed_dim=768,
            depth=12,
            num_heads=12,
            pretrained=False
        )
    
    def test_vit_initialization(self, vit_model):
        """Test ViT model initialization."""
        assert vit_model.img_size == 224
        assert vit_model.patch_size == 16
        assert vit_model.num_classes == 10
        assert isinstance(vit_model.vit, nn.Module)
        assert isinstance(vit_model.classifier, nn.Linear)
    
    def test_forward_pass(self, vit_model):
        """Test forward pass through ViT."""
        batch_size = 2
        x = torch.randn(batch_size, 3, 224, 224)
        
        output = vit_model(x)
        
        assert output.shape == (batch_size, 10)
        assert not torch.isnan(output).any()
    
    def test_feature_extraction(self, vit_model):
        """Test feature extraction without classifier."""
        batch_size = 2
        x = torch.randn(batch_size, 3, 224, 224)
        
        features = vit_model.get_features(x)
        
        assert features.shape == (batch_size, 768)
        assert not torch.isnan(features).any()
    
    def test_patch_features_extraction(self, vit_model):
        """Test patch feature extraction for dense tasks."""
        batch_size = 2
        x = torch.randn(batch_size, 3, 224, 224)
        
        patch_features = vit_model.get_patch_features(x)
        
        # 224/16 = 14, so 14*14 + 1(CLS) = 197 patches
        expected_num_patches = (224 // 16) ** 2 + 1
        assert patch_features.shape == (batch_size, expected_num_patches, 768)
        assert not torch.isnan(patch_features).any()
    
    def test_backbone_freezing(self, vit_model):
        """Test backbone parameter freezing."""
        # Initially all parameters should be trainable
        trainable_params = sum(1 for p in vit_model.vit.parameters() if p.requires_grad)
        assert trainable_params > 0
        
        # Freeze backbone
        vit_model.freeze_backbone()
        frozen_params = sum(1 for p in vit_model.vit.parameters() if not p.requires_grad)
        assert frozen_params > 0
        
        # Unfreeze backbone
        vit_model.unfreeze_backbone()
        unfrozen_params = sum(1 for p in vit_model.vit.parameters() if p.requires_grad)
        assert unfrozen_params > 0
    
    @patch('fed_vit_autorl.models.vit_perception.ViTModel')
    def test_pretrained_loading_success(self, mock_vit_model):
        """Test successful pretrained weight loading."""
        # Mock the from_pretrained method
        mock_pretrained = MagicMock()
        mock_pretrained.state_dict.return_value = {'encoder.layer.0.weight': torch.randn(768, 768)}
        mock_vit_model.from_pretrained.return_value = mock_pretrained
        
        # Create model with pretrained=True
        model = ViTPerception(pretrained=True)
        
        # Verify from_pretrained was called
        mock_vit_model.from_pretrained.assert_called_once_with('google/vit-base-patch16-384')
    
    @patch('fed_vit_autorl.models.vit_perception.ViTModel')
    def test_pretrained_loading_failure(self, mock_vit_model):
        """Test graceful handling of pretrained loading failure."""
        # Mock the from_pretrained to raise an exception
        mock_vit_model.from_pretrained.side_effect = Exception("Network error")
        
        # Should not raise an exception, but continue with random initialization
        model = ViTPerception(pretrained=True)
        assert model is not None


class TestEnhancedDetectionHead:
    """Test enhanced detection head."""
    
    @pytest.fixture
    def detection_head(self):
        """Create detection head for testing."""
        return DetectionHead(
            input_dim=768,
            num_classes=20,
            num_anchors=9,
            img_size=384
        )
    
    def test_detection_initialization(self, detection_head):
        """Test detection head initialization."""
        assert detection_head.input_dim == 768
        assert detection_head.num_classes == 20
        assert detection_head.num_anchors == 9
        assert detection_head.img_size == 384
        assert len(detection_head.class_names) == 20
        assert detection_head.anchors.shape == (9, 2)
    
    def test_detection_forward(self, detection_head):
        """Test forward pass through detection head."""
        batch_size = 2
        features = torch.randn(batch_size, 768)
        
        predictions = detection_head(features)
        
        # Check output structure
        assert 'classes' in predictions
        assert 'boxes' in predictions
        assert 'objectness' in predictions
        assert 'confidence' in predictions
        
        # Check shapes
        assert predictions['classes'].shape == (batch_size, 9, 20)
        assert predictions['boxes'].shape == (batch_size, 9, 4)
        assert predictions['objectness'].shape == (batch_size, 9)
        assert predictions['confidence'].shape == (batch_size, 9)
        
        # Check value ranges
        assert torch.all(predictions['classes'] >= 0) and torch.all(predictions['classes'] <= 1)
        assert torch.all(predictions['objectness'] >= 0) and torch.all(predictions['objectness'] <= 1)
        assert torch.all(predictions['confidence'] >= 0) and torch.all(predictions['confidence'] <= 1)
    
    def test_anchor_generation(self, detection_head):
        """Test anchor box generation."""
        anchors = detection_head.anchors
        
        # Should have 9 anchors (3 scales x 3 aspect ratios)
        assert anchors.shape == (9, 2)
        
        # All anchors should be positive
        assert torch.all(anchors > 0)
    
    def test_box_decoding(self, detection_head):
        """Test box prediction decoding."""
        batch_size = 2
        raw_boxes = torch.randn(batch_size, 9, 4)
        
        decoded_boxes = detection_head._decode_boxes(raw_boxes)
        
        assert decoded_boxes.shape == (batch_size, 9, 4)
        # After sigmoid and scaling, all coordinates should be positive
        assert torch.all(decoded_boxes >= 0)
        assert torch.all(decoded_boxes <= 384)  # Should be within image size
    
    def test_nms_functionality(self, detection_head):
        """Test Non-Maximum Suppression."""
        # Create overlapping boxes
        boxes = torch.tensor([
            [100, 100, 50, 50],  # Box 1
            [105, 105, 50, 50],  # Box 2 (overlapping with Box 1)
            [200, 200, 50, 50],  # Box 3 (separate)
        ], dtype=torch.float32)
        scores = torch.tensor([0.9, 0.8, 0.7])
        
        keep_indices = detection_head._nms(boxes, scores, threshold=0.5)
        
        # Should keep boxes 1 and 3 (remove overlapping box 2)
        assert len(keep_indices) == 2
        assert 0 in keep_indices  # Highest scoring box
        assert 2 in keep_indices  # Non-overlapping box
    
    def test_post_processing(self, detection_head):
        """Test complete post-processing pipeline."""
        batch_size = 2
        predictions = {
            'classes': torch.rand(batch_size, 9, 20),
            'boxes': torch.rand(batch_size, 9, 4) * 384,
            'objectness': torch.rand(batch_size, 9),
            'confidence': torch.rand(batch_size, 9),
        }
        
        results = detection_head.post_process(
            predictions, 
            conf_threshold=0.5, 
            nms_threshold=0.4
        )
        
        assert len(results) == batch_size
        for result in results:
            assert 'boxes' in result
            assert 'scores' in result
            assert 'classes' in result
            
            # All scores should be above threshold
            if result['scores'].numel() > 0:
                assert torch.all(result['scores'] > 0.5)


class TestEnhancedSegmentationHead:
    """Test enhanced segmentation head."""
    
    @pytest.fixture
    def segmentation_head(self):
        """Create segmentation head for testing."""
        return SegmentationHead(
            input_dim=768,
            num_classes=10,
            img_size=384,
            patch_size=16
        )
    
    def test_segmentation_initialization(self, segmentation_head):
        """Test segmentation head initialization."""
        assert segmentation_head.input_dim == 768
        assert segmentation_head.num_classes == 10
        assert segmentation_head.img_size == 384
        assert segmentation_head.patch_size == 16
        assert len(segmentation_head.class_names) == 10
        assert segmentation_head.patch_h == 24  # 384 / 16
        assert segmentation_head.patch_w == 24
    
    def test_segmentation_forward(self, segmentation_head):
        """Test forward pass through segmentation head."""
        batch_size = 2
        num_patches = (384 // 16) ** 2  # 576 patches
        # Include CLS token
        patch_features = torch.randn(batch_size, num_patches + 1, 768)
        
        predictions = segmentation_head(patch_features)
        
        # Check output structure
        assert 'segmentation' in predictions
        assert 'logits' in predictions
        assert 'features' in predictions
        
        # Check segmentation map shape
        seg_map = predictions['segmentation']
        assert seg_map.shape == (batch_size, 10, 384, 384)
        
        # Check probability distribution (should sum to 1 across classes)
        class_sums = seg_map.sum(dim=1)
        assert torch.allclose(class_sums, torch.ones_like(class_sums), atol=1e-6)
        
        # Check logits shape
        assert predictions['logits'].shape == (batch_size, 10, 384, 384)
        
        # Check multi-scale features
        assert len(predictions['features']) == 3
    
    def test_loss_computation(self, segmentation_head):
        """Test segmentation loss computation."""
        batch_size = 2
        predictions = {
            'logits': torch.randn(batch_size, 10, 384, 384),
            'segmentation': torch.rand(batch_size, 10, 384, 384),
        }
        targets = torch.randint(0, 10, (batch_size, 384, 384))
        
        loss = segmentation_head.compute_loss(predictions, targets)
        
        assert loss.item() >= 0  # Loss should be non-negative
        assert not torch.isnan(loss)
    
    def test_weighted_loss_computation(self, segmentation_head):
        """Test segmentation loss with class weights."""
        batch_size = 2
        predictions = {
            'logits': torch.randn(batch_size, 10, 384, 384),
            'segmentation': torch.rand(batch_size, 10, 384, 384),
        }
        targets = torch.randint(0, 10, (batch_size, 384, 384))
        class_weights = torch.rand(10)  # Random weights for each class
        
        loss = segmentation_head.compute_loss(predictions, targets, class_weights)
        
        assert loss.item() >= 0
        assert not torch.isnan(loss)
    
    def test_skip_connections(self):
        """Test model with and without skip connections."""
        # Model with skip connections
        model_with_skip = SegmentationHead(use_skip_connections=True)
        
        # Model without skip connections
        model_without_skip = SegmentationHead(use_skip_connections=False)
        
        batch_size = 1
        num_patches = (384 // 16) ** 2
        patch_features = torch.randn(batch_size, num_patches + 1, 768)
        
        # Both should work
        pred_with_skip = model_with_skip(patch_features)
        pred_without_skip = model_without_skip(patch_features)
        
        assert pred_with_skip['segmentation'].shape == pred_without_skip['segmentation'].shape


class TestModelIntegration:
    """Test integration between model components."""
    
    def test_vit_to_detection_pipeline(self):
        """Test ViT -> Detection head pipeline."""
        vit = ViTPerception(img_size=224, pretrained=False)
        detection_head = DetectionHead(input_dim=768, img_size=224)
        
        batch_size = 2
        x = torch.randn(batch_size, 3, 224, 224)
        
        # Extract features from ViT
        features = vit.get_features(x)
        
        # Pass through detection head
        detections = detection_head(features)
        
        assert detections['classes'].shape == (batch_size, 9, 20)
        assert not torch.isnan(detections['classes']).any()
    
    def test_vit_to_segmentation_pipeline(self):
        """Test ViT -> Segmentation head pipeline."""
        vit = ViTPerception(img_size=384, pretrained=False)
        seg_head = SegmentationHead(input_dim=768, img_size=384)
        
        batch_size = 2
        x = torch.randn(batch_size, 3, 384, 384)
        
        # Extract patch features from ViT
        patch_features = vit.get_patch_features(x)
        
        # Pass through segmentation head
        segmentation = seg_head(patch_features)
        
        assert segmentation['segmentation'].shape == (batch_size, 10, 384, 384)
        assert not torch.isnan(segmentation['segmentation']).any()
    
    def test_multi_task_model(self):
        """Test using ViT for both detection and segmentation."""
        vit = ViTPerception(img_size=384, pretrained=False)
        detection_head = DetectionHead(input_dim=768, img_size=384)
        seg_head = SegmentationHead(input_dim=768, img_size=384)
        
        batch_size = 2
        x = torch.randn(batch_size, 3, 384, 384)
        
        # Single forward pass through ViT
        cls_features = vit.get_features(x)
        patch_features = vit.get_patch_features(x)
        
        # Multi-task predictions
        detections = detection_head(cls_features)
        segmentation = seg_head(patch_features)
        
        # Both tasks should work with same ViT features
        assert detections['classes'].shape == (batch_size, 9, 20)
        assert segmentation['segmentation'].shape == (batch_size, 10, 384, 384)
    
    def test_gradient_flow(self):
        """Test that gradients flow properly through the models."""
        vit = ViTPerception(img_size=224, pretrained=False)
        detection_head = DetectionHead(input_dim=768, img_size=224)
        
        batch_size = 2
        x = torch.randn(batch_size, 3, 224, 224, requires_grad=True)
        
        # Forward pass
        features = vit.get_features(x)
        detections = detection_head(features)
        
        # Compute dummy loss
        loss = detections['objectness'].sum()
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        
        # Check ViT parameters have gradients
        vit_params_with_grad = sum(1 for p in vit.parameters() if p.grad is not None)
        assert vit_params_with_grad > 0
        
        # Check detection head parameters have gradients
        det_params_with_grad = sum(1 for p in detection_head.parameters() if p.grad is not None)
        assert det_params_with_grad > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])