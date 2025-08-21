"""
Unit tests for Vision Transformer perception models.

Tests the core ViT implementation including:
- Model architecture
- Forward pass
- Feature extraction
- Multi-modal inputs
- Temporal modeling
"""

import pytest
import torch
import torch.nn as nn

from fed_vit_autorl.models.vit_perception import ViTPerception


class TestViTPerception:
    """Test suite for Vision Transformer perception model."""

    def test_vit_initialization(self):
        """Test ViT model initialization with default parameters."""
        model = ViTPerception()

        assert isinstance(model, nn.Module)
        assert model.img_size == 224
        assert model.patch_size == 16
        assert model.embed_dim == 768
        assert model.depth == 12
        assert model.num_heads == 12

    def test_vit_custom_parameters(self):
        """Test ViT model initialization with custom parameters."""
        model = ViTPerception(
            img_size=384,
            patch_size=32,
            embed_dim=1024,
            depth=16,
            num_heads=16,
            num_classes=1000
        )

        assert model.img_size == 384
        assert model.patch_size == 32
        assert model.embed_dim == 1024
        assert model.depth == 16
        assert model.num_heads == 16

    def test_forward_pass_shape(self, sample_image_data):
        """Test forward pass produces correct output shape."""
        model = ViTPerception(num_classes=10)
        model.eval()

        with torch.no_grad():
            output = model(sample_image_data)

        batch_size = sample_image_data.shape[0]
        assert output.shape == (batch_size, 10)

    def test_feature_extraction(self, sample_image_data):
        """Test feature extraction from ViT backbone."""
        model = ViTPerception()
        model.eval()

        with torch.no_grad():
            features = model.extract_features(sample_image_data)

        batch_size = sample_image_data.shape[0]
        num_patches = (224 // 16) ** 2  # 196 patches
        assert features.shape == (batch_size, num_patches + 1, 768)  # +1 for CLS token

    def test_cls_token_extraction(self, sample_image_data):
        """Test CLS token extraction."""
        model = ViTPerception()
        model.eval()

        with torch.no_grad():
            cls_token = model.get_cls_token(sample_image_data)

        batch_size = sample_image_data.shape[0]
        assert cls_token.shape == (batch_size, 768)

    def test_different_input_sizes(self):
        """Test model with different input sizes."""
        test_sizes = [224, 384, 512]

        for img_size in test_sizes:
            model = ViTPerception(img_size=img_size, patch_size=16)
            model.eval()

            batch_size = 2
            input_tensor = torch.randn(batch_size, 3, img_size, img_size)

            with torch.no_grad():
                output = model(input_tensor)
                features = model.extract_features(input_tensor)

            assert output.shape[0] == batch_size

            num_patches = (img_size // 16) ** 2
            assert features.shape == (batch_size, num_patches + 1, 768)

    def test_patch_size_variations(self):
        """Test model with different patch sizes."""
        img_size = 224
        patch_sizes = [8, 16, 32]

        for patch_size in patch_sizes:
            if img_size % patch_size != 0:
                continue

            model = ViTPerception(img_size=img_size, patch_size=patch_size)
            model.eval()

            input_tensor = torch.randn(2, 3, img_size, img_size)

            with torch.no_grad():
                output = model(input_tensor)
                features = model.extract_features(input_tensor)

            num_patches = (img_size // patch_size) ** 2
            assert features.shape == (2, num_patches + 1, 768)

    def test_dropout_in_training_mode(self, sample_image_data):
        """Test that dropout is active in training mode."""
        model = ViTPerception(dropout=0.1)
        model.train()

        # Run forward pass multiple times and check for variation
        outputs = []
        for _ in range(5):
            output = model(sample_image_data)
            outputs.append(output.detach().clone())

        # Outputs should vary due to dropout
        for i in range(1, len(outputs)):
            assert not torch.allclose(outputs[0], outputs[i], atol=1e-6)

    def test_model_parameters_count(self):
        """Test that model has expected number of parameters."""
        model = ViTPerception()

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # ViT-Base should have around 86M parameters
        assert 80_000_000 < total_params < 90_000_000
        assert total_params == trainable_params  # All parameters should be trainable

    def test_gradient_flow(self, sample_image_data):
        """Test that gradients flow properly through the model."""
        model = ViTPerception(num_classes=10)
        criterion = nn.CrossEntropyLoss()

        # Create dummy targets
        batch_size = sample_image_data.shape[0]
        targets = torch.randint(0, 10, (batch_size,))

        # Forward pass
        outputs = model(sample_image_data)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Check that gradients exist and are non-zero for at least some parameters
        has_gradients = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_gradients = True
                break

        assert has_gradients, "No gradients found in model parameters"

    def test_model_serialization(self, temp_dir):
        """Test model saving and loading."""
        model = ViTPerception(num_classes=10)

        # Save model
        model_path = temp_dir / "vit_model.pth"
        torch.save(model.state_dict(), model_path)

        # Load model
        new_model = ViTPerception(num_classes=10)
        new_model.load_state_dict(torch.load(model_path))

        # Test that loaded model produces same output
        model.eval()
        new_model.eval()

        sample_input = torch.randn(1, 3, 224, 224)

        with torch.no_grad():
            original_output = model(sample_input)
            loaded_output = new_model(sample_input)

        assert torch.allclose(original_output, loaded_output, atol=1e-6)

    @pytest.mark.parametrize("batch_size", [1, 4, 8, 16])
    def test_different_batch_sizes(self, batch_size):
        """Test model with different batch sizes."""
        model = ViTPerception(num_classes=10)
        model.eval()

        input_tensor = torch.randn(batch_size, 3, 224, 224)

        with torch.no_grad():
            output = model(input_tensor)

        assert output.shape == (batch_size, 10)

    def test_attention_weights_extraction(self, sample_image_data):
        """Test extraction of attention weights from the model."""
        model = ViTPerception()
        model.eval()

        # Enable attention weight extraction
        model.return_attention = True

        with torch.no_grad():
            output, attention_weights = model(sample_image_data)

        assert isinstance(attention_weights, list)
        assert len(attention_weights) == model.depth  # One per transformer layer

        batch_size = sample_image_data.shape[0]
        num_patches = (224 // 16) ** 2 + 1  # +1 for CLS token

        for attn in attention_weights:
            assert attn.shape == (batch_size, model.num_heads, num_patches, num_patches)

    def test_positional_embedding_sizes(self):
        """Test positional embedding dimensions for different configurations."""
        configs = [
            (224, 16),  # Standard ViT-Base
            (384, 16),  # ViT with larger input
            (224, 32),  # ViT with larger patches
        ]

        for img_size, patch_size in configs:
            model = ViTPerception(img_size=img_size, patch_size=patch_size)

            num_patches = (img_size // patch_size) ** 2
            expected_pos_embed_size = num_patches + 1  # +1 for CLS token

            assert model.pos_embedding.shape[1] == expected_pos_embed_size
