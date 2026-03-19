"""Tests for TinyViT model."""

import pytest
import torch
from fed_vit_autorl.models import TinyViT, PatchEmbedding, TransformerBlock


class TestPatchEmbedding:
    def test_output_shape_patch8(self):
        patch_embed = PatchEmbedding(img_size=32, patch_size=8, in_channels=3, embed_dim=64)
        x = torch.randn(2, 3, 32, 32)
        out = patch_embed(x)
        assert out.shape == (2, 16, 64)  # (32/8)^2 = 16 patches

    def test_output_shape_patch16(self):
        patch_embed = PatchEmbedding(img_size=32, patch_size=16, in_channels=3, embed_dim=64)
        x = torch.randn(2, 3, 32, 32)
        out = patch_embed(x)
        assert out.shape == (2, 4, 64)   # (32/16)^2 = 4 patches

    def test_invalid_patch_size(self):
        with pytest.raises(AssertionError):
            PatchEmbedding(img_size=32, patch_size=7)


class TestTransformerBlock:
    def test_output_shape(self):
        block = TransformerBlock(embed_dim=64, num_heads=4)
        x = torch.randn(2, 16, 64)
        out = block(x)
        assert out.shape == x.shape

    def test_residual_connection(self):
        """Output should differ from input (not all zeros)."""
        block = TransformerBlock(embed_dim=64, num_heads=4)
        x = torch.randn(2, 16, 64)
        out = block(x)
        assert not torch.allclose(out, x)


class TestTinyViT:
    def setup_method(self):
        self.model = TinyViT(
            img_size=32,
            patch_size=8,
            num_classes=25,
            embed_dim=64,
            depth=2,
            num_heads=4,
        )

    def test_output_shape(self):
        x = torch.randn(4, 3, 32, 32)
        out = self.model(x)
        assert out.shape == (4, 25)

    def test_batch_size_1(self):
        x = torch.randn(1, 3, 32, 32)
        out = self.model(x)
        assert out.shape == (1, 25)

    def test_parameter_count(self):
        n = self.model.num_parameters()
        assert n > 0
        assert n < 5_000_000  # Should be tiny

    def test_depth3(self):
        model = TinyViT(depth=3, embed_dim=64, num_heads=4)
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 25)

    def test_no_nan_in_output(self):
        x = torch.randn(4, 3, 32, 32)
        out = self.model(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
