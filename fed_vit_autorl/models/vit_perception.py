"""Vision Transformer backbone for perception tasks."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig


class ViTPerception(nn.Module):
    """Vision Transformer for autonomous driving perception.
    
    Args:
        img_size: Input image size
        patch_size: Patch size for ViT
        num_classes: Number of output classes
        embed_dim: Embedding dimension
        depth: Number of transformer layers
        num_heads: Number of attention heads
        pretrained: Whether to use pretrained weights
    """
    
    def __init__(
        self,
        img_size: int = 384,
        patch_size: int = 16,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        
        # Configure ViT
        config = ViTConfig(
            image_size=img_size,
            patch_size=patch_size,
            num_channels=3,
            hidden_size=embed_dim,
            num_hidden_layers=depth,
            num_attention_heads=num_heads,
            intermediate_size=embed_dim * 4,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )
        
        self.vit = ViTModel(config)
        self.classifier = nn.Linear(embed_dim, num_classes)
        
        if pretrained:
            self._load_pretrained_weights()
    
    def _load_pretrained_weights(self) -> None:
        """Load pretrained ViT weights from transformers."""
        # This would load actual pretrained weights in practice
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ViT perception model.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Features tensor of shape (batch_size, num_classes)
        """
        outputs = self.vit(pixel_values=x)
        features = outputs.last_hidden_state[:, 0]  # CLS token
        return self.classifier(features)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification head.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Feature tensor of shape (batch_size, embed_dim)
        """
        outputs = self.vit(pixel_values=x)
        return outputs.last_hidden_state[:, 0]  # CLS token