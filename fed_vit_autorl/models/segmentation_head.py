"""Semantic segmentation head for ViT backbone."""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationHead(nn.Module):
    """Semantic segmentation head for lane detection and free space estimation.
    
    Args:
        input_dim: Input feature dimension from backbone
        num_classes: Number of segmentation classes
        img_size: Input image size for upsampling
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        num_classes: int = 10,
        img_size: int = 384,
    ) -> None:
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.img_size = img_size
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )
        
        # Segmentation head
        self.seg_head = nn.Linear(128, num_classes)
        
        # Upsampling layers
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(num_classes, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, num_classes, 4, stride=2, padding=1),
        )
        
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through segmentation head.
        
        Args:
            features: Input features from backbone (batch_size, input_dim)
            
        Returns:
            Dictionary containing:
                - 'segmentation': Segmentation map (batch_size, num_classes, H, W)
        """
        batch_size = features.size(0)
        
        # Decode features
        x = self.decoder(features)
        seg_logits = self.seg_head(x)
        
        # Reshape for upsampling (treat as 1x1 feature map)
        seg_logits = seg_logits.view(batch_size, self.num_classes, 1, 1)
        
        # Upsample to original image size
        seg_map = self.upsample(seg_logits)
        
        # Resize to exact input size if needed
        if seg_map.size(-1) != self.img_size:
            seg_map = F.interpolate(
                seg_map, 
                size=(self.img_size, self.img_size),
                mode='bilinear',
                align_corners=False
            )
        
        return {
            'segmentation': torch.softmax(seg_map, dim=1),
        }