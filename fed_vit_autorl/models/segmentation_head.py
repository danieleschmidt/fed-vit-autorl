"""Semantic segmentation head for ViT backbone."""

from typing import Dict, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SegmentationHead(nn.Module):
    """Semantic segmentation head for lane detection and free space estimation.

    Uses a sophisticated decoder with skip connections and multi-scale features
    for dense prediction tasks like lane detection, drivable area segmentation,
    and semantic understanding.

    Args:
        input_dim: Input feature dimension from backbone
        num_classes: Number of segmentation classes
        img_size: Input image size for upsampling
        patch_size: Patch size of the ViT backbone
        use_skip_connections: Whether to use skip connections
        class_names: Optional list of class names
    """

    # Automotive segmentation classes
    DEFAULT_CLASSES = [
        'road', 'lane_marking', 'sidewalk', 'building', 'wall', 'fence',
        'traffic_light', 'traffic_sign', 'vegetation', 'sky'
    ]

    def __init__(
        self,
        input_dim: int = 768,
        num_classes: int = 10,
        img_size: int = 384,
        patch_size: int = 16,
        use_skip_connections: bool = True,
        class_names: Optional[List[str]] = None,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.img_size = img_size
        self.patch_size = patch_size
        self.use_skip_connections = use_skip_connections
        self.class_names = class_names or self.DEFAULT_CLASSES[:num_classes]

        # Calculate patch grid size
        self.patch_h = self.patch_w = img_size // patch_size

        # Multi-scale feature processing
        self.feature_pyramid = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
            ),
            nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
            ),
            nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
            ),
        ])

        # Patch-to-image reconstruction
        self.patch_decoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, patch_size * patch_size * 32),
        )

        # Progressive upsampling with skip connections
        self.upconv1 = nn.Sequential(
            nn.ConvTranspose2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.upconv2 = nn.Sequential(
            nn.ConvTranspose2d(64 + 128, 32, 4, stride=2, padding=1),  # +128 for skip
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.upconv3 = nn.Sequential(
            nn.ConvTranspose2d(32 + 64, 16, 4, stride=2, padding=1),   # +64 for skip
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(16 + 256, num_classes, 3, padding=1),  # +256 for skip
            nn.BatchNorm2d(num_classes),
        )

        # Attention mechanism for feature refinement
        self.attention = nn.Sequential(
            nn.Conv2d(num_classes, num_classes // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_classes // 4, num_classes, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def forward(self, patch_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through segmentation head.

        Args:
            patch_features: Input patch features from backbone
                          (batch_size, num_patches + 1, input_dim)

        Returns:
            Dictionary containing:
                - 'segmentation': Segmentation map (batch_size, num_classes, H, W)
                - 'features': Multi-scale features for auxiliary losses
        """
        batch_size = patch_features.size(0)

        # Remove CLS token, keep only patch features
        patch_tokens = patch_features[:, 1:]  # (B, num_patches, input_dim)

        # Multi-scale feature extraction
        pyramid_features = []
        for fpn_layer in self.feature_pyramid:
            features = fpn_layer(patch_tokens)  # (B, num_patches, feat_dim)
            pyramid_features.append(features)

        # Decode patches to pixel features
        decoded = self.patch_decoder(patch_tokens)  # (B, num_patches, patch_size^2 * 32)
        decoded = decoded.view(
            batch_size, self.patch_h, self.patch_w,
            self.patch_size, self.patch_size, 32
        )

        # Rearrange to get image-like features
        decoded = decoded.permute(0, 5, 1, 3, 2, 4).contiguous()
        decoded = decoded.view(
            batch_size, 32,
            self.patch_h * self.patch_size,
            self.patch_w * self.patch_size
        )  # (B, 32, H//4, W//4)

        # Progressive upsampling with skip connections
        x = self.upconv1(decoded)  # (B, 64, H//2, W//2)

        if self.use_skip_connections:
            # Add skip connection from pyramid features
            skip_feat = pyramid_features[1].view(batch_size, self.patch_h, self.patch_w, -1)
            skip_feat = skip_feat.permute(0, 3, 1, 2)  # (B, 128, patch_h, patch_w)
            skip_feat = F.interpolate(skip_feat, size=x.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip_feat], dim=1)

        x = self.upconv2(x)  # (B, 32, H, W)

        if self.use_skip_connections:
            # Add another skip connection
            skip_feat = pyramid_features[2].view(batch_size, self.patch_h, self.patch_w, -1)
            skip_feat = skip_feat.permute(0, 3, 1, 2)  # (B, 64, patch_h, patch_w)
            skip_feat = F.interpolate(skip_feat, size=x.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip_feat], dim=1)

        x = self.upconv3(x)  # (B, 16, 2H, 2W)

        if self.use_skip_connections:
            # Final skip connection
            skip_feat = pyramid_features[0].view(batch_size, self.patch_h, self.patch_w, -1)
            skip_feat = skip_feat.permute(0, 3, 1, 2)  # (B, 256, patch_h, patch_w)
            skip_feat = F.interpolate(skip_feat, size=x.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip_feat], dim=1)

        # Final segmentation prediction
        seg_logits = self.final_conv(x)  # (B, num_classes, H, W)

        # Apply attention mechanism
        attention_weights = self.attention(seg_logits)
        seg_logits = seg_logits * attention_weights

        # Ensure correct output size
        if seg_logits.size(-1) != self.img_size:
            seg_logits = F.interpolate(
                seg_logits,
                size=(self.img_size, self.img_size),
                mode='bilinear',
                align_corners=False
            )

        return {
            'segmentation': torch.softmax(seg_logits, dim=1),
            'logits': seg_logits,
            'features': pyramid_features,
        }

    def _init_weights(self) -> None:
        """Initialize layer weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def compute_loss(self, predictions: Dict[str, torch.Tensor],
                    targets: torch.Tensor,
                    class_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute segmentation loss with class weighting.

        Args:
            predictions: Predictions from forward pass
            targets: Ground truth segmentation (B, H, W)
            class_weights: Optional class weights for imbalanced datasets

        Returns:
            Segmentation loss
        """
        logits = predictions['logits']

        # Compute cross entropy loss
        loss = F.cross_entropy(
            logits, targets,
            weight=class_weights,
            reduction='mean'
        )

        # Add auxiliary losses from multi-scale features if needed
        # This could include boundary loss, focal loss, etc.

        return loss
