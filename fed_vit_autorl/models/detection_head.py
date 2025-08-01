"""Object detection head for ViT backbone."""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn


class DetectionHead(nn.Module):
    """Object detection head for autonomous driving scenarios.
    
    Predicts bounding boxes and classes for objects like vehicles,
    pedestrians, traffic signs, etc.
    
    Args:
        input_dim: Input feature dimension from backbone
        num_classes: Number of object classes
        num_anchors: Number of anchor boxes per location
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        num_classes: int = 20,
        num_anchors: int = 9,
    ) -> None:
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        
        # Classification head
        self.cls_head = nn.Linear(256, num_anchors * num_classes)
        
        # Regression head (x, y, w, h)
        self.reg_head = nn.Linear(256, num_anchors * 4)
        
        # Objectness head
        self.obj_head = nn.Linear(256, num_anchors)
        
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through detection head.
        
        Args:
            features: Input features from backbone (batch_size, input_dim)
            
        Returns:
            Dictionary containing:
                - 'classes': Class predictions (batch_size, num_anchors, num_classes)
                - 'boxes': Box regression (batch_size, num_anchors, 4)
                - 'objectness': Objectness scores (batch_size, num_anchors)
        """
        batch_size = features.size(0)
        
        # Extract features
        x = self.feature_extractor(features)
        
        # Get predictions
        cls_pred = self.cls_head(x).view(batch_size, self.num_anchors, self.num_classes)
        reg_pred = self.reg_head(x).view(batch_size, self.num_anchors, 4)
        obj_pred = self.obj_head(x).view(batch_size, self.num_anchors)
        
        return {
            'classes': torch.softmax(cls_pred, dim=-1),
            'boxes': reg_pred,
            'objectness': torch.sigmoid(obj_pred),
        }