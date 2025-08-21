"""Object detection head for ViT backbone."""

from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DetectionHead(nn.Module):
    """Object detection head for autonomous driving scenarios.

    Predicts bounding boxes and classes for objects like vehicles,
    pedestrians, traffic signs, etc. Uses YOLO-style detection.

    Args:
        input_dim: Input feature dimension from backbone
        num_classes: Number of object classes (default: automotive classes)
        num_anchors: Number of anchor boxes per location
        img_size: Input image size for coordinate normalization
        class_names: Optional list of class names
    """

    # Automotive object classes
    DEFAULT_CLASSES = [
        'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'pedestrian',
        'traffic_light', 'traffic_sign', 'construction', 'animal',
        'emergency_vehicle', 'cyclist', 'road_barrier', 'work_zone',
        'parked_vehicle', 'moving_vehicle', 'vulnerable_road_user',
        'infrastructure', 'weather_hazard', 'other'
    ]

    def __init__(
        self,
        input_dim: int = 768,
        num_classes: int = 20,
        num_anchors: int = 9,
        img_size: int = 384,
        class_names: Optional[List[str]] = None,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.img_size = img_size
        self.class_names = class_names or self.DEFAULT_CLASSES[:num_classes]

        # Anchor generation
        self.anchors = self._generate_anchors()

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

        # Confidence head for detection confidence
        self.conf_head = nn.Linear(256, num_anchors)

        self._init_weights()

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

        # Confidence prediction
        conf_pred = self.conf_head(x).view(batch_size, self.num_anchors)

        return {
            'classes': torch.softmax(cls_pred, dim=-1),
            'boxes': self._decode_boxes(reg_pred),
            'objectness': torch.sigmoid(obj_pred),
            'confidence': torch.sigmoid(conf_pred),
        }

    def _generate_anchors(self) -> torch.Tensor:
        """Generate anchor boxes for different scales and aspect ratios."""
        # Different scales for multi-scale detection
        scales = [0.1, 0.3, 0.5]  # Relative to image size
        aspect_ratios = [0.5, 1.0, 2.0]  # Width/height ratios

        anchors = []
        for scale in scales:
            for ratio in aspect_ratios:
                w = scale * math.sqrt(ratio)
                h = scale / math.sqrt(ratio)
                anchors.append([w, h])

        return torch.tensor(anchors, dtype=torch.float32)

    def _decode_boxes(self, box_preds: torch.Tensor) -> torch.Tensor:
        """Decode box predictions relative to anchors.

        Args:
            box_preds: Raw box predictions (batch_size, num_anchors, 4)

        Returns:
            Decoded boxes in (x, y, w, h) format
        """
        # Apply sigmoid to get relative coordinates [0, 1]
        decoded = torch.sigmoid(box_preds)

        # Scale to image coordinates
        decoded = decoded * self.img_size

        return decoded

    def _init_weights(self) -> None:
        """Initialize layer weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def post_process(self, predictions: Dict[str, torch.Tensor],
                    conf_threshold: float = 0.5,
                    nms_threshold: float = 0.4) -> List[Dict[str, torch.Tensor]]:
        """Post-process predictions with NMS.

        Args:
            predictions: Raw predictions from forward pass
            conf_threshold: Confidence threshold for filtering
            nms_threshold: IoU threshold for NMS

        Returns:
            List of processed detections per batch item
        """
        batch_size = predictions['boxes'].size(0)
        results = []

        for i in range(batch_size):
            # Get predictions for this batch item
            boxes = predictions['boxes'][i]
            classes = predictions['classes'][i]
            objectness = predictions['objectness'][i]
            confidence = predictions['confidence'][i]

            # Combine objectness and confidence
            scores = objectness * confidence.max(dim=-1)[0]
            class_ids = classes.argmax(dim=-1)

            # Filter by confidence
            valid_mask = scores > conf_threshold
            if not valid_mask.any():
                results.append({'boxes': torch.empty(0, 4),
                              'scores': torch.empty(0),
                              'classes': torch.empty(0, dtype=torch.long)})
                continue

            filtered_boxes = boxes[valid_mask]
            filtered_scores = scores[valid_mask]
            filtered_classes = class_ids[valid_mask]

            # Apply NMS
            keep_indices = self._nms(filtered_boxes, filtered_scores, nms_threshold)

            results.append({
                'boxes': filtered_boxes[keep_indices],
                'scores': filtered_scores[keep_indices],
                'classes': filtered_classes[keep_indices],
            })

        return results

    def _nms(self, boxes: torch.Tensor, scores: torch.Tensor,
            threshold: float) -> torch.Tensor:
        """Non-Maximum Suppression.

        Args:
            boxes: Bounding boxes (N, 4)
            scores: Confidence scores (N,)
            threshold: IoU threshold

        Returns:
            Indices of boxes to keep
        """
        if boxes.numel() == 0:
            return torch.empty(0, dtype=torch.long)

        # Convert to corner format for IoU calculation
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2

        areas = (x2 - x1) * (y2 - y1)
        _, order = scores.sort(descending=True)

        keep = []
        while order.numel() > 0:
            if order.numel() == 1:
                keep.append(order.item())
                break

            i = order[0].item()
            keep.append(i)

            # Calculate IoU
            xx1 = torch.max(x1[i], x1[order[1:]])
            yy1 = torch.max(y1[i], y1[order[1:]])
            xx2 = torch.min(x2[i], x2[order[1:]])
            yy2 = torch.min(y2[i], y2[order[1:]])

            w = torch.clamp(xx2 - xx1, min=0)
            h = torch.clamp(yy2 - yy1, min=0)
            intersection = w * h

            iou = intersection / (areas[i] + areas[order[1:]] - intersection)

            # Keep boxes with IoU less than threshold
            order = order[1:][iou <= threshold]

        return torch.tensor(keep, dtype=torch.long)
