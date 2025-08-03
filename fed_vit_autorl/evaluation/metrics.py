"""Comprehensive metrics for autonomous driving and federated learning evaluation."""

import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
import time
from sklearn.metrics import average_precision_score, precision_recall_curve
import cv2


logger = logging.getLogger(__name__)


class PerceptionMetrics:
    """Metrics for perception model evaluation."""
    
    def __init__(self, num_classes: int = 20, iou_threshold: float = 0.5):
        """Initialize perception metrics.
        
        Args:
            num_classes: Number of object classes
            iou_threshold: IoU threshold for detection metrics
        """
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        
        # Metric accumulation
        self.detection_stats = defaultdict(list)
        self.segmentation_stats = defaultdict(list)
        
    def compute_detection_metrics(
        self,
        predictions: List[Dict[str, torch.Tensor]], 
        targets: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, float]:
        """Compute object detection metrics (mAP, precision, recall).
        
        Args:
            predictions: List of prediction dictionaries
            targets: List of ground truth dictionaries
            
        Returns:
            Dictionary of detection metrics
        """
        all_ious = []
        all_precisions = []
        all_recalls = []
        class_aps = []
        
        for pred, target in zip(predictions, targets):
            pred_boxes = pred.get('boxes', torch.empty(0, 4))
            pred_scores = pred.get('scores', torch.empty(0))
            pred_labels = pred.get('labels', torch.empty(0))
            
            target_boxes = target.get('boxes', torch.empty(0, 4))
            target_labels = target.get('labels', torch.empty(0))
            
            if len(pred_boxes) == 0 or len(target_boxes) == 0:
                continue
            
            # Compute IoU matrix
            ious = self._compute_iou_matrix(pred_boxes, target_boxes)
            
            # Match predictions to targets
            matches = self._match_predictions_to_targets(
                ious, pred_labels, target_labels, self.iou_threshold
            )
            
            # Compute precision and recall
            tp = matches['tp']
            fp = matches['fp'] 
            fn = matches['fn']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            all_precisions.append(precision)
            all_recalls.append(recall)
            all_ious.extend(ious.max(dim=1)[0].tolist())
        
        # Compute average precision for each class
        for class_id in range(self.num_classes):
            class_predictions = []
            class_targets = []
            
            for pred, target in zip(predictions, targets):
                # Extract class-specific predictions and targets
                pred_mask = pred.get('labels', torch.empty(0)) == class_id
                target_mask = target.get('labels', torch.empty(0)) == class_id
                
                if pred_mask.any():
                    class_predictions.extend(pred.get('scores', torch.empty(0))[pred_mask].tolist())
                if target_mask.any():
                    class_targets.extend([1] * target_mask.sum().item())
            
            if class_predictions and class_targets:
                # Pad with negative examples
                class_predictions.extend([0.0] * (len(class_predictions) - len(class_targets)))
                class_targets.extend([0] * (len(class_predictions) - len(class_targets)))
                
                ap = average_precision_score(class_targets, class_predictions)
                class_aps.append(ap)
        
        return {
            'mAP': np.mean(class_aps) if class_aps else 0.0,
            'mean_precision': np.mean(all_precisions) if all_precisions else 0.0,
            'mean_recall': np.mean(all_recalls) if all_recalls else 0.0,
            'mean_iou': np.mean(all_ious) if all_ious else 0.0,
            'detection_rate': len([p for p in all_precisions if p > 0.5]) / max(1, len(all_precisions)),
        }
    
    def compute_segmentation_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute semantic segmentation metrics (mIoU, pixel accuracy).
        
        Args:
            predictions: Predicted segmentation masks (B, H, W)
            targets: Ground truth masks (B, H, W)
            
        Returns:
            Dictionary of segmentation metrics
        """
        # Convert to numpy
        pred_np = predictions.detach().cpu().numpy().astype(int)
        target_np = targets.detach().cpu().numpy().astype(int)
        
        # Compute per-class IoU
        class_ious = []
        for class_id in range(self.num_classes):
            pred_mask = (pred_np == class_id)
            target_mask = (target_np == class_id)
            
            intersection = np.logical_and(pred_mask, target_mask).sum()
            union = np.logical_or(pred_mask, target_mask).sum()
            
            if union > 0:
                iou = intersection / union
                class_ious.append(iou)
        
        # Compute pixel accuracy
        correct_pixels = (pred_np == target_np).sum()
        total_pixels = target_np.size
        pixel_accuracy = correct_pixels / total_pixels
        
        return {
            'mIoU': np.mean(class_ious) if class_ious else 0.0,
            'pixel_accuracy': pixel_accuracy,
            'class_ious': class_ious,
        }
    
    def _compute_iou_matrix(
        self,
        boxes1: torch.Tensor,
        boxes2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute IoU matrix between two sets of boxes.
        
        Args:
            boxes1: First set of boxes (N, 4)
            boxes2: Second set of boxes (M, 4)
            
        Returns:
            IoU matrix (N, M)
        """
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        # Compute intersection
        x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
        y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
        x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
        y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        # Compute union
        union = area1[:, None] + area2 - intersection
        
        # Compute IoU
        iou = intersection / torch.clamp(union, min=1e-6)
        
        return iou
    
    def _match_predictions_to_targets(
        self,
        ious: torch.Tensor,
        pred_labels: torch.Tensor,
        target_labels: torch.Tensor,
        iou_threshold: float,
    ) -> Dict[str, int]:
        """Match predictions to ground truth targets.
        
        Args:
            ious: IoU matrix
            pred_labels: Predicted labels
            target_labels: Target labels
            iou_threshold: IoU threshold for positive matches
            
        Returns:
            Dictionary with TP, FP, FN counts
        """
        tp = 0
        fp = 0
        fn = 0
        
        matched_targets = set()
        
        # Sort predictions by IoU
        if ious.numel() > 0:
            max_ious, target_indices = ious.max(dim=1)
            
            for pred_idx, (max_iou, target_idx) in enumerate(zip(max_ious, target_indices)):
                if (max_iou >= iou_threshold and 
                    target_idx.item() not in matched_targets and
                    pred_labels[pred_idx] == target_labels[target_idx]):
                    tp += 1
                    matched_targets.add(target_idx.item())
                else:
                    fp += 1
        
        # Count unmatched targets as false negatives
        fn = len(target_labels) - len(matched_targets)
        
        return {'tp': tp, 'fp': fp, 'fn': fn}


class DrivingMetrics:
    """Metrics for driving behavior and safety evaluation."""
    
    def __init__(self):
        """Initialize driving metrics."""
        self.trajectory_data = []
        self.safety_events = []
        
    def compute_safety_metrics(
        self,
        trajectories: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Compute safety metrics from driving trajectories.
        
        Args:
            trajectories: List of trajectory dictionaries
            
        Returns:
            Dictionary of safety metrics
        """
        collision_count = 0
        hard_braking_count = 0
        lane_violations = 0
        total_distance = 0.0
        total_time = 0.0
        min_ttc_values = []
        
        for traj in trajectories:
            # Extract trajectory data
            positions = traj.get('positions', [])
            velocities = traj.get('velocities', [])
            actions = traj.get('actions', [])
            events = traj.get('safety_events', [])
            
            # Calculate distance and time
            if len(positions) > 1:
                distances = [
                    np.linalg.norm(np.array(positions[i+1]) - np.array(positions[i]))
                    for i in range(len(positions) - 1)
                ]
                total_distance += sum(distances)
                total_time += len(positions) * 0.1  # Assuming 10Hz
            
            # Count safety events
            for event in events:
                if event.get('type') == 'collision':
                    collision_count += 1
                elif event.get('type') == 'hard_braking':
                    hard_braking_count += 1
                elif event.get('type') == 'lane_violation':
                    lane_violations += 1
                elif event.get('type') == 'low_ttc':
                    min_ttc_values.append(event.get('ttc', float('inf')))
        
        # Calculate rates
        distance_km = total_distance / 1000.0
        time_hours = total_time / 3600.0
        
        return {
            'collision_rate': collision_count / max(distance_km, 0.001),
            'hard_braking_rate': hard_braking_count / max(distance_km, 0.001),
            'lane_violation_rate': lane_violations / max(distance_km, 0.001),
            'min_ttc': min(min_ttc_values) if min_ttc_values else float('inf'),
            'avg_ttc': np.mean(min_ttc_values) if min_ttc_values else float('inf'),
            'total_distance_km': distance_km,
            'total_time_hours': time_hours,
        }
    
    def compute_comfort_metrics(
        self,
        trajectories: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Compute driving comfort metrics.
        
        Args:
            trajectories: List of trajectory dictionaries
            
        Returns:
            Dictionary of comfort metrics
        """
        all_accelerations = []
        all_jerks = []
        all_lateral_accelerations = []
        all_yaw_rates = []
        
        for traj in trajectories:
            velocities = traj.get('velocities', [])
            positions = traj.get('positions', [])
            headings = traj.get('headings', [])
            
            if len(velocities) > 2:
                # Compute longitudinal acceleration
                accelerations = np.diff(velocities)
                all_accelerations.extend(accelerations)
                
                # Compute jerk (rate of acceleration change)
                if len(accelerations) > 1:
                    jerks = np.diff(accelerations)
                    all_jerks.extend(jerks)
            
            if len(positions) > 2:
                # Compute lateral acceleration (approximation)
                positions = np.array(positions)
                velocities_2d = np.diff(positions, axis=0)
                speeds = np.linalg.norm(velocities_2d, axis=1)
                
                if len(speeds) > 1:
                    # Curvature-based lateral acceleration
                    curvatures = self._compute_curvature(positions)
                    lateral_accs = speeds[:-1] ** 2 * curvatures
                    all_lateral_accelerations.extend(lateral_accs)
            
            if len(headings) > 1:
                # Compute yaw rate
                yaw_rates = np.diff(headings)
                all_yaw_rates.extend(yaw_rates)
        
        return {
            'rms_acceleration': np.sqrt(np.mean(np.array(all_accelerations) ** 2)) if all_accelerations else 0.0,
            'max_acceleration': np.max(np.abs(all_accelerations)) if all_accelerations else 0.0,
            'rms_jerk': np.sqrt(np.mean(np.array(all_jerks) ** 2)) if all_jerks else 0.0,
            'max_jerk': np.max(np.abs(all_jerks)) if all_jerks else 0.0,
            'rms_lateral_acceleration': np.sqrt(np.mean(np.array(all_lateral_accelerations) ** 2)) if all_lateral_accelerations else 0.0,
            'max_lateral_acceleration': np.max(np.abs(all_lateral_accelerations)) if all_lateral_accelerations else 0.0,
            'rms_yaw_rate': np.sqrt(np.mean(np.array(all_yaw_rates) ** 2)) if all_yaw_rates else 0.0,
            'max_yaw_rate': np.max(np.abs(all_yaw_rates)) if all_yaw_rates else 0.0,
        }
    
    def _compute_curvature(self, positions: np.ndarray) -> np.ndarray:
        """Compute path curvature from positions.
        
        Args:
            positions: Array of 2D positions (N, 2)
            
        Returns:
            Array of curvature values
        """
        if len(positions) < 3:
            return np.zeros(max(0, len(positions) - 2))
        
        # Compute first and second derivatives
        dx = np.gradient(positions[:, 0])
        dy = np.gradient(positions[:, 1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        
        # Curvature formula: |dx*ddy - dy*ddx| / (dx^2 + dy^2)^(3/2)
        numerator = np.abs(dx * ddy - dy * ddx)
        denominator = (dx**2 + dy**2)**(3/2)
        denominator = np.maximum(denominator, 1e-6)  # Avoid division by zero
        
        curvature = numerator / denominator
        return curvature[:-1]  # Remove last element to match trajectory length


class FederationMetrics:
    """Metrics for federated learning evaluation."""
    
    def __init__(self):
        """Initialize federation metrics."""
        self.round_data = []
        self.client_data = defaultdict(list)
        
    def compute_convergence_metrics(
        self,
        loss_history: List[float],
        accuracy_history: List[float],
    ) -> Dict[str, float]:
        """Compute convergence metrics for federated learning.
        
        Args:
            loss_history: History of global model loss values
            accuracy_history: History of global model accuracy values
            
        Returns:
            Dictionary of convergence metrics
        """
        if not loss_history or not accuracy_history:
            return {'convergence_rate': 0.0, 'final_loss': float('inf'), 'final_accuracy': 0.0}
        
        # Convergence rate (improvement per round)
        if len(loss_history) > 1:
            loss_improvement = loss_history[0] - loss_history[-1]
            convergence_rate = loss_improvement / len(loss_history)
        else:
            convergence_rate = 0.0
        
        # Stability (variance in recent rounds)
        recent_window = min(10, len(loss_history))
        recent_losses = loss_history[-recent_window:]
        loss_stability = 1.0 / (1.0 + np.var(recent_losses))
        
        recent_accuracies = accuracy_history[-recent_window:]
        accuracy_stability = 1.0 / (1.0 + np.var(recent_accuracies))
        
        return {
            'convergence_rate': convergence_rate,
            'loss_stability': loss_stability,
            'accuracy_stability': accuracy_stability,
            'final_loss': loss_history[-1],
            'final_accuracy': accuracy_history[-1],
            'best_accuracy': max(accuracy_history),
            'rounds_to_convergence': self._estimate_convergence_round(loss_history),
        }
    
    def compute_fairness_metrics(
        self,
        client_accuracies: Dict[str, List[float]],
        client_data_sizes: Dict[str, int],
    ) -> Dict[str, float]:
        """Compute fairness metrics across federated clients.
        
        Args:
            client_accuracies: Accuracy history for each client
            client_data_sizes: Data size for each client
            
        Returns:
            Dictionary of fairness metrics
        """
        if not client_accuracies:
            return {'fairness_score': 0.0}
        
        # Get final accuracies
        final_accuracies = []
        for client_id, acc_history in client_accuracies.items():
            if acc_history:
                final_accuracies.append(acc_history[-1])
        
        if not final_accuracies:
            return {'fairness_score': 0.0}
        
        # Statistical fairness measures
        mean_accuracy = np.mean(final_accuracies)
        std_accuracy = np.std(final_accuracies)
        min_accuracy = np.min(final_accuracies)
        max_accuracy = np.max(final_accuracies)
        
        # Fairness score (higher is more fair)
        fairness_score = 1.0 - (std_accuracy / (mean_accuracy + 1e-6))
        
        # Data distribution fairness
        if client_data_sizes:
            data_sizes = list(client_data_sizes.values())
            data_gini = self._compute_gini_coefficient(data_sizes)
        else:
            data_gini = 0.0
        
        return {
            'fairness_score': fairness_score,
            'accuracy_std': std_accuracy,
            'accuracy_range': max_accuracy - min_accuracy,
            'min_client_accuracy': min_accuracy,
            'max_client_accuracy': max_accuracy,
            'data_distribution_gini': data_gini,
        }
    
    def compute_efficiency_metrics(
        self,
        communication_costs: List[float],
        computation_times: List[float],
        participation_rates: List[float],
    ) -> Dict[str, float]:
        """Compute efficiency metrics for federated learning.
        
        Args:
            communication_costs: Communication costs per round
            computation_times: Computation times per round
            participation_rates: Client participation rates per round
            
        Returns:
            Dictionary of efficiency metrics
        """
        metrics = {}
        
        if communication_costs:
            metrics.update({
                'avg_communication_cost': np.mean(communication_costs),
                'total_communication_cost': np.sum(communication_costs),
                'communication_efficiency': 1.0 / (np.mean(communication_costs) + 1e-6),
            })
        
        if computation_times:
            metrics.update({
                'avg_computation_time': np.mean(computation_times),
                'total_computation_time': np.sum(computation_times),
                'computation_efficiency': 1.0 / (np.mean(computation_times) + 1e-6),
            })
        
        if participation_rates:
            metrics.update({
                'avg_participation_rate': np.mean(participation_rates),
                'min_participation_rate': np.min(participation_rates),
                'participation_stability': 1.0 / (1.0 + np.var(participation_rates)),
            })
        
        return metrics
    
    def _estimate_convergence_round(self, loss_history: List[float], threshold: float = 0.01) -> int:
        """Estimate the round when model converged.
        
        Args:
            loss_history: History of loss values
            threshold: Convergence threshold
            
        Returns:
            Round number when convergence occurred
        """
        if len(loss_history) < 2:
            return len(loss_history)
        
        for i in range(len(loss_history) - 1):
            improvement = loss_history[i] - loss_history[i + 1]
            relative_improvement = improvement / (loss_history[i] + 1e-6)
            
            if relative_improvement < threshold:
                return i + 1
        
        return len(loss_history)
    
    def _compute_gini_coefficient(self, values: List[float]) -> float:
        """Compute Gini coefficient for inequality measurement.
        
        Args:
            values: List of values to compute Gini coefficient for
            
        Returns:
            Gini coefficient (0 = perfect equality, 1 = perfect inequality)
        """
        if not values or len(values) < 2:
            return 0.0
        
        # Sort values
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        # Compute Gini coefficient
        cumsum = np.cumsum(sorted_values)
        gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
        
        return max(0.0, min(1.0, gini))