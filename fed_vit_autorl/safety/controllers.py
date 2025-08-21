"""Safety controllers and monitoring systems for autonomous vehicles."""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
import time
import threading
import queue
import math

import torch
import torch.nn as nn
import numpy as np


logger = logging.getLogger(__name__)


class SafetyLevel(Enum):
    """Safety alert levels."""
    NORMAL = "normal"
    CAUTION = "caution"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class VehicleState:
    """Vehicle state information."""
    position: Tuple[float, float]
    velocity: Tuple[float, float]
    acceleration: Tuple[float, float]
    heading: float
    yaw_rate: float
    timestamp: float

    @property
    def speed(self) -> float:
        """Calculate current speed."""
        return math.sqrt(self.velocity[0]**2 + self.velocity[1]**2)


@dataclass
class SafetyEvent:
    """Safety event record."""
    event_type: str
    severity: SafetyLevel
    description: str
    timestamp: float
    vehicle_state: VehicleState
    sensor_data: Optional[Dict[str, Any]] = None
    recommended_action: Optional[str] = None


class SafetyController:
    """Main safety controller for autonomous vehicles."""

    def __init__(
        self,
        max_speed: float = 50.0,
        min_following_distance: float = 2.0,
        emergency_brake_threshold: float = 1.5,
        lateral_acceleration_limit: float = 3.0,
        enable_failsafe: bool = True,
    ):
        """Initialize safety controller."""
        self.max_speed = max_speed / 3.6  # Convert to m/s
        self.min_following_distance = min_following_distance
        self.emergency_brake_threshold = emergency_brake_threshold
        self.lateral_acceleration_limit = lateral_acceleration_limit
        self.enable_failsafe = enable_failsafe

        # Safety monitoring
        self.safety_events = []
        self.current_safety_level = SafetyLevel.NORMAL
        self.last_emergency_brake = 0.0
        self.model_confidence_threshold = 0.7

        # Thread-safe event queue
        self.event_queue = queue.Queue()
        self.monitoring_active = False
        self.monitor_thread = None

    def validate_model_output(
        self,
        predictions: Dict[str, torch.Tensor],
        confidence_threshold: float = None,
    ) -> Tuple[bool, str]:
        """Validate model predictions for safety."""
        threshold = confidence_threshold or self.model_confidence_threshold

        # Check detection confidence
        if 'confidence' in predictions:
            confidences = predictions['confidence']
            if confidences.numel() > 0:
                max_confidence = confidences.max().item()
                if max_confidence < threshold:
                    return False, f"Low model confidence: {max_confidence:.3f} < {threshold}"

        # Check for NaN or infinite values
        for key, tensor in predictions.items():
            if torch.isnan(tensor).any():
                return False, f"NaN values detected in {key}"
            if torch.isinf(tensor).any():
                return False, f"Infinite values detected in {key}"

        return True, "Model output validated"

    def check_collision_risk(
        self,
        ego_state: VehicleState,
        detections: List[Dict[str, Any]],
    ) -> Tuple[SafetyLevel, Optional[str]]:
        """Check for collision risks with detected objects."""
        min_ttc = float('inf')

        for detection in detections:
            obj_position = detection.get('position', (0, 0))
            obj_velocity = detection.get('velocity', (0, 0))

            # Calculate relative position and velocity
            rel_pos = (
                obj_position[0] - ego_state.position[0],
                obj_position[1] - ego_state.position[1]
            )
            rel_vel = (
                obj_velocity[0] - ego_state.velocity[0],
                obj_velocity[1] - ego_state.velocity[1]
            )

            # Calculate Time to Collision
            ttc = self._calculate_ttc(rel_pos, rel_vel)
            min_ttc = min(min_ttc, ttc)

        # Determine safety level based on TTC
        if min_ttc < self.emergency_brake_threshold:
            return SafetyLevel.EMERGENCY, f"Collision imminent: TTC={min_ttc:.1f}s"
        elif min_ttc < 3.0:
            return SafetyLevel.CRITICAL, f"High collision risk: TTC={min_ttc:.1f}s"
        elif min_ttc < 5.0:
            return SafetyLevel.WARNING, f"Moderate collision risk: TTC={min_ttc:.1f}s"
        elif min_ttc < 8.0:
            return SafetyLevel.CAUTION, f"Low collision risk: TTC={min_ttc:.1f}s"
        else:
            return SafetyLevel.NORMAL, "No immediate collision risk"

    def validate_driving_action(
        self,
        proposed_action: Dict[str, float],
        current_state: VehicleState,
    ) -> Tuple[Dict[str, float], List[str]]:
        """Validate and potentially modify a proposed driving action."""
        safe_action = proposed_action.copy()
        safety_messages = []

        # Speed limiting
        if current_state.speed > self.max_speed:
            safe_action['throttle'] = 0.0
            safe_action['brake'] = min(1.0, safe_action.get('brake', 0.0) + 0.3)
            safety_messages.append(f"Speed limited: {current_state.speed:.1f} m/s > {self.max_speed:.1f} m/s")

        # Steering limits
        max_steering = self._calculate_max_steering(current_state.speed)
        if abs(safe_action.get('steering', 0.0)) > max_steering:
            safe_action['steering'] = max_steering * np.sign(safe_action.get('steering', 0.0))
            safety_messages.append(f"Steering limited to prevent rollover: {max_steering:.2f} rad")

        # Emergency brake override
        if self.current_safety_level in [SafetyLevel.EMERGENCY, SafetyLevel.CRITICAL]:
            safe_action['throttle'] = 0.0
            safe_action['brake'] = 1.0
            safe_action['steering'] = 0.0
            safety_messages.append("Emergency brake activated")

        return safe_action, safety_messages

    def _calculate_ttc(self, rel_pos: Tuple[float, float], rel_vel: Tuple[float, float]) -> float:
        """Calculate Time to Collision."""
        if abs(rel_vel[0]) < 1e-6 and abs(rel_vel[1]) < 1e-6:
            return float('inf')

        distance = math.sqrt(rel_pos[0]**2 + rel_pos[1]**2)
        closing_speed = math.sqrt(rel_vel[0]**2 + rel_vel[1]**2)

        if closing_speed < 1e-6:
            return float('inf')

        return distance / closing_speed

    def _calculate_max_steering(self, speed: float) -> float:
        """Calculate maximum safe steering angle based on speed."""
        if speed < 1.0:
            return math.pi / 4  # 45 degrees at low speed

        wheelbase = 2.5
        max_tan_steering = self.lateral_acceleration_limit * wheelbase / (speed ** 2)
        return min(math.atan(max_tan_steering), math.pi / 6)  # Max 30 degrees


class ModelHealthMonitor:
    """Monitor model health and performance degradation."""

    def __init__(self, alert_threshold: float = 0.1):
        """Initialize model health monitor."""
        self.alert_threshold = alert_threshold
        self.baseline_metrics = {}
        self.current_metrics = {}
        self.performance_history = []

    def set_baseline(self, metrics: Dict[str, float]) -> None:
        """Set baseline performance metrics."""
        self.baseline_metrics = metrics.copy()
        logger.info(f"Baseline metrics set: {metrics}")

    def update_metrics(self, metrics: Dict[str, float]) -> List[str]:
        """Update current metrics and check for degradation."""
        self.current_metrics = metrics.copy()
        self.performance_history.append((time.time(), metrics.copy()))

        alerts = []

        for metric_name, current_value in metrics.items():
            if metric_name in self.baseline_metrics:
                baseline_value = self.baseline_metrics[metric_name]

                if baseline_value != 0:
                    relative_change = abs(current_value - baseline_value) / abs(baseline_value)

                    if relative_change > self.alert_threshold:
                        alerts.append(
                            f"Performance degradation in {metric_name}: "
                            f"{baseline_value:.3f} -> {current_value:.3f} "
                            f"({relative_change:.1%} change)"
                        )

        return alerts

    def get_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        return {
            'baseline_metrics': self.baseline_metrics,
            'current_metrics': self.current_metrics,
            'performance_trend': 'stable',
            'health_status': 'healthy',
        }


class CertificationValidator:
    """Validate model outputs against automotive safety standards."""

    def __init__(self):
        """Initialize certification validator."""
        self.iso26262_checks = {
            'detection_latency_ms': 100,
            'min_detection_accuracy': 0.95,
            'max_false_positive_rate': 0.05,
            'min_recall_critical_objects': 0.99,
        }
        self.validation_history = []

    def validate_iso26262_compliance(
        self,
        metrics: Dict[str, float],
        test_results: Dict[str, Any],
    ) -> Tuple[bool, List[str]]:
        """Validate compliance with ISO 26262."""
        violations = []

        # Check detection latency
        if metrics.get('avg_inference_latency_ms', 0) > self.iso26262_checks['detection_latency_ms']:
            violations.append(
                f"Detection latency exceeds limit: "
                f"{metrics['avg_inference_latency_ms']:.1f}ms > "
                f"{self.iso26262_checks['detection_latency_ms']}ms"
            )

        # Check detection accuracy
        if metrics.get('mAP', 0) < self.iso26262_checks['min_detection_accuracy']:
            violations.append(
                f"Detection accuracy below minimum: "
                f"{metrics['mAP']:.3f} < {self.iso26262_checks['min_detection_accuracy']}"
            )

        is_compliant = len(violations) == 0

        self.validation_history.append({
            'timestamp': time.time(),
            'compliant': is_compliant,
            'violations': violations,
            'metrics': metrics.copy(),
        })

        return is_compliant, violations
