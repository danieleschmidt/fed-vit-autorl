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
    position: Tuple[float, float]  # x, y coordinates
    velocity: Tuple[float, float]  # vx, vy
    acceleration: Tuple[float, float]  # ax, ay
    heading: float  # radians
    yaw_rate: float  # rad/s
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
    """Main safety controller for autonomous vehicles.

    Implements multiple safety layers including:
    - Collision avoidance
    - Speed limiting
    - Lane keeping
    - Emergency braking
    - Model validation
    """

    def __init__(
        self,
        max_speed: float = 50.0,  # km/h
        min_following_distance: float = 2.0,  # seconds
        emergency_brake_threshold: float = 1.5,  # seconds TTC
        lateral_acceleration_limit: float = 3.0,  # m/s²
        enable_failsafe: bool = True,
    ):
        """Initialize safety controller.

        Args:
            max_speed: Maximum allowed speed in km/h
            min_following_distance: Minimum following time in seconds
            emergency_brake_threshold: TTC threshold for emergency braking
            lateral_acceleration_limit: Maximum lateral acceleration
            enable_failsafe: Whether to enable failsafe mode
        """
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

    def start_monitoring(self) -> None:
        """Start safety monitoring thread."""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(target=self._monitoring_loop)
            self.monitor_thread.start()
            logger.info("Safety monitoring started")

    def stop_monitoring(self) -> None:
        """Stop safety monitoring thread."""
        if self.monitoring_active:
            self.monitoring_active = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=1.0)
            logger.info("Safety monitoring stopped")

    def validate_model_output(
        self,
        predictions: Dict[str, torch.Tensor],
        confidence_threshold: float = None,
    ) -> Tuple[bool, str]:
        """Validate model predictions for safety.

        Args:
            predictions: Model predictions dictionary
            confidence_threshold: Minimum confidence threshold

        Returns:
            Tuple of (is_safe, reason)
        """
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

        # Check bounding box validity
        if 'boxes' in predictions:
            boxes = predictions['boxes']
            if boxes.numel() > 0:
                # Check for negative dimensions
                widths = boxes[:, 2] - boxes[:, 0]
                heights = boxes[:, 3] - boxes[:, 1]
                if (widths <= 0).any() or (heights <= 0).any():
                    return False, "Invalid bounding box dimensions"

        return True, "Model output validated"

    def check_collision_risk(
        self,
        ego_state: VehicleState,
        detections: List[Dict[str, Any]],
    ) -> Tuple[SafetyLevel, Optional[str]]:
        """Check for collision risks with detected objects.

        Args:
            ego_state: Current vehicle state
            detections: List of detected objects

        Returns:
            Tuple of (safety_level, description)
        """
        min_ttc = float('inf')
        closest_object = None

        for detection in detections:
            # Extract object information
            obj_position = detection.get('position', (0, 0))
            obj_velocity = detection.get('velocity', (0, 0))
            obj_bbox = detection.get('bbox', [0, 0, 0, 0])

            # Calculate relative position and velocity
            rel_pos = (
                obj_position[0] - ego_state.position[0],
                obj_position[1] - ego_state.position[1]
            )
            rel_vel = (
                obj_velocity[0] - ego_state.velocity[0],
                obj_velocity[1] - ego_state.velocity[1]
            )

            # Calculate Time to Collision (TTC)
            ttc = self._calculate_ttc(rel_pos, rel_vel)

            if ttc < min_ttc:
                min_ttc = ttc
                closest_object = detection

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
        """Validate and potentially modify a proposed driving action.

        Args:
            proposed_action: Dictionary with 'throttle', 'brake', 'steering'
            current_state: Current vehicle state

        Returns:
            Tuple of (safe_action, safety_messages)
        """
        safe_action = proposed_action.copy()
        safety_messages = []

        # Speed limiting
        if current_state.speed > self.max_speed:
            # Force braking if exceeding max speed
            safe_action['throttle'] = 0.0
            safe_action['brake'] = min(1.0, safe_action.get('brake', 0.0) + 0.3)
            safety_messages.append(f"Speed limited: {current_state.speed:.1f} m/s > {self.max_speed:.1f} m/s")

        # Steering limits to prevent rollovers
        max_steering = self._calculate_max_steering(current_state.speed)
        if abs(safe_action.get('steering', 0.0)) > max_steering:
            safe_action['steering'] = max_steering * np.sign(safe_action.get('steering', 0.0))
            safety_messages.append(f"Steering limited to prevent rollover: {max_steering:.2f} rad")

        # Throttle and brake consistency
        if safe_action.get('throttle', 0.0) > 0.1 and safe_action.get('brake', 0.0) > 0.1:
            # Prioritize braking
            safe_action['throttle'] = 0.0
            safety_messages.append("Throttle zeroed due to simultaneous brake input")

        # Emergency brake override
        current_time = time.time()
        if self.current_safety_level in [SafetyLevel.EMERGENCY, SafetyLevel.CRITICAL]:
            if current_time - self.last_emergency_brake > 0.1:  # Avoid repeated emergency braking
                safe_action['throttle'] = 0.0
                safe_action['brake'] = 1.0
                safe_action['steering'] = 0.0  # Straight ahead
                self.last_emergency_brake = current_time
                safety_messages.append("Emergency brake activated")

        return safe_action, safety_messages

    def _calculate_ttc(self, rel_pos: Tuple[float, float],
                      rel_vel: Tuple[float, float]) -> float:
        """Calculate Time to Collision."""
        if abs(rel_vel[0]) < 1e-6 and abs(rel_vel[1]) < 1e-6:
            return float('inf')

        # Simple TTC calculation (can be made more sophisticated)
        distance = math.sqrt(rel_pos[0]**2 + rel_pos[1]**2)
        closing_speed = math.sqrt(rel_vel[0]**2 + rel_vel[1]**2)

        if closing_speed < 1e-6:
            return float('inf')

        return distance / closing_speed

    def _calculate_max_steering(self, speed: float) -> float:
        """Calculate maximum safe steering angle based on speed."""
        if speed < 1.0:
            return math.pi / 4  # 45 degrees at low speed

        # Limit based on lateral acceleration
        # lat_accel = v^2 * tan(steering) / wheelbase
        # Assuming wheelbase of 2.5m
        wheelbase = 2.5
        max_tan_steering = self.lateral_acceleration_limit * wheelbase / (speed ** 2)

        return min(math.atan(max_tan_steering), math.pi / 6)  # Max 30 degrees

    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                # Process queued events
                while not self.event_queue.empty():
                    event = self.event_queue.get_nowait()
                    self._process_safety_event(event)

                time.sleep(0.01)  # 100Hz monitoring

            except Exception as e:
                logger.error(f"Safety monitoring error: {e}")

    def _process_safety_event(self, event: SafetyEvent) -> None:
        """Process a safety event."""
        self.safety_events.append(event)

        # Update current safety level
        if event.severity.value > self.current_safety_level.value:
            self.current_safety_level = event.severity

        # Log critical events
        if event.severity in [SafetyLevel.CRITICAL, SafetyLevel.EMERGENCY]:
            logger.warning(f"Safety event: {event.description}")

        # Trigger emergency protocols if needed
        if event.severity == SafetyLevel.EMERGENCY and self.enable_failsafe:
            self._activate_failsafe_mode(event)

    def _activate_failsafe_mode(self, trigger_event: SafetyEvent) -> None:
        """Activate failsafe mode."""
        logger.critical(f"FAILSAFE ACTIVATED: {trigger_event.description}")
        # In a real system, this would:
        # 1. Override all control inputs
        # 2. Apply emergency braking
        # 3. Activate hazard lights
        # 4. Send emergency signals to other vehicles
        # 5. Log detailed incident report

    def get_safety_status(self) -> Dict[str, Any]:
        """Get current safety status."""
        return {
            'safety_level': self.current_safety_level.value,
            'recent_events': [{
                'type': event.event_type,
                'severity': event.severity.value,
                'description': event.description,
                'timestamp': event.timestamp,
            } for event in self.safety_events[-10:]],  # Last 10 events
            'monitoring_active': self.monitoring_active,
            'failsafe_enabled': self.enable_failsafe,
        }


class ModelHealthMonitor:
    """Monitor model health and performance degradation."""

    def __init__(self, alert_threshold: float = 0.1):
        """Initialize model health monitor.

        Args:
            alert_threshold: Threshold for performance degradation alerts
        """
        self.alert_threshold = alert_threshold
        self.baseline_metrics = {}
        self.current_metrics = {}
        self.performance_history = []
        self.anomaly_scores = []

    def set_baseline(self, metrics: Dict[str, float]) -> None:
        """Set baseline performance metrics."""
        self.baseline_metrics = metrics.copy()
        logger.info(f"Baseline metrics set: {metrics}")

    def update_metrics(self, metrics: Dict[str, float]) -> List[str]:
        """Update current metrics and check for degradation.

        Args:
            metrics: Current performance metrics

        Returns:
            List of alert messages
        """
        self.current_metrics = metrics.copy()
        self.performance_history.append((time.time(), metrics.copy()))

        alerts = []

        # Check for significant degradation
        for metric_name, current_value in metrics.items():
            if metric_name in self.baseline_metrics:
                baseline_value = self.baseline_metrics[metric_name]

                # Calculate relative change
                if baseline_value != 0:
                    relative_change = abs(current_value - baseline_value) / abs(baseline_value)

                    if relative_change > self.alert_threshold:
                        alerts.append(
                            f"Performance degradation in {metric_name}: "
                            f"{baseline_value:.3f} -> {current_value:.3f} "
                            f"({relative_change:.1%} change)"
                        )

        # Detect anomalies using simple statistical method
        if len(self.performance_history) > 10:
            anomaly_score = self._calculate_anomaly_score(metrics)
            self.anomaly_scores.append(anomaly_score)

            if anomaly_score > 2.0:  # 2 standard deviations
                alerts.append(f"Anomalous performance detected: score={anomaly_score:.2f}")

        return alerts

    def _calculate_anomaly_score(self, current_metrics: Dict[str, float]) -> float:
        """Calculate anomaly score based on historical performance."""
        if len(self.performance_history) < 10:
            return 0.0

        # Get recent history (last 20 measurements)
        recent_history = self.performance_history[-20:]

        total_score = 0.0
        metric_count = 0

        for metric_name, current_value in current_metrics.items():
            # Extract historical values for this metric
            historical_values = [
                metrics[metric_name] for _, metrics in recent_history
                if metric_name in metrics
            ]

            if len(historical_values) >= 5:
                mean_value = np.mean(historical_values)
                std_value = np.std(historical_values)

                if std_value > 0:
                    # Z-score based anomaly detection
                    z_score = abs(current_value - mean_value) / std_value
                    total_score += z_score
                    metric_count += 1

        return total_score / max(metric_count, 1)

    def get_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        report = {
            'baseline_metrics': self.baseline_metrics,
            'current_metrics': self.current_metrics,
            'performance_trend': 'stable',  # Could implement trend analysis
            'anomaly_score': self.anomaly_scores[-1] if self.anomaly_scores else 0.0,
            'health_status': 'healthy',  # Overall health assessment
        }

        # Determine overall health status
        recent_alerts = self.update_metrics(self.current_metrics)
        if recent_alerts:
            if any('anomalous' in alert.lower() for alert in recent_alerts):
                report['health_status'] = 'degraded'
            else:
                report['health_status'] = 'warning'

        return report


class CertificationValidator:
    """Validate model outputs against automotive safety standards."""

    def __init__(self):
        """Initialize certification validator."""
        self.iso26262_checks = {
            'detection_latency_ms': 100,  # Max detection latency
            'min_detection_accuracy': 0.95,  # Minimum detection accuracy
            'max_false_positive_rate': 0.05,  # Maximum false positive rate
            'min_recall_critical_objects': 0.99,  # Minimum recall for critical objects
        }

        self.validation_history = []

    def validate_iso26262_compliance(
        self,
        metrics: Dict[str, float],
        test_results: Dict[str, Any],
    ) -> Tuple[bool, List[str]]:
        """Validate compliance with ISO 26262 functional safety standard.

        Args:
            metrics: Performance metrics
            test_results: Detailed test results

        Returns:
            Tuple of (is_compliant, violations)
        """
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

        # Check false positive rate
        fp_rate = 1.0 - metrics.get('mean_precision', 1.0)
        if fp_rate > self.iso26262_checks['max_false_positive_rate']:
            violations.append(
                f"False positive rate too high: "
                f"{fp_rate:.3f} > {self.iso26262_checks['max_false_positive_rate']}"
            )

        # Check recall for critical objects (pedestrians, cyclists)
        critical_recall = test_results.get('critical_object_recall', 0.0)
        if critical_recall < self.iso26262_checks['min_recall_critical_objects']:
            violations.append(
                f"Critical object recall too low: "
                f"{critical_recall:.3f} < {self.iso26262_checks['min_recall_critical_objects']}"
            )

        is_compliant = len(violations) == 0

        # Record validation result
        self.validation_history.append({
            'timestamp': time.time(),
            'compliant': is_compliant,
            'violations': violations,
            'metrics': metrics.copy(),
        })

        return is_compliant, violations

    def generate_certification_report(self) -> str:
        """Generate certification compliance report."""
        if not self.validation_history:
            return "No validation data available"

        recent_validations = self.validation_history[-10:]  # Last 10 validations
        compliant_count = sum(1 for v in recent_validations if v['compliant'])
        compliance_rate = compliant_count / len(recent_validations)

        report = [
            "=== ISO 26262 Compliance Report ===",
            f"Recent compliance rate: {compliance_rate:.1%} ({compliant_count}/{len(recent_validations)})",
            "",
            "Requirements:",
        ]

        for requirement, threshold in self.iso26262_checks.items():
            report.append(f"  {requirement}: {threshold}")

        if recent_validations:
            latest = recent_validations[-1]
            report.extend([
                "",
                "Latest validation:",
                f"  Status: {'COMPLIANT' if latest['compliant'] else 'NON-COMPLIANT'}",
            ])

            if latest['violations']:
                report.append("  Violations:")
                for violation in latest['violations']:
                    report.append(f"    - {violation}")

        return "\
".join(report)
