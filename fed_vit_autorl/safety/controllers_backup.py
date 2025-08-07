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
            safe_action['steering'] = max_steering * np.sign(safe_action.get('steering', 0.0))\n            safety_messages.append(f"Steering limited to prevent rollover: {max_steering:.2f} rad")
        \n        # Throttle and brake consistency\n        if safe_action.get('throttle', 0.0) > 0.1 and safe_action.get('brake', 0.0) > 0.1:\n            # Prioritize braking\n            safe_action['throttle'] = 0.0\n            safety_messages.append(\"Throttle zeroed due to simultaneous brake input\")\n        \n        # Emergency brake override\n        current_time = time.time()\n        if self.current_safety_level in [SafetyLevel.EMERGENCY, SafetyLevel.CRITICAL]:\n            if current_time - self.last_emergency_brake > 0.1:  # Avoid repeated emergency braking\n                safe_action['throttle'] = 0.0\n                safe_action['brake'] = 1.0\n                safe_action['steering'] = 0.0  # Straight ahead\n                self.last_emergency_brake = current_time\n                safety_messages.append(\"Emergency brake activated\")\n        \n        return safe_action, safety_messages\n    \n    def _calculate_ttc(self, rel_pos: Tuple[float, float], \n                      rel_vel: Tuple[float, float]) -> float:\n        \"\"\"Calculate Time to Collision.\"\"\"\n        if abs(rel_vel[0]) < 1e-6 and abs(rel_vel[1]) < 1e-6:\n            return float('inf')\n        \n        # Simple TTC calculation (can be made more sophisticated)\n        distance = math.sqrt(rel_pos[0]**2 + rel_pos[1]**2)\n        closing_speed = math.sqrt(rel_vel[0]**2 + rel_vel[1]**2)\n        \n        if closing_speed < 1e-6:\n            return float('inf')\n        \n        return distance / closing_speed\n    \n    def _calculate_max_steering(self, speed: float) -> float:\n        \"\"\"Calculate maximum safe steering angle based on speed.\"\"\"\n        if speed < 1.0:\n            return math.pi / 4  # 45 degrees at low speed\n        \n        # Limit based on lateral acceleration\n        # lat_accel = v^2 * tan(steering) / wheelbase\n        # Assuming wheelbase of 2.5m\n        wheelbase = 2.5\n        max_tan_steering = self.lateral_acceleration_limit * wheelbase / (speed ** 2)\n        \n        return min(math.atan(max_tan_steering), math.pi / 6)  # Max 30 degrees\n    \n    def _monitoring_loop(self) -> None:\n        \"\"\"Background monitoring loop.\"\"\"\n        while self.monitoring_active:\n            try:\n                # Process queued events\n                while not self.event_queue.empty():\n                    event = self.event_queue.get_nowait()\n                    self._process_safety_event(event)\n                \n                time.sleep(0.01)  # 100Hz monitoring\n            \n            except Exception as e:\n                logger.error(f\"Safety monitoring error: {e}\")\n    \n    def _process_safety_event(self, event: SafetyEvent) -> None:\n        \"\"\"Process a safety event.\"\"\"\n        self.safety_events.append(event)\n        \n        # Update current safety level\n        if event.severity.value > self.current_safety_level.value:\n            self.current_safety_level = event.severity\n        \n        # Log critical events\n        if event.severity in [SafetyLevel.CRITICAL, SafetyLevel.EMERGENCY]:\n            logger.warning(f\"Safety event: {event.description}\")\n        \n        # Trigger emergency protocols if needed\n        if event.severity == SafetyLevel.EMERGENCY and self.enable_failsafe:\n            self._activate_failsafe_mode(event)\n    \n    def _activate_failsafe_mode(self, trigger_event: SafetyEvent) -> None:\n        \"\"\"Activate failsafe mode.\"\"\"\n        logger.critical(f\"FAILSAFE ACTIVATED: {trigger_event.description}\")\n        # In a real system, this would:\n        # 1. Override all control inputs\n        # 2. Apply emergency braking\n        # 3. Activate hazard lights\n        # 4. Send emergency signals to other vehicles\n        # 5. Log detailed incident report\n    \n    def get_safety_status(self) -> Dict[str, Any]:\n        \"\"\"Get current safety status.\"\"\"\n        return {\n            'safety_level': self.current_safety_level.value,\n            'recent_events': [{\n                'type': event.event_type,\n                'severity': event.severity.value,\n                'description': event.description,\n                'timestamp': event.timestamp,\n            } for event in self.safety_events[-10:]],  # Last 10 events\n            'monitoring_active': self.monitoring_active,\n            'failsafe_enabled': self.enable_failsafe,\n        }


class ModelHealthMonitor:
    \"\"\"Monitor model health and performance degradation.\"\"\"\n    \n    def __init__(self, alert_threshold: float = 0.1):\n        \"\"\"Initialize model health monitor.\n        \n        Args:\n            alert_threshold: Threshold for performance degradation alerts\n        \"\"\"\n        self.alert_threshold = alert_threshold\n        self.baseline_metrics = {}\n        self.current_metrics = {}\n        self.performance_history = []\n        self.anomaly_scores = []\n        \n    def set_baseline(self, metrics: Dict[str, float]) -> None:\n        \"\"\"Set baseline performance metrics.\"\"\"\n        self.baseline_metrics = metrics.copy()\n        logger.info(f\"Baseline metrics set: {metrics}\")\n    \n    def update_metrics(self, metrics: Dict[str, float]) -> List[str]:\n        \"\"\"Update current metrics and check for degradation.\n        \n        Args:\n            metrics: Current performance metrics\n            \n        Returns:\n            List of alert messages\n        \"\"\"\n        self.current_metrics = metrics.copy()\n        self.performance_history.append((time.time(), metrics.copy()))\n        \n        alerts = []\n        \n        # Check for significant degradation\n        for metric_name, current_value in metrics.items():\n            if metric_name in self.baseline_metrics:\n                baseline_value = self.baseline_metrics[metric_name]\n                \n                # Calculate relative change\n                if baseline_value != 0:\n                    relative_change = abs(current_value - baseline_value) / abs(baseline_value)\n                    \n                    if relative_change > self.alert_threshold:\n                        alerts.append(\n                            f\"Performance degradation in {metric_name}: \"\n                            f\"{baseline_value:.3f} -> {current_value:.3f} \"\n                            f\"({relative_change:.1%} change)\"\n                        )\n        \n        # Detect anomalies using simple statistical method\n        if len(self.performance_history) > 10:\n            anomaly_score = self._calculate_anomaly_score(metrics)\n            self.anomaly_scores.append(anomaly_score)\n            \n            if anomaly_score > 2.0:  # 2 standard deviations\n                alerts.append(f\"Anomalous performance detected: score={anomaly_score:.2f}\")\n        \n        return alerts\n    \n    def _calculate_anomaly_score(self, current_metrics: Dict[str, float]) -> float:\n        \"\"\"Calculate anomaly score based on historical performance.\"\"\"\n        if len(self.performance_history) < 10:\n            return 0.0\n        \n        # Get recent history (last 20 measurements)\n        recent_history = self.performance_history[-20:]\n        \n        total_score = 0.0\n        metric_count = 0\n        \n        for metric_name, current_value in current_metrics.items():\n            # Extract historical values for this metric\n            historical_values = [\n                metrics[metric_name] for _, metrics in recent_history\n                if metric_name in metrics\n            ]\n            \n            if len(historical_values) >= 5:\n                mean_value = np.mean(historical_values)\n                std_value = np.std(historical_values)\n                \n                if std_value > 0:\n                    # Z-score based anomaly detection\n                    z_score = abs(current_value - mean_value) / std_value\n                    total_score += z_score\n                    metric_count += 1\n        \n        return total_score / max(metric_count, 1)\n    \n    def get_health_report(self) -> Dict[str, Any]:\n        \"\"\"Generate comprehensive health report.\"\"\"\n        report = {\n            'baseline_metrics': self.baseline_metrics,\n            'current_metrics': self.current_metrics,\n            'performance_trend': 'stable',  # Could implement trend analysis\n            'anomaly_score': self.anomaly_scores[-1] if self.anomaly_scores else 0.0,\n            'health_status': 'healthy',  # Overall health assessment\n        }\n        \n        # Determine overall health status\n        recent_alerts = self.update_metrics(self.current_metrics)\n        if recent_alerts:\n            if any('anomalous' in alert.lower() for alert in recent_alerts):\n                report['health_status'] = 'degraded'\n            else:\n                report['health_status'] = 'warning'\n        \n        return report


class CertificationValidator:
    \"\"\"Validate model outputs against automotive safety standards.\"\"\"\n    \n    def __init__(self):\n        \"\"\"Initialize certification validator.\"\"\"\n        self.iso26262_checks = {\n            'detection_latency_ms': 100,  # Max detection latency\n            'min_detection_accuracy': 0.95,  # Minimum detection accuracy\n            'max_false_positive_rate': 0.05,  # Maximum false positive rate\n            'min_recall_critical_objects': 0.99,  # Minimum recall for critical objects\n        }\n        \n        self.validation_history = []\n    \n    def validate_iso26262_compliance(\n        self,\n        metrics: Dict[str, float],\n        test_results: Dict[str, Any],\n    ) -> Tuple[bool, List[str]]:\n        \"\"\"Validate compliance with ISO 26262 functional safety standard.\n        \n        Args:\n            metrics: Performance metrics\n            test_results: Detailed test results\n            \n        Returns:\n            Tuple of (is_compliant, violations)\n        \"\"\"\n        violations = []\n        \n        # Check detection latency\n        if metrics.get('avg_inference_latency_ms', 0) > self.iso26262_checks['detection_latency_ms']:\n            violations.append(\n                f\"Detection latency exceeds limit: \"\n                f\"{metrics['avg_inference_latency_ms']:.1f}ms > \"\n                f\"{self.iso26262_checks['detection_latency_ms']}ms\"\n            )\n        \n        # Check detection accuracy\n        if metrics.get('mAP', 0) < self.iso26262_checks['min_detection_accuracy']:\n            violations.append(\n                f\"Detection accuracy below minimum: \"\n                f\"{metrics['mAP']:.3f} < {self.iso26262_checks['min_detection_accuracy']}\"\n            )\n        \n        # Check false positive rate\n        fp_rate = 1.0 - metrics.get('mean_precision', 1.0)\n        if fp_rate > self.iso26262_checks['max_false_positive_rate']:\n            violations.append(\n                f\"False positive rate too high: \"\n                f\"{fp_rate:.3f} > {self.iso26262_checks['max_false_positive_rate']}\"\n            )\n        \n        # Check recall for critical objects (pedestrians, cyclists)\n        critical_recall = test_results.get('critical_object_recall', 0.0)\n        if critical_recall < self.iso26262_checks['min_recall_critical_objects']:\n            violations.append(\n                f\"Critical object recall too low: \"\n                f\"{critical_recall:.3f} < {self.iso26262_checks['min_recall_critical_objects']}\"\n            )\n        \n        is_compliant = len(violations) == 0\n        \n        # Record validation result\n        self.validation_history.append({\n            'timestamp': time.time(),\n            'compliant': is_compliant,\n            'violations': violations,\n            'metrics': metrics.copy(),\n        })\n        \n        return is_compliant, violations\n    \n    def generate_certification_report(self) -> str:\n        \"\"\"Generate certification compliance report.\"\"\"\n        if not self.validation_history:\n            return \"No validation data available\"\n        \n        recent_validations = self.validation_history[-10:]  # Last 10 validations\n        compliant_count = sum(1 for v in recent_validations if v['compliant'])\n        compliance_rate = compliant_count / len(recent_validations)\n        \n        report = [\n            \"=== ISO 26262 Compliance Report ===\",\n            f\"Recent compliance rate: {compliance_rate:.1%} ({compliant_count}/{len(recent_validations)})\",\n            \"\",\n            \"Requirements:\",\n        ]\n        \n        for requirement, threshold in self.iso26262_checks.items():\n            report.append(f\"  {requirement}: {threshold}\")\n        \n        if recent_validations:\n            latest = recent_validations[-1]\n            report.extend([\n                \"\",\n                \"Latest validation:\",\n                f\"  Status: {'COMPLIANT' if latest['compliant'] else 'NON-COMPLIANT'}\",\n            ])\n            \n            if latest['violations']:\n                report.append(\"  Violations:\")\n                for violation in latest['violations']:\n                    report.append(f\"    - {violation}\")\n        \n        return \"\\n\".join(report)