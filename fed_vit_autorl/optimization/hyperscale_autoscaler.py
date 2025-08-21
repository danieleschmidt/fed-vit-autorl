"""Advanced hyperscale auto-scaling system for federated learning."""

import asyncio
import time
import logging
import threading
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np
from collections import defaultdict, deque
import psutil
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..error_handling import with_error_handling, handle_error, ErrorCategory, ErrorSeverity

logger = logging.getLogger(__name__)


class ScalingPolicy(Enum):
    """Auto-scaling policies."""
    REACTIVE = "reactive"          # Scale based on current metrics
    PREDICTIVE = "predictive"      # Scale based on predicted demand
    HYBRID = "hybrid"              # Combine reactive and predictive
    RESOURCE_AWARE = "resource_aware"  # Scale based on resource efficiency


class ResourceType(Enum):
    """Resource types for scaling."""
    CLIENT = "client"
    SERVER = "server"
    COMPUTE = "compute"
    MEMORY = "memory"
    NETWORK = "network"
    STORAGE = "storage"


@dataclass
class ScalingMetrics:
    """Metrics for auto-scaling decisions."""
    timestamp: float = field(default_factory=time.time)
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: float = 0.0
    network_throughput: float = 0.0
    training_loss: float = 0.0
    convergence_rate: float = 0.0
    client_count: int = 0
    round_duration: float = 0.0
    communication_overhead: float = 0.0
    model_accuracy: float = 0.0
    energy_consumption: float = 0.0
    cost_efficiency: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'gpu_usage': self.gpu_usage,
            'network_throughput': self.network_throughput,
            'training_loss': self.training_loss,
            'convergence_rate': self.convergence_rate,
            'client_count': self.client_count,
            'round_duration': self.round_duration,
            'communication_overhead': self.communication_overhead,
            'model_accuracy': self.model_accuracy,
            'energy_consumption': self.energy_consumption,
            'cost_efficiency': self.cost_efficiency,
        }


@dataclass
class ScalingAction:
    """Represents a scaling action."""
    resource_type: ResourceType
    action: str  # "scale_up", "scale_down", "scale_out", "scale_in"
    target_value: int
    reason: str
    confidence: float
    estimated_impact: Dict[str, float]
    execution_time: Optional[float] = None
    success: Optional[bool] = None


class MetricsCollector:
    """Collects and aggregates system metrics for scaling decisions."""

    def __init__(self, collection_interval: float = 1.0, history_size: int = 1000):
        """Initialize metrics collector.

        Args:
            collection_interval: Interval between metric collections
            history_size: Number of historical metrics to retain
        """
        self.collection_interval = collection_interval
        self.history_size = history_size
        self.metrics_history: deque = deque(maxlen=history_size)
        self.is_collecting = False
        self.collection_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    @with_error_handling(max_retries=2, auto_recover=True)
    def start_collection(self) -> None:
        """Start metrics collection."""
        if self.is_collecting:
            logger.warning("Metrics collection already started")
            return

        self.is_collecting = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        logger.info("Started metrics collection")

    def stop_collection(self) -> None:
        """Stop metrics collection."""
        self.is_collecting = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5.0)
        logger.info("Stopped metrics collection")

    def _collection_loop(self) -> None:
        """Main collection loop."""
        while self.is_collecting:
            try:
                metrics = self._collect_current_metrics()
                with self._lock:
                    self.metrics_history.append(metrics)
                time.sleep(self.collection_interval)
            except Exception as e:
                handle_error(
                    e,
                    context={'operation': 'metrics_collection'},
                    auto_recover=True
                )
                time.sleep(self.collection_interval)

    def _collect_current_metrics(self) -> ScalingMetrics:
        """Collect current system metrics."""
        metrics = ScalingMetrics()

        # System resource metrics
        metrics.cpu_usage = psutil.cpu_percent(interval=0.1)
        memory_info = psutil.virtual_memory()
        metrics.memory_usage = memory_info.percent

        # GPU metrics (if available)
        try:
            if torch.cuda.is_available():
                metrics.gpu_usage = torch.cuda.utilization()
        except Exception:
            metrics.gpu_usage = 0.0

        # Network metrics
        try:
            net_io = psutil.net_io_counters()
            if hasattr(self, '_prev_net_io'):
                time_diff = time.time() - self._prev_net_time
                bytes_diff = (net_io.bytes_sent + net_io.bytes_recv) - self._prev_net_bytes
                metrics.network_throughput = bytes_diff / time_diff / 1024 / 1024  # MB/s
            else:
                metrics.network_throughput = 0.0

            self._prev_net_io = net_io
            self._prev_net_bytes = net_io.bytes_sent + net_io.bytes_recv
            self._prev_net_time = time.time()
        except Exception:
            metrics.network_throughput = 0.0

        return metrics

    def get_recent_metrics(self, window_size: int = 10) -> List[ScalingMetrics]:
        """Get recent metrics within window."""
        with self._lock:
            return list(self.metrics_history)[-window_size:]

    def get_metric_statistics(self, metric_name: str, window_size: int = 60) -> Dict[str, float]:
        """Get statistics for a specific metric."""
        with self._lock:
            recent_metrics = list(self.metrics_history)[-window_size:]

        if not recent_metrics:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}

        values = [getattr(m, metric_name, 0.0) for m in recent_metrics]

        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'p95': np.percentile(values, 95),
            'p99': np.percentile(values, 99),
        }


class PredictiveScaler:
    """Predictive scaling using time series forecasting."""

    def __init__(self, prediction_horizon: int = 300):  # 5 minutes
        """Initialize predictive scaler.

        Args:
            prediction_horizon: Time horizon for predictions (seconds)
        """
        self.prediction_horizon = prediction_horizon
        self.models = {}  # Trained prediction models
        self.feature_history = defaultdict(list)

    @with_error_handling(max_retries=1, auto_recover=True)
    def predict_demand(
        self,
        metrics_history: List[ScalingMetrics],
        resource_type: ResourceType,
    ) -> Tuple[float, float]:
        """Predict future resource demand.

        Args:
            metrics_history: Historical metrics
            resource_type: Type of resource to predict

        Returns:
            Tuple of (predicted_demand, confidence)
        """
        if len(metrics_history) < 10:
            return 0.0, 0.0  # Not enough data

        try:
            # Simple linear trend prediction (can be replaced with ML models)
            feature_map = {
                ResourceType.CPU: 'cpu_usage',
                ResourceType.MEMORY: 'memory_usage',
                ResourceType.COMPUTE: 'gpu_usage',
                ResourceType.NETWORK: 'network_throughput',
                ResourceType.CLIENT: 'client_count',
            }

            feature_name = feature_map.get(resource_type)
            if not feature_name:
                return 0.0, 0.0

            # Extract time series
            values = [getattr(m, feature_name, 0.0) for m in metrics_history[-60:]]
            timestamps = [m.timestamp for m in metrics_history[-60:]]

            if len(values) < 5:
                return values[-1] if values else 0.0, 0.5

            # Simple linear regression for trend
            x = np.array(range(len(values)))
            y = np.array(values)

            # Remove outliers
            q75, q25 = np.percentile(y, [75, 25])
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            mask = (y >= lower_bound) & (y <= upper_bound)

            if np.sum(mask) < 3:
                return y[-1], 0.3

            x_clean = x[mask]
            y_clean = y[mask]

            # Fit linear trend
            coeffs = np.polyfit(x_clean, y_clean, 1)
            slope, intercept = coeffs

            # Predict future value
            future_x = len(values) + (self.prediction_horizon / 60)  # Assuming 1-minute intervals
            predicted_value = slope * future_x + intercept

            # Calculate confidence based on recent variance
            recent_variance = np.var(y[-10:])
            confidence = max(0.1, 1.0 - min(1.0, recent_variance / np.mean(y)))

            return max(0.0, predicted_value), confidence

        except Exception as e:
            handle_error(
                e,
                context={'operation': 'predict_demand', 'resource_type': resource_type.value}
            )
            return 0.0, 0.0

    def update_model(self, metrics_history: List[ScalingMetrics]) -> None:
        """Update prediction models with new data."""
        # Placeholder for model training/updating
        pass


class HyperscaleAutoscaler:
    """Advanced auto-scaler for federated learning systems."""

    def __init__(
        self,
        scaling_policy: ScalingPolicy = ScalingPolicy.HYBRID,
        min_clients: int = 10,
        max_clients: int = 1000,
        target_cpu_usage: float = 70.0,
        target_memory_usage: float = 80.0,
        scale_up_threshold: float = 85.0,
        scale_down_threshold: float = 50.0,
        cooldown_period: float = 300.0,  # 5 minutes
        prediction_weight: float = 0.3,
    ):
        """Initialize hyperscale autoscaler.

        Args:
            scaling_policy: Scaling policy to use
            min_clients: Minimum number of clients
            max_clients: Maximum number of clients
            target_cpu_usage: Target CPU usage percentage
            target_memory_usage: Target memory usage percentage
            scale_up_threshold: Threshold for scaling up
            scale_down_threshold: Threshold for scaling down
            cooldown_period: Cooldown period between scaling actions
            prediction_weight: Weight for predictive scaling (0-1)
        """
        self.scaling_policy = scaling_policy
        self.min_clients = min_clients
        self.max_clients = max_clients
        self.target_cpu_usage = target_cpu_usage
        self.target_memory_usage = target_memory_usage
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.cooldown_period = cooldown_period
        self.prediction_weight = prediction_weight

        self.metrics_collector = MetricsCollector()
        self.predictive_scaler = PredictiveScaler()

        self.last_scaling_actions: Dict[ResourceType, float] = {}
        self.scaling_history: List[ScalingAction] = []
        self.current_resources = {
            ResourceType.CLIENT: min_clients,
            ResourceType.SERVER: 1,
            ResourceType.COMPUTE: 1,
        }

        self.scaling_callbacks: Dict[ResourceType, Callable] = {}
        self.is_running = False
        self.autoscaler_thread: Optional[threading.Thread] = None

        logger.info("Initialized hyperscale autoscaler")

    def register_scaling_callback(
        self,
        resource_type: ResourceType,
        callback: Callable[[ScalingAction], bool]
    ) -> None:
        """Register callback for scaling actions.

        Args:
            resource_type: Type of resource
            callback: Function to execute scaling action
        """
        self.scaling_callbacks[resource_type] = callback
        logger.info(f"Registered scaling callback for {resource_type.value}")

    @with_error_handling(max_retries=2, auto_recover=True)
    def start_autoscaling(self) -> None:
        """Start the autoscaling engine."""
        if self.is_running:
            logger.warning("Autoscaler already running")
            return

        self.is_running = True
        self.metrics_collector.start_collection()
        self.autoscaler_thread = threading.Thread(target=self._autoscaling_loop, daemon=True)
        self.autoscaler_thread.start()

        logger.info("Started hyperscale autoscaler")

    def stop_autoscaling(self) -> None:
        """Stop the autoscaling engine."""
        self.is_running = False
        self.metrics_collector.stop_collection()

        if self.autoscaler_thread:
            self.autoscaler_thread.join(timeout=10.0)

        logger.info("Stopped hyperscale autoscaler")

    def _autoscaling_loop(self) -> None:
        """Main autoscaling loop."""
        while self.is_running:
            try:
                # Get recent metrics
                metrics_history = self.metrics_collector.get_recent_metrics(window_size=60)

                if len(metrics_history) < 5:
                    time.sleep(30)  # Wait for more data
                    continue

                # Evaluate scaling decisions
                scaling_decisions = self._evaluate_scaling_decisions(metrics_history)

                # Execute scaling actions
                for action in scaling_decisions:
                    self._execute_scaling_action(action)

                time.sleep(30)  # Check every 30 seconds

            except Exception as e:
                handle_error(
                    e,
                    context={'operation': 'autoscaling_loop'},
                    auto_recover=True
                )
                time.sleep(30)

    @with_error_handling(max_retries=1, auto_recover=True)
    def _evaluate_scaling_decisions(
        self,
        metrics_history: List[ScalingMetrics]
    ) -> List[ScalingAction]:
        """Evaluate whether scaling actions are needed."""
        decisions = []
        current_time = time.time()

        if not metrics_history:
            return decisions

        latest_metrics = metrics_history[-1]

        # Check cooldown periods
        def in_cooldown(resource_type: ResourceType) -> bool:
            last_action_time = self.last_scaling_actions.get(resource_type, 0)
            return current_time - last_action_time < self.cooldown_period

        # Evaluate CPU scaling
        if not in_cooldown(ResourceType.COMPUTE):
            cpu_stats = self.metrics_collector.get_metric_statistics('cpu_usage')
            cpu_decision = self._evaluate_resource_scaling(
                current_value=latest_metrics.cpu_usage,
                stats=cpu_stats,
                resource_type=ResourceType.COMPUTE,
                metrics_history=metrics_history
            )
            if cpu_decision:
                decisions.append(cpu_decision)

        # Evaluate memory scaling
        if not in_cooldown(ResourceType.MEMORY):
            memory_stats = self.metrics_collector.get_metric_statistics('memory_usage')
            memory_decision = self._evaluate_resource_scaling(
                current_value=latest_metrics.memory_usage,
                stats=memory_stats,
                resource_type=ResourceType.MEMORY,
                metrics_history=metrics_history
            )
            if memory_decision:
                decisions.append(memory_decision)

        # Evaluate client scaling (federated learning specific)
        if not in_cooldown(ResourceType.CLIENT):
            client_decision = self._evaluate_client_scaling(metrics_history)
            if client_decision:
                decisions.append(client_decision)

        return decisions

    def _evaluate_resource_scaling(
        self,
        current_value: float,
        stats: Dict[str, float],
        resource_type: ResourceType,
        metrics_history: List[ScalingMetrics]
    ) -> Optional[ScalingAction]:
        """Evaluate scaling for a specific resource."""

        # Reactive scaling component
        reactive_score = 0.0
        if current_value > self.scale_up_threshold:
            reactive_score = 1.0  # Scale up
        elif current_value < self.scale_down_threshold:
            reactive_score = -1.0  # Scale down
        elif stats['mean'] > self.scale_up_threshold:
            reactive_score = 0.7  # Scale up based on average
        elif stats['mean'] < self.scale_down_threshold:
            reactive_score = -0.7  # Scale down based on average

        # Predictive scaling component
        predictive_score = 0.0
        if self.scaling_policy in [ScalingPolicy.PREDICTIVE, ScalingPolicy.HYBRID]:
            predicted_demand, confidence = self.predictive_scaler.predict_demand(
                metrics_history, resource_type
            )

            if predicted_demand > self.scale_up_threshold:
                predictive_score = confidence
            elif predicted_demand < self.scale_down_threshold:
                predictive_score = -confidence

        # Combine reactive and predictive scores
        if self.scaling_policy == ScalingPolicy.REACTIVE:
            final_score = reactive_score
        elif self.scaling_policy == ScalingPolicy.PREDICTIVE:
            final_score = predictive_score
        else:  # HYBRID
            final_score = (1 - self.prediction_weight) * reactive_score + \
                         self.prediction_weight * predictive_score

        # Generate scaling action
        if abs(final_score) > 0.5:  # Threshold for action
            if final_score > 0:
                action_type = "scale_up"
                target_value = self.current_resources.get(resource_type, 1) + 1
                reason = f"High {resource_type.value} usage: {current_value:.1f}%"
            else:
                action_type = "scale_down"
                target_value = max(1, self.current_resources.get(resource_type, 1) - 1)
                reason = f"Low {resource_type.value} usage: {current_value:.1f}%"

            return ScalingAction(
                resource_type=resource_type,
                action=action_type,
                target_value=target_value,
                reason=reason,
                confidence=abs(final_score),
                estimated_impact={
                    'cost_change': -0.1 if action_type == "scale_down" else 0.2,
                    'performance_change': 0.1 if action_type == "scale_up" else -0.05
                }
            )

        return None

    def _evaluate_client_scaling(
        self,
        metrics_history: List[ScalingMetrics]
    ) -> Optional[ScalingAction]:
        """Evaluate federated learning client scaling."""
        if not metrics_history:
            return None

        latest_metrics = metrics_history[-1]
        current_clients = self.current_resources.get(ResourceType.CLIENT, self.min_clients)

        # Factors for client scaling
        convergence_rate = latest_metrics.convergence_rate
        round_duration = latest_metrics.round_duration
        communication_overhead = latest_metrics.communication_overhead

        # Scale up if:
        # - Convergence is slow and we have capacity
        # - Round duration is acceptable
        # - Communication overhead is manageable
        scale_up_score = 0.0
        scale_down_score = 0.0

        if convergence_rate < 0.01 and current_clients < self.max_clients:
            scale_up_score += 0.3

        if round_duration < 300 and current_clients < self.max_clients:  # < 5 minutes
            scale_up_score += 0.2

        if communication_overhead < 0.3:
            scale_up_score += 0.2

        # Scale down if:
        # - Communication overhead is high
        # - Round duration is too long
        # - We have more than minimum clients
        if communication_overhead > 0.7 and current_clients > self.min_clients:
            scale_down_score += 0.4

        if round_duration > 600 and current_clients > self.min_clients:  # > 10 minutes
            scale_down_score += 0.3

        # Make scaling decision
        if scale_up_score > 0.4:
            target_clients = min(self.max_clients, current_clients + max(1, current_clients // 10))
            return ScalingAction(
                resource_type=ResourceType.CLIENT,
                action="scale_out",
                target_value=target_clients,
                reason=f"Slow convergence, adding clients (score: {scale_up_score:.2f})",
                confidence=scale_up_score,
                estimated_impact={
                    'convergence_improvement': 0.15,
                    'communication_overhead_increase': 0.1
                }
            )
        elif scale_down_score > 0.4:
            target_clients = max(self.min_clients, current_clients - max(1, current_clients // 20))
            return ScalingAction(
                resource_type=ResourceType.CLIENT,
                action="scale_in",
                target_value=target_clients,
                reason=f"High communication overhead, reducing clients (score: {scale_down_score:.2f})",
                confidence=scale_down_score,
                estimated_impact={
                    'convergence_impact': -0.05,
                    'communication_overhead_reduction': 0.2
                }
            )

        return None

    @with_error_handling(max_retries=2, auto_recover=True)
    def _execute_scaling_action(self, action: ScalingAction) -> None:
        """Execute a scaling action."""
        start_time = time.time()

        logger.info(
            f"Executing scaling action: {action.action} {action.resource_type.value} "
            f"to {action.target_value} (reason: {action.reason})"
        )

        try:
            # Get callback for this resource type
            callback = self.scaling_callbacks.get(action.resource_type)

            if callback:
                success = callback(action)
                action.success = success

                if success:
                    # Update current resource count
                    self.current_resources[action.resource_type] = action.target_value
                    self.last_scaling_actions[action.resource_type] = time.time()
                    logger.info(f"Successfully executed scaling action: {action.action}")
                else:
                    logger.error(f"Failed to execute scaling action: {action.action}")
            else:
                logger.warning(f"No callback registered for {action.resource_type.value}")
                action.success = False

        except Exception as e:
            action.success = False
            handle_error(
                e,
                context={
                    'operation': 'execute_scaling_action',
                    'action': action.action,
                    'resource_type': action.resource_type.value
                }
            )

        action.execution_time = time.time() - start_time
        self.scaling_history.append(action)

        # Keep history limited
        if len(self.scaling_history) > 1000:
            self.scaling_history = self.scaling_history[-800:]

    def get_scaling_statistics(self) -> Dict[str, Any]:
        """Get autoscaling statistics."""
        if not self.scaling_history:
            return {'total_actions': 0}

        successful_actions = [a for a in self.scaling_history if a.success]
        failed_actions = [a for a in self.scaling_history if not a.success]

        recent_actions = [a for a in self.scaling_history
                         if time.time() - (a.execution_time or 0) < 3600]  # Last hour

        return {
            'total_actions': len(self.scaling_history),
            'successful_actions': len(successful_actions),
            'failed_actions': len(failed_actions),
            'success_rate': len(successful_actions) / len(self.scaling_history),
            'recent_actions': len(recent_actions),
            'current_resources': dict(self.current_resources),
            'scaling_policy': self.scaling_policy.value,
            'average_execution_time': np.mean([
                a.execution_time for a in self.scaling_history
                if a.execution_time is not None
            ]) if successful_actions else 0.0,
        }

    def force_scaling_action(
        self,
        resource_type: ResourceType,
        action: str,
        target_value: int,
        reason: str = "Manual override"
    ) -> bool:
        """Force a scaling action (bypass cooldown and thresholds)."""
        scaling_action = ScalingAction(
            resource_type=resource_type,
            action=action,
            target_value=target_value,
            reason=reason,
            confidence=1.0,
            estimated_impact={}
        )

        self._execute_scaling_action(scaling_action)
        return scaling_action.success or False
