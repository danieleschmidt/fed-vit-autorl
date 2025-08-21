"""Auto-scaling and load balancing for federated learning systems."""

import time
import logging
import threading
import queue
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import statistics
from collections import defaultdict, deque
import hashlib

logger = logging.getLogger(__name__)


class ScalingAction(Enum):
    """Auto-scaling actions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NO_ACTION = "no_action"


@dataclass
class LoadMetrics:
    """Load metrics for auto-scaling decisions."""
    cpu_usage: float
    memory_usage: float
    request_rate: float
    response_time: float
    active_connections: int
    queue_length: int
    timestamp: float


@dataclass
class ScalingEvent:
    """Auto-scaling event record."""
    timestamp: float
    action: ScalingAction
    trigger_metric: str
    metric_value: float
    threshold: float
    current_instances: int
    target_instances: int
    reason: str


class AutoScaler:
    """Automatic scaling system for federated learning components."""

    def __init__(
        self,
        min_instances: int = 1,
        max_instances: int = 10,
        target_cpu_usage: float = 70.0,
        target_memory_usage: float = 80.0,
        target_response_time: float = 1.0,
        scale_up_cooldown: float = 300.0,  # 5 minutes
        scale_down_cooldown: float = 600.0,  # 10 minutes
        scale_up_threshold: float = 1.2,  # 20% above target
        scale_down_threshold: float = 0.8,  # 20% below target
        evaluation_window: int = 10,  # Number of metrics to consider
    ):
        """Initialize auto-scaler.

        Args:
            min_instances: Minimum number of instances
            max_instances: Maximum number of instances
            target_cpu_usage: Target CPU usage percentage
            target_memory_usage: Target memory usage percentage
            target_response_time: Target response time in seconds
            scale_up_cooldown: Cooldown period after scale up (seconds)
            scale_down_cooldown: Cooldown period after scale down (seconds)
            scale_up_threshold: Multiplier for scale up trigger
            scale_down_threshold: Multiplier for scale down trigger
            evaluation_window: Number of recent metrics to evaluate
        """
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.target_cpu_usage = target_cpu_usage
        self.target_memory_usage = target_memory_usage
        self.target_response_time = target_response_time
        self.scale_up_cooldown = scale_up_cooldown
        self.scale_down_cooldown = scale_down_cooldown
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.evaluation_window = evaluation_window

        self.current_instances = min_instances
        self.metrics_history: deque = deque(maxlen=evaluation_window * 2)
        self.scaling_history: List[ScalingEvent] = []
        self.last_scale_up_time = 0.0
        self.last_scale_down_time = 0.0

        self.scaling_callbacks: List[Callable[[ScalingEvent], None]] = []
        self._lock = threading.Lock()

        logger.info(f"Initialized auto-scaler (min={min_instances}, max={max_instances})")

    def add_scaling_callback(self, callback: Callable[[ScalingEvent], None]) -> None:
        """Add callback for scaling events.

        Args:
            callback: Function to call when scaling occurs
        """
        self.scaling_callbacks.append(callback)

    def record_metrics(self, metrics: LoadMetrics) -> None:
        """Record load metrics for scaling decisions.

        Args:
            metrics: Current load metrics
        """
        with self._lock:
            self.metrics_history.append(metrics)

            # Trigger evaluation if we have enough metrics
            if len(self.metrics_history) >= self.evaluation_window:
                self._evaluate_scaling()

    def _evaluate_scaling(self) -> Optional[ScalingEvent]:
        """Evaluate whether scaling is needed."""
        if len(self.metrics_history) < self.evaluation_window:
            return None

        recent_metrics = list(self.metrics_history)[-self.evaluation_window:]
        current_time = time.time()

        # Calculate average metrics
        avg_cpu = statistics.mean(m.cpu_usage for m in recent_metrics)
        avg_memory = statistics.mean(m.memory_usage for m in recent_metrics)
        avg_response_time = statistics.mean(m.response_time for m in recent_metrics)
        avg_queue_length = statistics.mean(m.queue_length for m in recent_metrics)

        # Determine if scaling is needed
        scaling_decision = self._should_scale(
            avg_cpu, avg_memory, avg_response_time, avg_queue_length, current_time
        )

        if scaling_decision[0] != ScalingAction.NO_ACTION:
            scaling_event = self._execute_scaling(scaling_decision, current_time)
            return scaling_event

        return None

    def _should_scale(
        self,
        avg_cpu: float,
        avg_memory: float,
        avg_response_time: float,
        avg_queue_length: float,
        current_time: float,
    ) -> Tuple[ScalingAction, str, float, float]:
        """Determine if scaling is needed.

        Returns:
            Tuple of (action, metric_name, metric_value, threshold)
        """
        # Check cooldown periods
        if (current_time - self.last_scale_up_time) < self.scale_up_cooldown:
            if avg_cpu > self.target_cpu_usage * self.scale_up_threshold:
                return (ScalingAction.NO_ACTION, "cpu_cooldown", avg_cpu, self.target_cpu_usage)

        if (current_time - self.last_scale_down_time) < self.scale_down_cooldown:
            if avg_cpu < self.target_cpu_usage * self.scale_down_threshold:
                return (ScalingAction.NO_ACTION, "cpu_cooldown", avg_cpu, self.target_cpu_usage)

        # Check scale up conditions
        if self.current_instances < self.max_instances:
            if avg_cpu > self.target_cpu_usage * self.scale_up_threshold:
                return (ScalingAction.SCALE_UP, "cpu_usage", avg_cpu, self.target_cpu_usage * self.scale_up_threshold)

            if avg_memory > self.target_memory_usage * self.scale_up_threshold:
                return (ScalingAction.SCALE_UP, "memory_usage", avg_memory, self.target_memory_usage * self.scale_up_threshold)

            if avg_response_time > self.target_response_time * self.scale_up_threshold:
                return (ScalingAction.SCALE_UP, "response_time", avg_response_time, self.target_response_time * self.scale_up_threshold)

            if avg_queue_length > 10:  # Arbitrary queue length threshold
                return (ScalingAction.SCALE_UP, "queue_length", avg_queue_length, 10.0)

        # Check scale down conditions
        if self.current_instances > self.min_instances:
            if (avg_cpu < self.target_cpu_usage * self.scale_down_threshold and
                avg_memory < self.target_memory_usage * self.scale_down_threshold and
                avg_response_time < self.target_response_time * self.scale_down_threshold and
                avg_queue_length < 5):
                return (ScalingAction.SCALE_DOWN, "all_metrics_low", avg_cpu, self.target_cpu_usage * self.scale_down_threshold)

        return (ScalingAction.NO_ACTION, "no_trigger", 0.0, 0.0)

    def _execute_scaling(
        self,
        scaling_decision: Tuple[ScalingAction, str, float, float],
        current_time: float,
    ) -> ScalingEvent:
        """Execute scaling action.

        Args:
            scaling_decision: Scaling decision tuple
            current_time: Current timestamp

        Returns:
            Scaling event record
        """
        action, metric_name, metric_value, threshold = scaling_decision

        previous_instances = self.current_instances

        if action == ScalingAction.SCALE_UP:
            self.current_instances = min(self.current_instances + 1, self.max_instances)
            self.last_scale_up_time = current_time
            reason = f"High {metric_name}: {metric_value:.2f} > {threshold:.2f}"

        elif action == ScalingAction.SCALE_DOWN:
            self.current_instances = max(self.current_instances - 1, self.min_instances)
            self.last_scale_down_time = current_time
            reason = f"Low resource utilization across all metrics"

        else:
            reason = "No scaling needed"

        scaling_event = ScalingEvent(
            timestamp=current_time,
            action=action,
            trigger_metric=metric_name,
            metric_value=metric_value,
            threshold=threshold,
            current_instances=previous_instances,
            target_instances=self.current_instances,
            reason=reason,
        )

        self.scaling_history.append(scaling_event)

        # Keep only last 100 scaling events
        if len(self.scaling_history) > 100:
            self.scaling_history = self.scaling_history[-100:]

        # Notify callbacks
        for callback in self.scaling_callbacks:
            try:
                callback(scaling_event)
            except Exception as e:
                logger.error(f"Scaling callback failed: {e}")

        logger.info(f"Scaling executed: {action.value} from {previous_instances} to {self.current_instances} instances")
        return scaling_event

    def get_current_instances(self) -> int:
        """Get current number of instances."""
        return self.current_instances

    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get auto-scaling statistics."""
        if not self.scaling_history:
            return {
                "total_scaling_events": 0,
                "current_instances": self.current_instances,
            }

        scale_ups = sum(1 for event in self.scaling_history if event.action == ScalingAction.SCALE_UP)
        scale_downs = sum(1 for event in self.scaling_history if event.action == ScalingAction.SCALE_DOWN)

        recent_events = [e for e in self.scaling_history if time.time() - e.timestamp < 3600]  # Last hour

        return {
            "total_scaling_events": len(self.scaling_history),
            "scale_ups": scale_ups,
            "scale_downs": scale_downs,
            "current_instances": self.current_instances,
            "recent_events": len(recent_events),
            "avg_instances": statistics.mean(e.target_instances for e in self.scaling_history),
            "last_scaling_time": self.scaling_history[-1].timestamp if self.scaling_history else 0,
        }


class LoadBalancer:
    """Load balancer for distributing requests across instances."""

    def __init__(
        self,
        algorithm: str = "round_robin",
        health_check_interval: float = 30.0,
        max_retries: int = 3,
        timeout: float = 5.0,
    ):
        """Initialize load balancer.

        Args:
            algorithm: Load balancing algorithm ("round_robin", "least_connections", "weighted_round_robin")
            health_check_interval: Health check interval in seconds
            max_retries: Maximum retries for failed requests
            timeout: Request timeout in seconds
        """
        self.algorithm = algorithm
        self.health_check_interval = health_check_interval
        self.max_retries = max_retries
        self.timeout = timeout

        self.instances: Dict[str, Dict[str, Any]] = {}
        self.current_index = 0
        self.request_counts: Dict[str, int] = defaultdict(int)
        self.response_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

        self._lock = threading.Lock()
        self._health_check_thread = None
        self._stop_health_checks = False

        logger.info(f"Initialized load balancer with {algorithm} algorithm")

    def add_instance(
        self,
        instance_id: str,
        endpoint: str,
        weight: float = 1.0,
        health_check_url: Optional[str] = None,
    ) -> None:
        """Add an instance to the load balancer.

        Args:
            instance_id: Unique instance identifier
            endpoint: Instance endpoint URL
            weight: Instance weight for weighted algorithms
            health_check_url: Optional health check URL
        """
        with self._lock:
            self.instances[instance_id] = {
                "endpoint": endpoint,
                "weight": weight,
                "healthy": True,
                "connections": 0,
                "total_requests": 0,
                "failed_requests": 0,
                "last_request_time": 0,
                "health_check_url": health_check_url or endpoint,
                "added_time": time.time(),
            }

        logger.info(f"Added instance {instance_id} to load balancer")

        # Start health checks if this is the first instance
        if len(self.instances) == 1 and not self._health_check_thread:
            self._start_health_checks()

    def remove_instance(self, instance_id: str) -> None:
        """Remove an instance from the load balancer.

        Args:
            instance_id: Instance identifier to remove
        """
        with self._lock:
            if instance_id in self.instances:
                del self.instances[instance_id]
                if instance_id in self.request_counts:
                    del self.request_counts[instance_id]
                if instance_id in self.response_times:
                    del self.response_times[instance_id]

                logger.info(f"Removed instance {instance_id} from load balancer")

    def get_next_instance(self) -> Optional[str]:
        """Get the next instance to route request to.

        Returns:
            Instance ID or None if no healthy instances
        """
        with self._lock:
            healthy_instances = [
                instance_id for instance_id, info in self.instances.items()
                if info["healthy"]
            ]

            if not healthy_instances:
                return None

            if self.algorithm == "round_robin":
                return self._round_robin_selection(healthy_instances)
            elif self.algorithm == "least_connections":
                return self._least_connections_selection(healthy_instances)
            elif self.algorithm == "weighted_round_robin":
                return self._weighted_round_robin_selection(healthy_instances)
            else:
                # Default to round robin
                return self._round_robin_selection(healthy_instances)

    def _round_robin_selection(self, healthy_instances: List[str]) -> str:
        """Round robin instance selection."""
        if not healthy_instances:
            return None

        selected = healthy_instances[self.current_index % len(healthy_instances)]
        self.current_index += 1
        return selected

    def _least_connections_selection(self, healthy_instances: List[str]) -> str:
        """Select instance with least connections."""
        if not healthy_instances:
            return None

        min_connections = float('inf')
        selected_instance = None

        for instance_id in healthy_instances:
            connections = self.instances[instance_id]["connections"]
            if connections < min_connections:
                min_connections = connections
                selected_instance = instance_id

        return selected_instance

    def _weighted_round_robin_selection(self, healthy_instances: List[str]) -> str:
        """Weighted round robin selection."""
        if not healthy_instances:
            return None

        # Calculate total weight
        total_weight = sum(self.instances[instance_id]["weight"] for instance_id in healthy_instances)

        # Select based on weight
        current_weight = 0
        target_weight = (self.current_index % int(total_weight * 100)) / 100.0

        for instance_id in healthy_instances:
            current_weight += self.instances[instance_id]["weight"]
            if current_weight >= target_weight:
                self.current_index += 1
                return instance_id

        # Fallback to first instance
        self.current_index += 1
        return healthy_instances[0]

    def record_request(self, instance_id: str, response_time: float, success: bool = True) -> None:
        """Record request metrics for an instance.

        Args:
            instance_id: Instance that handled the request
            response_time: Request response time in seconds
            success: Whether request was successful
        """
        with self._lock:
            if instance_id not in self.instances:
                return

            instance = self.instances[instance_id]
            instance["total_requests"] += 1
            instance["last_request_time"] = time.time()

            if not success:
                instance["failed_requests"] += 1

            self.response_times[instance_id].append(response_time)
            self.request_counts[instance_id] += 1

    def _start_health_checks(self) -> None:
        """Start health check monitoring thread."""
        self._stop_health_checks = False
        self._health_check_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self._health_check_thread.start()
        logger.info("Started health check monitoring")

    def _health_check_loop(self) -> None:
        """Health check monitoring loop."""
        while not self._stop_health_checks:
            try:
                with self._lock:
                    instances_to_check = list(self.instances.items())

                for instance_id, instance_info in instances_to_check:
                    # Simple health check - in production this would make HTTP requests
                    # For now, simulate based on recent request success rate
                    total_requests = instance_info.get("total_requests", 0)
                    failed_requests = instance_info.get("failed_requests", 0)

                    if total_requests > 0:
                        failure_rate = failed_requests / total_requests
                        # Mark as unhealthy if failure rate > 50%
                        is_healthy = failure_rate <= 0.5
                    else:
                        # No requests yet, assume healthy
                        is_healthy = True

                    with self._lock:
                        if instance_id in self.instances:
                            old_health = self.instances[instance_id]["healthy"]
                            self.instances[instance_id]["healthy"] = is_healthy

                            if old_health != is_healthy:
                                status = "healthy" if is_healthy else "unhealthy"
                                logger.info(f"Instance {instance_id} marked as {status}")

                time.sleep(self.health_check_interval)

            except Exception as e:
                logger.error(f"Health check error: {e}")
                time.sleep(self.health_check_interval)

    def stop_health_checks(self) -> None:
        """Stop health check monitoring."""
        self._stop_health_checks = True
        if self._health_check_thread:
            self._health_check_thread.join(timeout=5.0)

    def get_load_balancer_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        with self._lock:
            total_instances = len(self.instances)
            healthy_instances = sum(1 for info in self.instances.values() if info["healthy"])

            total_requests = sum(info.get("total_requests", 0) for info in self.instances.values())
            total_failures = sum(info.get("failed_requests", 0) for info in self.instances.values())

            instance_stats = {}
            for instance_id, info in self.instances.items():
                response_times = list(self.response_times.get(instance_id, []))
                instance_stats[instance_id] = {
                    "healthy": info["healthy"],
                    "total_requests": info.get("total_requests", 0),
                    "failed_requests": info.get("failed_requests", 0),
                    "avg_response_time": statistics.mean(response_times) if response_times else 0,
                    "connections": info.get("connections", 0),
                    "weight": info.get("weight", 1.0),
                }

            return {
                "algorithm": self.algorithm,
                "total_instances": total_instances,
                "healthy_instances": healthy_instances,
                "total_requests": total_requests,
                "total_failures": total_failures,
                "success_rate": (total_requests - total_failures) / total_requests if total_requests > 0 else 1.0,
                "instance_stats": instance_stats,
            }
