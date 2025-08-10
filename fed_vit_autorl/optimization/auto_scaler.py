"""Advanced auto-scaling system for Fed-ViT-AutoRL."""

import time
import logging
import threading
import math
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import deque
import asyncio
import json

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


class ScalingDirection(Enum):
    """Scaling directions."""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"


class ScalingMetric(Enum):
    """Metrics used for scaling decisions."""
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    GPU_USAGE = "gpu_usage"
    QUEUE_LENGTH = "queue_length"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    CUSTOM = "custom"


@dataclass
class ScalingEvent:
    """Record of a scaling event."""
    timestamp: float
    direction: ScalingDirection
    trigger_metric: ScalingMetric
    metric_value: float
    threshold: float
    old_capacity: int
    new_capacity: int
    reason: str


@dataclass
class MetricThresholds:
    """Thresholds for scaling metrics."""
    scale_up_threshold: float
    scale_down_threshold: float
    metric_type: ScalingMetric
    evaluation_period: float = 60.0  # seconds
    min_data_points: int = 3


class PredictiveScaler:
    """Predictive auto-scaler using trend analysis and machine learning."""
    
    def __init__(self,
                 min_instances: int = 1,
                 max_instances: int = 10,
                 scale_up_cooldown: float = 300.0,  # 5 minutes
                 scale_down_cooldown: float = 600.0,  # 10 minutes
                 prediction_window: float = 300.0):  # 5 minutes
        """Initialize predictive scaler.
        
        Args:
            min_instances: Minimum number of instances
            max_instances: Maximum number of instances
            scale_up_cooldown: Cooldown period after scaling up
            scale_down_cooldown: Cooldown period after scaling down
            prediction_window: Time window for predictions
        """
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.current_instances = min_instances
        self.scale_up_cooldown = scale_up_cooldown
        self.scale_down_cooldown = scale_down_cooldown
        self.prediction_window = prediction_window
        
        # Metric tracking
        self.metrics_history: Dict[ScalingMetric, deque] = {
            metric: deque(maxlen=1000) for metric in ScalingMetric
        }
        self.thresholds: Dict[ScalingMetric, MetricThresholds] = {}
        
        # Event tracking
        self.scaling_events: List[ScalingEvent] = []
        self.last_scale_up = 0.0
        self.last_scale_down = 0.0
        
        # Threading
        self._lock = threading.RLock()
        self._monitoring = False
        self._monitor_thread = None
        
        # Callbacks
        self.scale_up_callback: Optional[Callable[[int], bool]] = None
        self.scale_down_callback: Optional[Callable[[int], bool]] = None
        
        logger.info(f"Initialized predictive scaler (min={min_instances}, max={max_instances})")
    
    def add_metric_threshold(self, 
                           metric: ScalingMetric,
                           scale_up_threshold: float,
                           scale_down_threshold: float,
                           evaluation_period: float = 60.0) -> None:
        """Add metric threshold for scaling decisions.
        
        Args:
            metric: Metric type
            scale_up_threshold: Threshold for scaling up
            scale_down_threshold: Threshold for scaling down
            evaluation_period: Period to evaluate metric over
        """
        self.thresholds[metric] = MetricThresholds(
            scale_up_threshold=scale_up_threshold,
            scale_down_threshold=scale_down_threshold,
            metric_type=metric,
            evaluation_period=evaluation_period
        )
        logger.info(f"Added threshold for {metric.value}: up={scale_up_threshold}, down={scale_down_threshold}")
    
    def record_metric(self, metric: ScalingMetric, value: float, timestamp: Optional[float] = None) -> None:
        """Record metric value.
        
        Args:
            metric: Metric type
            value: Metric value
            timestamp: Optional timestamp (uses current time if None)
        """
        if timestamp is None:
            timestamp = time.time()
        
        with self._lock:
            self.metrics_history[metric].append((timestamp, value))
    
    def get_metric_trend(self, metric: ScalingMetric, window_seconds: float = 300.0) -> Optional[float]:
        """Calculate trend for metric over time window.
        
        Args:
            metric: Metric type
            window_seconds: Time window in seconds
            
        Returns:
            Trend slope (positive = increasing, negative = decreasing)
        """
        with self._lock:
            history = self.metrics_history[metric]
            if len(history) < 2:
                return None
            
            current_time = time.time()
            recent_data = [
                (ts, val) for ts, val in history
                if current_time - ts <= window_seconds
            ]
            
            if len(recent_data) < 2:
                return None
            
            # Simple linear regression for trend
            n = len(recent_data)
            sum_x = sum(ts for ts, _ in recent_data)
            sum_y = sum(val for _, val in recent_data)
            sum_xy = sum(ts * val for ts, val in recent_data)
            sum_x2 = sum(ts * ts for ts, _ in recent_data)
            
            denominator = n * sum_x2 - sum_x * sum_x
            if denominator == 0:
                return None
            
            slope = (n * sum_xy - sum_x * sum_y) / denominator
            return slope
    
    def predict_future_load(self, metric: ScalingMetric, 
                          prediction_horizon: float = 300.0) -> Optional[float]:
        """Predict future metric value.
        
        Args:
            metric: Metric type
            prediction_horizon: How far ahead to predict (seconds)
            
        Returns:
            Predicted metric value
        """
        trend = self.get_metric_trend(metric)
        if trend is None:
            return None
        
        with self._lock:
            history = self.metrics_history[metric]
            if not history:
                return None
            
            # Get current value
            current_value = history[-1][1]
            
            # Predict future value using trend
            predicted_value = current_value + (trend * prediction_horizon)
            
            return predicted_value
    
    def should_scale(self) -> Tuple[ScalingDirection, Optional[ScalingMetric], str]:
        """Determine if scaling is needed.
        
        Returns:
            Tuple of (direction, triggering_metric, reason)
        """
        current_time = time.time()
        
        # Check cooldown periods
        if (current_time - self.last_scale_up < self.scale_up_cooldown and
            current_time - self.last_scale_down < self.scale_down_cooldown):
            return ScalingDirection.STABLE, None, "In cooldown period"
        
        # Check each metric
        for metric, threshold in self.thresholds.items():
            avg_value = self._get_average_metric(metric, threshold.evaluation_period)
            if avg_value is None:
                continue
            
            # Check for scale up
            if (avg_value > threshold.scale_up_threshold and
                self.current_instances < self.max_instances and
                current_time - self.last_scale_up >= self.scale_up_cooldown):
                
                # Use predictive scaling
                predicted_value = self.predict_future_load(metric, self.prediction_window)
                if predicted_value and predicted_value > threshold.scale_up_threshold * 1.2:
                    return ScalingDirection.UP, metric, f"Predicted load increase: {predicted_value:.2f}"
                
                return ScalingDirection.UP, metric, f"Current load: {avg_value:.2f} > {threshold.scale_up_threshold}"
            
            # Check for scale down
            if (avg_value < threshold.scale_down_threshold and
                self.current_instances > self.min_instances and
                current_time - self.last_scale_down >= self.scale_down_cooldown):
                
                # Use predictive scaling
                predicted_value = self.predict_future_load(metric, self.prediction_window)
                if predicted_value and predicted_value < threshold.scale_down_threshold * 0.8:
                    return ScalingDirection.DOWN, metric, f"Predicted load decrease: {predicted_value:.2f}"
                
                return ScalingDirection.DOWN, metric, f"Current load: {avg_value:.2f} < {threshold.scale_down_threshold}"
        
        return ScalingDirection.STABLE, None, "All metrics within thresholds"
    
    def _get_average_metric(self, metric: ScalingMetric, period_seconds: float) -> Optional[float]:
        """Get average metric value over time period.
        
        Args:
            metric: Metric type
            period_seconds: Time period in seconds
            
        Returns:
            Average metric value or None if insufficient data
        """
        with self._lock:
            history = self.metrics_history[metric]
            if not history:
                return None
            
            current_time = time.time()
            recent_values = [
                val for ts, val in history
                if current_time - ts <= period_seconds
            ]
            
            if not recent_values:
                return None
            
            return sum(recent_values) / len(recent_values)
    
    def scale(self, direction: ScalingDirection, 
              trigger_metric: Optional[ScalingMetric] = None,
              reason: str = "") -> bool:
        """Execute scaling action.
        
        Args:
            direction: Scaling direction
            trigger_metric: Metric that triggered scaling
            reason: Reason for scaling
            
        Returns:
            True if scaling was successful
        """
        if direction == ScalingDirection.STABLE:
            return True
        
        old_instances = self.current_instances
        
        if direction == ScalingDirection.UP:
            new_instances = min(self.current_instances + 1, self.max_instances)
            if new_instances == self.current_instances:
                logger.info("Already at maximum instances")
                return False
            
            if self.scale_up_callback:
                success = self.scale_up_callback(new_instances)
            else:
                success = True  # Assume success if no callback
            
            if success:
                self.current_instances = new_instances
                self.last_scale_up = time.time()
                
        else:  # Scale down
            new_instances = max(self.current_instances - 1, self.min_instances)
            if new_instances == self.current_instances:
                logger.info("Already at minimum instances")
                return False
            
            if self.scale_down_callback:
                success = self.scale_down_callback(new_instances)
            else:
                success = True  # Assume success if no callback
            
            if success:
                self.current_instances = new_instances
                self.last_scale_down = time.time()
        
        if success:
            # Record scaling event
            event = ScalingEvent(
                timestamp=time.time(),
                direction=direction,
                trigger_metric=trigger_metric or ScalingMetric.CUSTOM,
                metric_value=0.0,  # Could be populated with actual value
                threshold=0.0,     # Could be populated with actual threshold
                old_capacity=old_instances,
                new_capacity=self.current_instances,
                reason=reason
            )
            self.scaling_events.append(event)
            
            logger.info(f"Scaled {direction.value}: {old_instances} -> {self.current_instances} instances")
        
        return success
    
    def start_monitoring(self, check_interval: float = 30.0) -> None:
        """Start automatic monitoring and scaling.
        
        Args:
            check_interval: Interval between scaling checks in seconds
        """
        if self._monitoring:
            logger.warning("Monitoring already started")
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(check_interval,),
            daemon=True
        )
        self._monitor_thread.start()
        logger.info(f"Started auto-scaling monitoring (interval={check_interval}s)")
    
    def stop_monitoring(self) -> None:
        """Stop automatic monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("Stopped auto-scaling monitoring")
    
    def _monitoring_loop(self, check_interval: float) -> None:
        """Main monitoring loop."""
        while self._monitoring:
            try:
                direction, metric, reason = self.should_scale()
                if direction != ScalingDirection.STABLE:
                    success = self.scale(direction, metric, reason)
                    if success:
                        logger.info(f"Auto-scaled {direction.value}: {reason}")
                    else:
                        logger.warning(f"Auto-scaling {direction.value} failed: {reason}")
                
                time.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(check_interval)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics for scaling decisions."""
        metrics = {}
        
        if _PSUTIL_AVAILABLE:
            metrics['cpu_usage'] = psutil.cpu_percent(interval=1)
            metrics['memory_usage'] = psutil.virtual_memory().percent
            
            # Record metrics
            self.record_metric(ScalingMetric.CPU_USAGE, metrics['cpu_usage'])
            self.record_metric(ScalingMetric.MEMORY_USAGE, metrics['memory_usage'])
        
        return metrics
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get scaling statistics and history.
        
        Returns:
            Dictionary with scaling statistics
        """
        with self._lock:
            scale_up_events = [e for e in self.scaling_events if e.direction == ScalingDirection.UP]
            scale_down_events = [e for e in self.scaling_events if e.direction == ScalingDirection.DOWN]
            
            return {
                'current_instances': self.current_instances,
                'min_instances': self.min_instances,
                'max_instances': self.max_instances,
                'total_scale_events': len(self.scaling_events),
                'scale_up_events': len(scale_up_events),
                'scale_down_events': len(scale_down_events),
                'last_scale_up': self.last_scale_up,
                'last_scale_down': self.last_scale_down,
                'monitoring_active': self._monitoring,
                'recent_events': [
                    {
                        'timestamp': e.timestamp,
                        'direction': e.direction.value,
                        'trigger_metric': e.trigger_metric.value,
                        'old_capacity': e.old_capacity,
                        'new_capacity': e.new_capacity,
                        'reason': e.reason
                    }
                    for e in self.scaling_events[-10:]  # Last 10 events
                ]
            }


class ResourceOptimizer:
    """Intelligent resource optimization for federated learning."""
    
    def __init__(self):
        """Initialize resource optimizer."""
        self.scaler = PredictiveScaler()
        self._setup_default_thresholds()
        
        logger.info("Initialized resource optimizer")
    
    def _setup_default_thresholds(self) -> None:
        """Setup default scaling thresholds."""
        # CPU-based scaling
        self.scaler.add_metric_threshold(
            ScalingMetric.CPU_USAGE,
            scale_up_threshold=75.0,    # Scale up at 75% CPU
            scale_down_threshold=25.0,  # Scale down at 25% CPU
            evaluation_period=120.0     # 2-minute evaluation period
        )
        
        # Memory-based scaling
        self.scaler.add_metric_threshold(
            ScalingMetric.MEMORY_USAGE,
            scale_up_threshold=80.0,    # Scale up at 80% memory
            scale_down_threshold=30.0,  # Scale down at 30% memory
            evaluation_period=60.0      # 1-minute evaluation period
        )
        
        # Latency-based scaling
        self.scaler.add_metric_threshold(
            ScalingMetric.LATENCY,
            scale_up_threshold=100.0,   # Scale up at 100ms latency
            scale_down_threshold=20.0,  # Scale down at 20ms latency
            evaluation_period=30.0      # 30-second evaluation period
        )
    
    def optimize_resources(self) -> Dict[str, Any]:
        """Optimize resource allocation based on current metrics.
        
        Returns:
            Optimization results and recommendations
        """
        # Get current system metrics
        current_metrics = self.scaler.get_current_metrics()
        
        # Check scaling needs
        direction, metric, reason = self.scaler.should_scale()
        
        # Generate recommendations
        recommendations = []
        
        if direction == ScalingDirection.UP:
            recommendations.append({
                'type': 'scale_up',
                'reason': reason,
                'priority': 'high',
                'estimated_benefit': 'Improved performance and reduced latency'
            })
        elif direction == ScalingDirection.DOWN:
            recommendations.append({
                'type': 'scale_down', 
                'reason': reason,
                'priority': 'medium',
                'estimated_benefit': 'Reduced resource costs'
            })
        
        # Memory optimization recommendations
        if current_metrics.get('memory_usage', 0) > 70:
            recommendations.append({
                'type': 'memory_optimization',
                'reason': 'High memory usage detected',
                'priority': 'medium',
                'actions': ['Clear unused caches', 'Reduce batch sizes', 'Enable memory pooling']
            })
        
        # CPU optimization recommendations
        if current_metrics.get('cpu_usage', 0) > 80:
            recommendations.append({
                'type': 'cpu_optimization',
                'reason': 'High CPU usage detected',
                'priority': 'high',
                'actions': ['Increase thread pool size', 'Optimize model inference', 'Enable parallel processing']
            })
        
        return {
            'current_metrics': current_metrics,
            'scaling_decision': {
                'direction': direction.value,
                'trigger_metric': metric.value if metric else None,
                'reason': reason
            },
            'recommendations': recommendations,
            'scaling_stats': self.scaler.get_scaling_stats()
        }
    
    def start_optimization(self) -> None:
        """Start continuous resource optimization."""
        self.scaler.start_monitoring(check_interval=60.0)
        logger.info("Started continuous resource optimization")
    
    def stop_optimization(self) -> None:
        """Stop continuous resource optimization."""
        self.scaler.stop_monitoring()
        logger.info("Stopped continuous resource optimization")