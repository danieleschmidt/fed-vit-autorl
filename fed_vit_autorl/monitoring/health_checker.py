"""Health monitoring and system diagnostics."""

import time
import logging
try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False
    psutil = None

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False
    np = None

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    OFFLINE = "offline"


@dataclass
class HealthCheck:
    """Individual health check result."""
    name: str
    status: HealthStatus
    message: str
    timestamp: float
    metrics: Dict[str, Any]
    remediation: Optional[str] = None


class HealthChecker:
    """Base health monitoring system."""
    
    def __init__(
        self,
        check_interval: float = 60.0,
        critical_thresholds: Optional[Dict[str, float]] = None,
        warning_thresholds: Optional[Dict[str, float]] = None,
    ):
        """Initialize health checker.
        
        Args:
            check_interval: Time between health checks in seconds
            critical_thresholds: Critical threshold values
            warning_thresholds: Warning threshold values
        """
        self.check_interval = check_interval
        self.last_check_time = 0.0
        self.health_history: List[Dict[str, HealthCheck]] = []
        
        # Default thresholds
        self.critical_thresholds = critical_thresholds or {
            "cpu_usage": 90.0,
            "memory_usage": 95.0,
            "disk_usage": 95.0,
            "gpu_memory_usage": 95.0,
            "temperature": 85.0,
        }
        
        self.warning_thresholds = warning_thresholds or {
            "cpu_usage": 75.0,
            "memory_usage": 80.0,
            "disk_usage": 85.0,
            "gpu_memory_usage": 80.0,
            "temperature": 75.0,
        }
        
        logger.info("Initialized health checker")
    
    def check_system_resources(self) -> HealthCheck:
        """Check system resource utilization."""
        try:
            # Check if psutil is available
            if not _PSUTIL_AVAILABLE:
                return HealthCheck(
                    name="system_resources",
                    status=HealthStatus.CRITICAL,
                    message="System monitoring requires psutil package",
                    timestamp=time.time(),
                    metrics={},
                    remediation="Install psutil: pip install psutil"
                )
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # Determine status
            status = HealthStatus.HEALTHY
            messages = []
            
            if cpu_percent > self.critical_thresholds["cpu_usage"]:
                status = HealthStatus.CRITICAL
                messages.append(f"CPU usage critical: {cpu_percent:.1f}%")
            elif cpu_percent > self.warning_thresholds["cpu_usage"]:
                status = HealthStatus.WARNING
                messages.append(f"CPU usage high: {cpu_percent:.1f}%")
            
            if memory_percent > self.critical_thresholds["memory_usage"]:
                status = HealthStatus.CRITICAL
                messages.append(f"Memory usage critical: {memory_percent:.1f}%")
            elif memory_percent > self.warning_thresholds["memory_usage"]:
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.WARNING
                messages.append(f"Memory usage high: {memory_percent:.1f}%")
            
            if disk_percent > self.critical_thresholds["disk_usage"]:
                status = HealthStatus.CRITICAL
                messages.append(f"Disk usage critical: {disk_percent:.1f}%")
            elif disk_percent > self.warning_thresholds["disk_usage"]:
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.WARNING
                messages.append(f"Disk usage high: {disk_percent:.1f}%")
            
            message = "; ".join(messages) if messages else "System resources normal"
            
            return HealthCheck(
                name="system_resources",
                status=status,
                message=message,
                timestamp=time.time(),
                metrics={
                    "cpu_usage": cpu_percent,
                    "memory_usage": memory_percent,
                    "disk_usage": disk_percent,
                    "memory_available": memory.available / (1024**3),  # GB
                    "disk_free": disk.free / (1024**3),  # GB
                },
                remediation="Consider reducing workload or scaling resources" if status != HealthStatus.HEALTHY else None
            )
            
        except Exception as e:
            logger.error(f"System resource check failed: {e}")
            return HealthCheck(
                name="system_resources",
                status=HealthStatus.CRITICAL,
                message=f"Health check failed: {str(e)}",
                timestamp=time.time(),
                metrics={},
                remediation="Check system monitoring tools"
            )
    
    def check_gpu_resources(self) -> Optional[HealthCheck]:
        """Check GPU resource utilization."""
        if not torch.cuda.is_available():
            return None
        
        try:
            device_count = torch.cuda.device_count()
            gpu_metrics = {}
            status = HealthStatus.HEALTHY
            messages = []
            
            for i in range(device_count):
                # Memory usage
                memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
                memory_cached = torch.cuda.memory_reserved(i) / (1024**3)  # GB
                total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
                
                memory_usage = (memory_allocated / total_memory) * 100
                
                gpu_metrics[f"gpu_{i}_memory_allocated"] = memory_allocated
                gpu_metrics[f"gpu_{i}_memory_cached"] = memory_cached
                gpu_metrics[f"gpu_{i}_memory_usage"] = memory_usage
                gpu_metrics[f"gpu_{i}_total_memory"] = total_memory
                
                # Check thresholds
                if memory_usage > self.critical_thresholds["gpu_memory_usage"]:
                    status = HealthStatus.CRITICAL
                    messages.append(f"GPU {i} memory critical: {memory_usage:.1f}%")
                elif memory_usage > self.warning_thresholds["gpu_memory_usage"]:
                    if status == HealthStatus.HEALTHY:
                        status = HealthStatus.WARNING
                    messages.append(f"GPU {i} memory high: {memory_usage:.1f}%")
            
            message = "; ".join(messages) if messages else f"GPU resources normal ({device_count} devices)"
            
            return HealthCheck(
                name="gpu_resources",
                status=status,
                message=message,
                timestamp=time.time(),
                metrics=gpu_metrics,
                remediation="Clear GPU cache or reduce batch size" if status != HealthStatus.HEALTHY else None
            )
            
        except Exception as e:
            logger.error(f"GPU resource check failed: {e}")
            return HealthCheck(
                name="gpu_resources",
                status=HealthStatus.CRITICAL,
                message=f"GPU check failed: {str(e)}",
                timestamp=time.time(),
                metrics={},
                remediation="Check CUDA installation and drivers"
            )
    
    def check_model_health(self, model) -> HealthCheck:
        """Check model state and parameters."""
        try:
            # Check if torch is available
            if not _TORCH_AVAILABLE:
                return HealthCheck(
                    name="model_health",
                    status=HealthStatus.CRITICAL,
                    message="Model health checking requires PyTorch",
                    timestamp=time.time(),
                    metrics={},
                    remediation="Install PyTorch: pip install torch"
                )
            
            if model is None:
                return HealthCheck(
                    name="model_health",
                    status=HealthStatus.CRITICAL,
                    message="No model provided for health check",
                    timestamp=time.time(),
                    metrics={},
                    remediation="Provide a valid model instance"
                )
            status = HealthStatus.HEALTHY
            messages = []
            metrics = {}
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            metrics["total_parameters"] = total_params
            metrics["trainable_parameters"] = trainable_params
            
            # Check for NaN or infinite values
            nan_params = 0
            inf_params = 0
            param_norms = []
            
            for name, param in model.named_parameters():
                if torch.isnan(param).any():
                    nan_params += 1
                    status = HealthStatus.CRITICAL
                    messages.append(f"NaN values in parameter: {name}")
                
                if torch.isinf(param).any():
                    inf_params += 1
                    status = HealthStatus.CRITICAL
                    messages.append(f"Infinite values in parameter: {name}")
                
                param_norms.append(torch.norm(param).item())
            
            metrics["nan_parameters"] = nan_params
            metrics["inf_parameters"] = inf_params
            metrics["avg_param_norm"] = np.mean(param_norms) if param_norms else 0.0
            metrics["max_param_norm"] = np.max(param_norms) if param_norms else 0.0
            
            # Check gradient health if available
            grad_norms = []
            nan_grads = 0
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        nan_grads += 1
                        status = HealthStatus.CRITICAL
                        messages.append(f"NaN gradients in parameter: {name}")
                    
                    grad_norms.append(torch.norm(param.grad).item())
            
            metrics["nan_gradients"] = nan_grads
            metrics["avg_grad_norm"] = np.mean(grad_norms) if grad_norms else 0.0
            metrics["max_grad_norm"] = np.max(grad_norms) if grad_norms else 0.0
            
            # Check for exploding gradients
            if grad_norms and np.max(grad_norms) > 100:
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.WARNING
                messages.append(f"Large gradients detected: max norm = {np.max(grad_norms):.2f}")
            
            message = "; ".join(messages) if messages else "Model parameters healthy"
            
            return HealthCheck(
                name="model_health",
                status=status,
                message=message,
                timestamp=time.time(),
                metrics=metrics,
                remediation="Check learning rate, add gradient clipping, or restart training" if status != HealthStatus.HEALTHY else None
            )
            
        except Exception as e:
            logger.error(f"Model health check failed: {e}")
            return HealthCheck(
                name="model_health",
                status=HealthStatus.CRITICAL,
                message=f"Model check failed: {str(e)}",
                timestamp=time.time(),
                metrics={},
                remediation="Check model architecture and initialization"
            )
    
    def run_all_checks(self, model = None) -> Dict[str, HealthCheck]:
        """Run all health checks."""
        current_time = time.time()
        
        if current_time - self.last_check_time < self.check_interval:
            # Return cached results if check interval hasn't passed
            return self.health_history[-1] if self.health_history else {}
        
        checks = {}
        
        # System resource check
        checks["system"] = self.check_system_resources()
        
        # GPU resource check
        gpu_check = self.check_gpu_resources()
        if gpu_check:
            checks["gpu"] = gpu_check
        
        # Model health check
        if model is not None:
            checks["model"] = self.check_model_health(model)
        
        # Store results
        self.health_history.append(checks)
        self.last_check_time = current_time
        
        # Keep only last 100 check results
        if len(self.health_history) > 100:
            self.health_history = self.health_history[-100:]
        
        return checks
    
    def get_overall_status(self, checks: Optional[Dict[str, HealthCheck]] = None) -> HealthStatus:
        """Get overall system health status."""
        if checks is None:
            if not self.health_history:
                return HealthStatus.OFFLINE
            checks = self.health_history[-1]
        
        if not checks:
            return HealthStatus.OFFLINE
        
        # Determine worst status
        statuses = [check.status for check in checks.values()]
        
        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.WARNING in statuses:
            return HealthStatus.WARNING
        elif HealthStatus.HEALTHY in statuses:
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.OFFLINE
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get summary of health status."""
        if not self.health_history:
            return {"status": "no_data", "checks": 0}
        
        latest_checks = self.health_history[-1]
        overall_status = self.get_overall_status(latest_checks)
        
        # Calculate health trends
        recent_checks = self.health_history[-10:]  # Last 10 checks
        status_counts = {"healthy": 0, "warning": 0, "critical": 0, "offline": 0}
        
        for checks in recent_checks:
            status = self.get_overall_status(checks)
            status_counts[status.value] += 1
        
        return {
            "overall_status": overall_status.value,
            "last_check": latest_checks,
            "check_count": len(self.health_history),
            "recent_status_distribution": status_counts,
            "uptime_percentage": (status_counts["healthy"] / len(recent_checks)) * 100,
        }


class FederatedHealthChecker(HealthChecker):
    """Health checker specialized for federated learning systems."""
    
    def __init__(self, **kwargs):
        """Initialize federated health checker."""
        # Federated-specific thresholds
        federated_critical = kwargs.get("critical_thresholds", {})
        federated_critical.update({
            "client_participation_rate": 0.1,  # At least 10% participation
            "communication_latency": 10.0,     # Max 10 seconds
            "privacy_budget_remaining": 0.1,   # At least 10% budget left
            "aggregation_time": 300.0,         # Max 5 minutes per round
        })
        
        federated_warning = kwargs.get("warning_thresholds", {})
        federated_warning.update({
            "client_participation_rate": 0.3,  # Warn below 30%
            "communication_latency": 5.0,      # Warn above 5 seconds
            "privacy_budget_remaining": 0.3,   # Warn below 30%
            "aggregation_time": 120.0,         # Warn above 2 minutes
        })
        
        kwargs["critical_thresholds"] = federated_critical
        kwargs["warning_thresholds"] = federated_warning
        
        super().__init__(**kwargs)
        
        self.client_health_history: Dict[str, List[HealthCheck]] = {}
        self.federation_metrics: Dict[str, Any] = {}
        
        logger.info("Initialized federated health checker")
    
    def check_federation_health(
        self,
        participation_rate: float,
        avg_communication_latency: float,
        privacy_budget_remaining: float,
        last_aggregation_time: float,
    ) -> HealthCheck:
        """Check federated learning system health."""
        try:
            status = HealthStatus.HEALTHY
            messages = []
            metrics = {
                "participation_rate": participation_rate,
                "communication_latency": avg_communication_latency,
                "privacy_budget_remaining": privacy_budget_remaining,
                "aggregation_time": last_aggregation_time,
            }
            
            # Check participation rate
            if participation_rate < self.critical_thresholds["client_participation_rate"]:
                status = HealthStatus.CRITICAL
                messages.append(f"Critical low participation: {participation_rate:.1%}")
            elif participation_rate < self.warning_thresholds["client_participation_rate"]:
                status = HealthStatus.WARNING
                messages.append(f"Low participation: {participation_rate:.1%}")
            
            # Check communication latency
            if avg_communication_latency > self.critical_thresholds["communication_latency"]:
                status = HealthStatus.CRITICAL
                messages.append(f"Critical high latency: {avg_communication_latency:.1f}s")
            elif avg_communication_latency > self.warning_thresholds["communication_latency"]:
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.WARNING
                messages.append(f"High latency: {avg_communication_latency:.1f}s")
            
            # Check privacy budget
            if privacy_budget_remaining < self.critical_thresholds["privacy_budget_remaining"]:
                status = HealthStatus.CRITICAL
                messages.append(f"Critical low privacy budget: {privacy_budget_remaining:.1%}")
            elif privacy_budget_remaining < self.warning_thresholds["privacy_budget_remaining"]:
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.WARNING
                messages.append(f"Low privacy budget: {privacy_budget_remaining:.1%}")
            
            # Check aggregation time
            if last_aggregation_time > self.critical_thresholds["aggregation_time"]:
                status = HealthStatus.CRITICAL
                messages.append(f"Critical slow aggregation: {last_aggregation_time:.1f}s")
            elif last_aggregation_time > self.warning_thresholds["aggregation_time"]:
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.WARNING
                messages.append(f"Slow aggregation: {last_aggregation_time:.1f}s")
            
            message = "; ".join(messages) if messages else "Federation health normal"
            remediation = None
            
            if status != HealthStatus.HEALTHY:
                remediation_actions = []
                if participation_rate < 0.3:
                    remediation_actions.append("Check client connectivity and incentives")
                if avg_communication_latency > 5.0:
                    remediation_actions.append("Optimize network infrastructure")
                if privacy_budget_remaining < 0.3:
                    remediation_actions.append("Adjust privacy parameters or reset budget")
                if last_aggregation_time > 120.0:
                    remediation_actions.append("Optimize aggregation algorithm or increase compute")
                
                remediation = "; ".join(remediation_actions)
            
            return HealthCheck(
                name="federation_health",
                status=status,
                message=message,
                timestamp=time.time(),
                metrics=metrics,
                remediation=remediation
            )
            
        except Exception as e:
            logger.error(f"Federation health check failed: {e}")
            return HealthCheck(
                name="federation_health",
                status=HealthStatus.CRITICAL,
                message=f"Federation check failed: {str(e)}",
                timestamp=time.time(),
                metrics={},
                remediation="Check federated learning infrastructure"
            )
    
    def check_client_health(self, client_id: str, client_metrics: Dict[str, Any]) -> HealthCheck:
        """Check individual client health."""
        try:
            status = HealthStatus.HEALTHY
            messages = []
            
            # Check if client is responsive
            last_seen = client_metrics.get("last_seen", 0)
            time_since_seen = time.time() - last_seen
            
            if time_since_seen > 3600:  # 1 hour
                status = HealthStatus.CRITICAL
                messages.append(f"Client offline for {time_since_seen/3600:.1f} hours")
            elif time_since_seen > 600:  # 10 minutes
                status = HealthStatus.WARNING
                messages.append(f"Client not seen for {time_since_seen/60:.1f} minutes")
            
            # Check client performance metrics
            avg_loss = client_metrics.get("avg_loss", float("inf"))
            if avg_loss == float("inf"):
                status = HealthStatus.WARNING
                messages.append("No training loss reported")
            elif avg_loss > 10.0:  # High loss threshold
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.WARNING
                messages.append(f"High training loss: {avg_loss:.3f}")
            
            # Check safety metrics for vehicle clients
            safety_violations = client_metrics.get("safety_violations", 0)
            if safety_violations > 5:
                status = HealthStatus.CRITICAL
                messages.append(f"Multiple safety violations: {safety_violations}")
            elif safety_violations > 0:
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.WARNING
                messages.append(f"Safety violations detected: {safety_violations}")
            
            message = "; ".join(messages) if messages else f"Client {client_id} healthy"
            
            health_check = HealthCheck(
                name=f"client_{client_id}",
                status=status,
                message=message,
                timestamp=time.time(),
                metrics=client_metrics,
                remediation="Check client connectivity and performance" if status != HealthStatus.HEALTHY else None
            )
            
            # Store client health history
            if client_id not in self.client_health_history:
                self.client_health_history[client_id] = []
            
            self.client_health_history[client_id].append(health_check)
            
            # Keep only last 50 checks per client
            if len(self.client_health_history[client_id]) > 50:
                self.client_health_history[client_id] = self.client_health_history[client_id][-50:]
            
            return health_check
            
        except Exception as e:
            logger.error(f"Client health check failed for {client_id}: {e}")
            return HealthCheck(
                name=f"client_{client_id}",
                status=HealthStatus.CRITICAL,
                message=f"Client check failed: {str(e)}",
                timestamp=time.time(),
                metrics={},
                remediation="Check client status and connection"
            )
    
    def get_federation_summary(self) -> Dict[str, Any]:
        """Get comprehensive federation health summary."""
        summary = self.get_health_summary()
        
        # Add client-specific summary
        client_statuses = {}
        for client_id, health_history in self.client_health_history.items():
            if health_history:
                latest_health = health_history[-1]
                client_statuses[client_id] = latest_health.status.value
        
        summary.update({
            "total_clients": len(self.client_health_history),
            "client_statuses": client_statuses,
            "healthy_clients": sum(1 for status in client_statuses.values() if status == "healthy"),
            "warning_clients": sum(1 for status in client_statuses.values() if status == "warning"),
            "critical_clients": sum(1 for status in client_statuses.values() if status == "critical"),
            "offline_clients": sum(1 for status in client_statuses.values() if status == "offline"),
        })
        
        return summary