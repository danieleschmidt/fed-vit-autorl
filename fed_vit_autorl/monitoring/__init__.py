"""Monitoring and observability components."""

from .health_checker import HealthChecker, FederatedHealthChecker

# Optional imports for additional monitoring components
try:
    from .metrics_collector import MetricsCollector, FederatedMetricsCollector
except ImportError:
    MetricsCollector = None
    FederatedMetricsCollector = None

try:
    from .performance_monitor import PerformanceMonitor
except ImportError:
    PerformanceMonitor = None

try:
    from .alerting import AlertManager
except ImportError:
    AlertManager = None

try:
    from .dashboard_generator import DashboardGenerator
except ImportError:
    DashboardGenerator = None

__all__ = [
    "HealthChecker",
    "FederatedHealthChecker", 
    "MetricsCollector",
    "FederatedMetricsCollector",
    "PerformanceMonitor",
    "AlertManager",
    "DashboardGenerator",
]