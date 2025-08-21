"""Memory optimization and management for federated learning."""

import gc
import time
import logging
import threading
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass
from collections import defaultdict, deque
import weakref

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False
    np = None

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False
    psutil = None

logger = logging.getLogger(__name__)


@dataclass
class MemoryUsage:
    """Memory usage statistics."""
    total_memory: float  # GB
    available_memory: float  # GB
    used_memory: float  # GB
    memory_percent: float
    swap_used: float  # GB
    gpu_memory_allocated: float  # GB
    gpu_memory_cached: float  # GB
    timestamp: float


class MemoryOptimizer:
    """Advanced memory optimization and management system."""

    def __init__(
        self,
        memory_limit: Optional[float] = None,  # GB
        warning_threshold: float = 80.0,  # %
        critical_threshold: float = 95.0,  # %
        gc_interval: float = 60.0,  # seconds
        enable_auto_cleanup: bool = True,
        enable_memory_profiling: bool = False,
    ):
        """Initialize memory optimizer.

        Args:
            memory_limit: Maximum memory usage in GB (None for system limit)
            warning_threshold: Warning threshold as percentage of limit
            critical_threshold: Critical threshold as percentage of limit
            gc_interval: Garbage collection interval in seconds
            enable_auto_cleanup: Whether to enable automatic cleanup
            enable_memory_profiling: Whether to enable memory profiling
        """
        self.memory_limit = memory_limit
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.gc_interval = gc_interval
        self.enable_auto_cleanup = enable_auto_cleanup
        self.enable_memory_profiling = enable_memory_profiling

        # Memory tracking
        self.memory_history: deque = deque(maxlen=1000)
        self.memory_callbacks: List[Callable[[MemoryUsage], None]] = []
        self.cached_objects: Dict[str, Any] = {}
        self.weak_references: Dict[str, weakref.ref] = {}
        self.memory_pools: Dict[str, List[Any]] = defaultdict(list)

        # Monitoring
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitoring = False
        self._lock = threading.Lock()

        # Statistics
        self.cleanup_stats = {
            "total_cleanups": 0,
            "memory_freed": 0.0,
            "last_cleanup": 0.0,
            "auto_cleanups": 0,
            "manual_cleanups": 0,
        }

        if _PSUTIL_AVAILABLE:
            system_memory = psutil.virtual_memory().total / (1024**3)  # GB
            if not self.memory_limit:
                self.memory_limit = system_memory * 0.8  # Use 80% of system memory
        else:
            if not self.memory_limit:
                self.memory_limit = 8.0  # Default 8GB limit

        logger.info(f"Initialized memory optimizer with {self.memory_limit:.1f}GB limit")

    def start_monitoring(self) -> None:
        """Start memory monitoring."""
        if self._monitoring:
            return

        self._monitoring = True
        self._stop_monitoring = False
        self._monitor_thread = threading.Thread(target=self._monitor_memory, daemon=True)
        self._monitor_thread.start()

        logger.info("Started memory monitoring")

    def stop_monitoring(self) -> None:
        """Stop memory monitoring."""
        if not self._monitoring:
            return

        self._stop_monitoring = True
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)

        self._monitoring = False
        logger.info("Stopped memory monitoring")

    def _monitor_memory(self) -> None:
        """Memory monitoring loop."""
        while not self._stop_monitoring:
            try:
                usage = self.get_memory_usage()

                with self._lock:
                    self.memory_history.append(usage)

                # Check thresholds
                if usage.memory_percent >= self.critical_threshold:
                    logger.critical(f"Critical memory usage: {usage.memory_percent:.1f}%")
                    if self.enable_auto_cleanup:
                        self.emergency_cleanup()
                elif usage.memory_percent >= self.warning_threshold:
                    logger.warning(f"High memory usage: {usage.memory_percent:.1f}%")
                    if self.enable_auto_cleanup:
                        self.cleanup_memory(aggressive=False)

                # Notify callbacks
                for callback in self.memory_callbacks:
                    try:
                        callback(usage)
                    except Exception as e:
                        logger.error(f"Memory callback failed: {e}")

                time.sleep(self.gc_interval)

            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                time.sleep(self.gc_interval)

    def get_memory_usage(self) -> MemoryUsage:
        """Get current memory usage statistics."""
        if _PSUTIL_AVAILABLE:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()

            total_memory = memory.total / (1024**3)
            available_memory = memory.available / (1024**3)
            used_memory = memory.used / (1024**3)
            memory_percent = memory.percent
            swap_used = swap.used / (1024**3)
        else:
            # Fallback values when psutil is not available
            total_memory = self.memory_limit
            available_memory = self.memory_limit * 0.5  # Estimate
            used_memory = total_memory - available_memory
            memory_percent = (used_memory / total_memory) * 100
            swap_used = 0.0

        # GPU memory if available
        gpu_memory_allocated = 0.0
        gpu_memory_cached = 0.0

        if _TORCH_AVAILABLE and torch and torch.cuda.is_available():
            try:
                gpu_memory_allocated = torch.cuda.memory_allocated() / (1024**3)
                gpu_memory_cached = torch.cuda.memory_reserved() / (1024**3)
            except Exception as e:
                logger.debug(f"GPU memory check failed: {e}")

        return MemoryUsage(
            total_memory=total_memory,
            available_memory=available_memory,
            used_memory=used_memory,
            memory_percent=memory_percent,
            swap_used=swap_used,
            gpu_memory_allocated=gpu_memory_allocated,
            gpu_memory_cached=gpu_memory_cached,
            timestamp=time.time(),
        )

    def cleanup_memory(self, aggressive: bool = False) -> float:
        """Perform memory cleanup.

        Args:
            aggressive: Whether to perform aggressive cleanup

        Returns:
            Amount of memory freed in GB
        """
        start_usage = self.get_memory_usage()

        # Clear caches
        self.clear_caches()

        # Clear memory pools if aggressive
        if aggressive:
            self.clear_memory_pools()

        # Python garbage collection
        collected = gc.collect()
        logger.debug(f"Garbage collection freed {collected} objects")

        # PyTorch GPU memory cleanup
        if _TORCH_AVAILABLE and torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Force another garbage collection
        if aggressive:
            for generation in range(3):
                gc.collect(generation)

        end_usage = self.get_memory_usage()
        memory_freed = start_usage.used_memory - end_usage.used_memory

        # Update statistics
        self.cleanup_stats["total_cleanups"] += 1
        self.cleanup_stats["memory_freed"] += max(0, memory_freed)
        self.cleanup_stats["last_cleanup"] = time.time()

        if aggressive:
            self.cleanup_stats["manual_cleanups"] += 1
        else:
            self.cleanup_stats["auto_cleanups"] += 1

        logger.info(f"Memory cleanup freed {memory_freed:.3f}GB (aggressive={aggressive})")
        return max(0, memory_freed)

    def emergency_cleanup(self) -> float:
        """Perform emergency memory cleanup.

        Returns:
            Amount of memory freed in GB
        """
        logger.warning("Performing emergency memory cleanup")

        # Clear all caches and pools
        self.clear_caches()
        self.clear_memory_pools()

        # Aggressive garbage collection
        total_collected = 0
        for _ in range(5):  # Multiple passes
            collected = gc.collect()
            total_collected += collected
            if collected == 0:
                break

        # Clear PyTorch cache multiple times
        if _TORCH_AVAILABLE and torch and torch.cuda.is_available():
            for _ in range(3):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        logger.info(f"Emergency cleanup collected {total_collected} objects")
        return self.cleanup_memory(aggressive=True)

    def clear_caches(self) -> None:
        """Clear internal caches."""
        with self._lock:
            cleared_count = len(self.cached_objects)
            self.cached_objects.clear()

            # Clean up weak references
            dead_refs = []
            for key, ref in self.weak_references.items():
                if ref() is None:
                    dead_refs.append(key)

            for key in dead_refs:
                del self.weak_references[key]

            logger.debug(f"Cleared {cleared_count} cached objects and {len(dead_refs)} dead references")

    def clear_memory_pools(self) -> None:
        """Clear memory pools."""
        with self._lock:
            total_objects = sum(len(pool) for pool in self.memory_pools.values())
            self.memory_pools.clear()
            logger.debug(f"Cleared {total_objects} objects from memory pools")

    def cache_object(self, key: str, obj: Any, weak: bool = False) -> None:
        """Cache an object for reuse.

        Args:
            key: Cache key
            obj: Object to cache
            weak: Whether to use weak reference
        """
        with self._lock:
            if weak:
                self.weak_references[key] = weakref.ref(obj)
            else:
                self.cached_objects[key] = obj

    def get_cached_object(self, key: str) -> Optional[Any]:
        """Get cached object.

        Args:
            key: Cache key

        Returns:
            Cached object or None
        """
        with self._lock:
            # Check regular cache
            if key in self.cached_objects:
                return self.cached_objects[key]

            # Check weak references
            if key in self.weak_references:
                ref = self.weak_references[key]
                obj = ref()
                if obj is None:
                    # Object was garbage collected
                    del self.weak_references[key]
                return obj

            return None

    def add_to_pool(self, pool_name: str, obj: Any) -> None:
        """Add object to memory pool for reuse.

        Args:
            pool_name: Pool name
            obj: Object to add
        """
        with self._lock:
            self.memory_pools[pool_name].append(obj)

    def get_from_pool(self, pool_name: str) -> Optional[Any]:
        """Get object from memory pool.

        Args:
            pool_name: Pool name

        Returns:
            Pooled object or None
        """
        with self._lock:
            pool = self.memory_pools.get(pool_name, [])
            if pool:
                return pool.pop()
            return None

    def add_memory_callback(self, callback: Callable[[MemoryUsage], None]) -> None:
        """Add memory usage callback.

        Args:
            callback: Function to call with memory usage data
        """
        self.memory_callbacks.append(callback)

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics.

        Returns:
            Memory statistics dictionary
        """
        current_usage = self.get_memory_usage()

        with self._lock:
            recent_usage = list(self.memory_history)[-10:] if self.memory_history else [current_usage]

            avg_memory_percent = sum(u.memory_percent for u in recent_usage) / len(recent_usage)
            max_memory_percent = max(u.memory_percent for u in recent_usage)
            min_memory_percent = min(u.memory_percent for u in recent_usage)

            cache_stats = {
                "cached_objects": len(self.cached_objects),
                "weak_references": len(self.weak_references),
                "memory_pools": {name: len(pool) for name, pool in self.memory_pools.items()},
                "total_pooled_objects": sum(len(pool) for pool in self.memory_pools.values()),
            }

        return {
            "current_usage": {
                "memory_percent": current_usage.memory_percent,
                "used_memory_gb": current_usage.used_memory,
                "available_memory_gb": current_usage.available_memory,
                "gpu_memory_gb": current_usage.gpu_memory_allocated,
                "gpu_cached_gb": current_usage.gpu_memory_cached,
            },
            "recent_stats": {
                "avg_memory_percent": avg_memory_percent,
                "max_memory_percent": max_memory_percent,
                "min_memory_percent": min_memory_percent,
            },
            "thresholds": {
                "warning_threshold": self.warning_threshold,
                "critical_threshold": self.critical_threshold,
                "memory_limit_gb": self.memory_limit,
            },
            "cleanup_stats": self.cleanup_stats.copy(),
            "cache_stats": cache_stats,
            "monitoring_active": self._monitoring,
        }

    def optimize_for_inference(self) -> None:
        """Optimize memory for inference workload."""
        logger.info("Optimizing memory for inference")

        # Clear training-specific caches
        self.cleanup_memory(aggressive=True)

        # Set PyTorch to inference mode optimizations
        if _TORCH_AVAILABLE and torch:
            torch.set_grad_enabled(False)
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

    def optimize_for_training(self) -> None:
        """Optimize memory for training workload."""
        logger.info("Optimizing memory for training")

        # Enable gradient computation
        if _TORCH_AVAILABLE and torch:
            torch.set_grad_enabled(True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        # Clear inference-specific caches
        self.cleanup_memory(aggressive=False)

    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            self.stop_monitoring()
        except Exception:
            pass  # Ignore cleanup errors during destruction"
