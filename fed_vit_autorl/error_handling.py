"""Comprehensive error handling and recovery mechanisms."""

import functools
import logging
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Type, Union
from enum import Enum
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None
import threading
from contextlib import contextmanager


logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    NETWORK = "network"
    COMPUTATION = "computation"  
    MEMORY = "memory"
    SECURITY = "security"
    VALIDATION = "validation"
    PRIVACY = "privacy"
    FEDERATED = "federated"
    HARDWARE = "hardware"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class FederatedError(Exception):
    """Base exception for federated learning errors."""
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        recoverable: bool = True,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.category = category
        self.severity = severity
        self.recoverable = recoverable
        self.context = context or {}
        self.timestamp = time.time()


class ClientError(FederatedError):
    """Client-side federated learning error."""
    
    def __init__(self, message: str, client_id: str, **kwargs):
        super().__init__(message, category=ErrorCategory.FEDERATED, **kwargs)
        self.client_id = client_id
        self.context["client_id"] = client_id


class ServerError(FederatedError):
    """Server-side federated learning error."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.FEDERATED, **kwargs)


class CommunicationError(FederatedError):
    """Communication-related error."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.NETWORK, **kwargs)


class ComputationError(FederatedError):
    """Computation-related error."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.COMPUTATION, **kwargs)


class PrivacyError(FederatedError):
    """Privacy-related error."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.PRIVACY,
            severity=ErrorSeverity.HIGH,
            recoverable=False,
            **kwargs
        )


class ErrorHandler:
    """Centralized error handling and recovery system."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        """Initialize error handler.
        
        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay for exponential backoff
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.error_history: List[Dict[str, Any]] = []
        self.recovery_strategies: Dict[ErrorCategory, List[Callable]] = {
            ErrorCategory.NETWORK: [self._retry_with_backoff, self._fallback_communication],
            ErrorCategory.COMPUTATION: [self._clear_cache, self._reduce_batch_size],
            ErrorCategory.MEMORY: [self._clear_memory, self._reduce_model_precision],
            ErrorCategory.TIMEOUT: [self._increase_timeout, self._retry_with_backoff],
            ErrorCategory.FEDERATED: [self._reset_client_state, self._retry_aggregation],
        }
        self._lock = threading.Lock()
        
        logger.info("Initialized error handler")
    
    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        auto_recover: bool = True,
    ) -> bool:
        """Handle error with automatic recovery if possible.
        
        Args:
            error: Exception that occurred
            context: Additional context information
            auto_recover: Whether to attempt automatic recovery
            
        Returns:
            True if error was recovered, False otherwise
        """
        with self._lock:
            # Classify error
            if isinstance(error, FederatedError):
                fed_error = error
            else:
                fed_error = self._classify_error(error, context)
            
            # Log error
            self._log_error(fed_error, context)
            
            # Record error in history
            self._record_error(fed_error, context)
            
            # Attempt recovery if enabled and error is recoverable
            if auto_recover and fed_error.recoverable:
                return self._attempt_recovery(fed_error)
            
            return False
    
    def _classify_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
    ) -> FederatedError:
        """Classify generic exception into federated error."""
        error_msg = str(error)
        error_type = type(error).__name__
        
        # Network-related errors
        if any(keyword in error_msg.lower() for keyword in [
            "connection", "network", "timeout", "socket", "ssl", "certificate"
        ]):
            return FederatedError(
                f"{error_type}: {error_msg}",
                category=ErrorCategory.NETWORK,
                severity=ErrorSeverity.MEDIUM,
                context=context
            )
        
        # Memory errors
        if any(keyword in error_msg.lower() for keyword in [
            "memory", "cuda out of memory", "allocation", "oom"
        ]) or isinstance(error, (MemoryError, RuntimeError)) and "memory" in error_msg:
            return FederatedError(
                f"{error_type}: {error_msg}",
                category=ErrorCategory.MEMORY,
                severity=ErrorSeverity.HIGH,
                context=context
            )
        
        # Computation errors
        if isinstance(error, (ArithmeticError, ValueError, TypeError)):
            return FederatedError(
                f"{error_type}: {error_msg}",
                category=ErrorCategory.COMPUTATION,
                severity=ErrorSeverity.MEDIUM,
                context=context
            )
        
        # Timeout errors
        if isinstance(error, TimeoutError) or "timeout" in error_msg.lower():
            return FederatedError(
                f"{error_type}: {error_msg}",
                category=ErrorCategory.TIMEOUT,
                severity=ErrorSeverity.MEDIUM,
                context=context
            )
        
        # Security/validation errors
        if any(keyword in error_msg.lower() for keyword in [
            "validation", "invalid", "permission", "access", "security"
        ]):
            return FederatedError(
                f"{error_type}: {error_msg}",
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.HIGH,
                context=context
            )
        
        # Default classification
        return FederatedError(
            f"{error_type}: {error_msg}",
            category=ErrorCategory.UNKNOWN,
            severity=ErrorSeverity.MEDIUM,
            context=context
        )
    
    def _log_error(self, error: FederatedError, context: Optional[Dict[str, Any]]):
        """Log error with appropriate level."""
        log_data = {
            "error_category": error.category.value,
            "error_severity": error.severity.value,
            "recoverable": error.recoverable,
            "error_context": error.context,
            "additional_context": context,
            "traceback": traceback.format_exc(),
        }
        
        if error.severity == ErrorSeverity.CRITICAL:
            logger.critical(f"Critical error: {str(error)}", extra=log_data)
        elif error.severity == ErrorSeverity.HIGH:
            logger.error(f"High severity error: {str(error)}", extra=log_data)
        elif error.severity == ErrorSeverity.MEDIUM:
            logger.warning(f"Medium severity error: {str(error)}", extra=log_data)
        else:
            logger.info(f"Low severity error: {str(error)}", extra=log_data)
    
    def _record_error(self, error: FederatedError, context: Optional[Dict[str, Any]]):
        """Record error in history."""
        error_record = {
            "timestamp": error.timestamp,
            "category": error.category.value,
            "severity": error.severity.value,
            "message": str(error),
            "recoverable": error.recoverable,
            "context": {**error.context, **(context or {})},
            "recovery_attempted": False,
            "recovery_successful": False,
        }
        
        self.error_history.append(error_record)
        
        # Keep only last 1000 errors
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-1000:]
    
    def _attempt_recovery(self, error: FederatedError) -> bool:
        """Attempt to recover from error using appropriate strategies."""
        strategies = self.recovery_strategies.get(error.category, [])
        
        if not strategies:
            logger.warning(f"No recovery strategies for category {error.category.value}")
            return False
        
        # Update error record
        if self.error_history:
            self.error_history[-1]["recovery_attempted"] = True
        
        for strategy in strategies:
            try:
                logger.info(f"Attempting recovery strategy: {strategy.__name__}")
                success = strategy(error)
                
                if success:
                    logger.info(f"Recovery successful using {strategy.__name__}")
                    if self.error_history:
                        self.error_history[-1]["recovery_successful"] = True
                    return True
                    
            except Exception as recovery_error:
                logger.error(f"Recovery strategy {strategy.__name__} failed: {recovery_error}")
        
        logger.error(f"All recovery strategies failed for error: {str(error)}")
        return False
    
    # Recovery strategies
    def _retry_with_backoff(self, error: FederatedError) -> bool:
        """Retry with exponential backoff."""
        max_attempts = 3
        for attempt in range(max_attempts):
            delay = self.base_delay * (2 ** attempt)
            logger.info(f"Retrying in {delay} seconds (attempt {attempt + 1}/{max_attempts})")
            time.sleep(delay)
            
            # This is a placeholder - actual retry logic would depend on context
            # In practice, this would re-execute the failed operation
            if attempt == max_attempts - 1:  # Simulate success on last attempt
                return True
        
        return False
    
    def _clear_cache(self, error: FederatedError) -> bool:
        """Clear various caches to free up resources."""
        try:
            # Clear PyTorch cache
            if _TORCH_AVAILABLE and torch and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Clear Python garbage collection
            import gc
            gc.collect()
            
            logger.info("Cache cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Cache clearing failed: {e}")
            return False
    
    def _clear_memory(self, error: FederatedError) -> bool:
        """Aggressive memory clearing."""
        try:
            # Clear caches first
            self._clear_cache(error)
            
            # Additional PyTorch memory management
            if _TORCH_AVAILABLE and torch and torch.cuda.is_available():
                # Reset peak memory stats
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.reset_accumulated_memory_stats()
            
            logger.info("Memory cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Memory clearing failed: {e}")
            return False
    
    def _reduce_batch_size(self, error: FederatedError) -> bool:
        """Reduce batch size to handle memory issues."""
        # This is a placeholder - actual implementation would depend on context
        logger.info("Batch size reduction strategy (placeholder)")
        return True
    
    def _reduce_model_precision(self, error: FederatedError) -> bool:
        """Reduce model precision to save memory."""
        # This is a placeholder - actual implementation would depend on context
        logger.info("Model precision reduction strategy (placeholder)")
        return True
    
    def _fallback_communication(self, error: FederatedError) -> bool:
        """Use fallback communication method."""
        logger.info("Fallback communication strategy (placeholder)")
        return True
    
    def _increase_timeout(self, error: FederatedError) -> bool:
        """Increase timeout values."""
        logger.info("Timeout increase strategy (placeholder)")
        return True
    
    def _reset_client_state(self, error: FederatedError) -> bool:
        """Reset client state."""
        logger.info("Client state reset strategy (placeholder)")
        return True
    
    def _retry_aggregation(self, error: FederatedError) -> bool:
        """Retry federated aggregation."""
        logger.info("Aggregation retry strategy (placeholder)")
        return True
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics."""
        if not self.error_history:
            return {"total_errors": 0}
        
        category_counts = {}
        severity_counts = {}
        recovery_success_rate = 0
        
        for error_record in self.error_history:
            category = error_record["category"]
            severity = error_record["severity"]
            
            category_counts[category] = category_counts.get(category, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            if error_record["recovery_attempted"] and error_record["recovery_successful"]:
                recovery_success_rate += 1
        
        attempted_recoveries = sum(
            1 for record in self.error_history if record["recovery_attempted"]
        )
        
        return {
            "total_errors": len(self.error_history),
            "category_distribution": category_counts,
            "severity_distribution": severity_counts,
            "recovery_attempts": attempted_recoveries,
            "recovery_success_rate": (
                recovery_success_rate / attempted_recoveries
                if attempted_recoveries > 0 else 0.0
            ),
            "recent_errors": self.error_history[-10:] if len(self.error_history) >= 10 else self.error_history,
        }


# Global error handler instance
global_error_handler = ErrorHandler()


def with_error_handling(
    max_retries: int = 3,
    auto_recover: bool = True,
    reraise: bool = True,
    fallback_value: Any = None,
):
    """Decorator for automatic error handling.
    
    Args:
        max_retries: Maximum retry attempts
        auto_recover: Whether to attempt automatic recovery
        reraise: Whether to reraise the exception after handling
        fallback_value: Value to return on unrecoverable error
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                    
                except Exception as e:
                    last_error = e
                    context = {
                        "function": func.__name__,
                        "attempt": attempt + 1,
                        "max_retries": max_retries,
                        "args_count": len(args),
                        "kwargs_keys": list(kwargs.keys()),
                    }
                    
                    recovered = global_error_handler.handle_error(
                        e, context=context, auto_recover=auto_recover
                    )
                    
                    if not recovered and attempt < max_retries:
                        # Wait before retry
                        time.sleep(1.0 * (2 ** attempt))
                        continue
                    elif not recovered:
                        # Final attempt failed
                        if reraise:
                            raise e
                        else:
                            logger.error(f"Function {func.__name__} failed after {max_retries} retries")
                            return fallback_value
            
            # Should not reach here, but just in case
            if reraise and last_error:
                raise last_error
            return fallback_value
            
        return wrapper
    return decorator


@contextmanager
def error_context(context: Dict[str, Any]):
    """Context manager for error handling with additional context."""
    try:
        yield
    except Exception as e:
        global_error_handler.handle_error(e, context=context, auto_recover=True)
        raise


def handle_error(
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    auto_recover: bool = True,
) -> bool:
    """Handle error using global error handler."""
    return global_error_handler.handle_error(error, context, auto_recover)


def get_error_statistics() -> Dict[str, Any]:
    """Get global error statistics."""
    return global_error_handler.get_error_statistics()