"""Centralized logging configuration for Fed-ViT-AutoRL."""

import logging
import logging.handlers
import sys
import os
from typing import Optional, Dict, Any
import json
from datetime import datetime


class StructuredFormatter(logging.Formatter):
    """Structured JSON formatter for logs."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_data = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception information if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields if present
        for key, value in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message"
            }:
                log_data[key] = value

        return json.dumps(log_data, default=str)


class FederatedLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter for federated learning context."""

    def process(self, msg, kwargs):
        """Add federated context to log messages."""
        extra = kwargs.get("extra", {})

        # Add federated context from adapter
        if hasattr(self, "extra"):
            extra.update(self.extra)

        kwargs["extra"] = extra
        return msg, kwargs


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    structured_logging: bool = True,
    max_file_size: int = 100 * 1024 * 1024,  # 100MB
    backup_count: int = 5,
    federated_context: Optional[Dict[str, Any]] = None,
) -> logging.Logger:
    """Setup centralized logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file to write logs to
        structured_logging: Whether to use structured JSON logging
        max_file_size: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
        federated_context: Additional context for federated learning

    Returns:
        Configured logger
    """
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))

    if structured_logging:
        console_formatter = StructuredFormatter()
    else:
        console_formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        # Ensure directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding="utf-8"
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))

        if structured_logging:
            file_formatter = StructuredFormatter()
        else:
            file_formatter = logging.Formatter(
                fmt="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )

        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # Setup specific loggers
    fed_vit_logger = logging.getLogger("fed_vit_autorl")

    # Create federated logger adapter if context provided
    if federated_context:
        fed_vit_logger = FederatedLoggerAdapter(fed_vit_logger, federated_context)

    # Reduce noise from external libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    # PyTorch specific
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)

    fed_vit_logger.info("Logging system initialized", extra={
        "log_level": log_level,
        "structured_logging": structured_logging,
        "log_file": log_file,
        "federated_context": federated_context is not None,
    })

    return fed_vit_logger


def get_federated_logger(
    component: str,
    client_id: Optional[str] = None,
    round_number: Optional[int] = None,
    additional_context: Optional[Dict[str, Any]] = None,
) -> logging.Logger:
    """Get logger with federated learning context.

    Args:
        component: Component name (e.g., 'client', 'server', 'aggregator')
        client_id: Optional client identifier
        round_number: Optional federated round number
        additional_context: Additional context to include

    Returns:
        Logger with federated context
    """
    logger_name = f"fed_vit_autorl.{component}"
    base_logger = logging.getLogger(logger_name)

    # Build context
    context = {"component": component}

    if client_id:
        context["client_id"] = client_id

    if round_number is not None:
        context["round"] = round_number

    if additional_context:
        context.update(additional_context)

    return FederatedLoggerAdapter(base_logger, context)


class LoggingConfig:
    """Global logging configuration manager."""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.config = {
                "log_level": "INFO",
                "structured_logging": True,
                "log_file": None,
                "max_file_size": 100 * 1024 * 1024,
                "backup_count": 5,
            }
            self._initialized = True

    def configure(self, **kwargs):
        """Update logging configuration."""
        self.config.update(kwargs)

        # Re-setup logging with new configuration
        setup_logging(**self.config)

    def get_logger(self, name: str) -> logging.Logger:
        """Get logger with current configuration."""
        return logging.getLogger(name)

    def get_federated_logger(
        self,
        component: str,
        **context
    ) -> logging.Logger:
        """Get federated logger with context."""
        return get_federated_logger(component, **context)


# Global logging configuration instance
logging_config = LoggingConfig()


# Convenience functions
def configure_logging(**kwargs):
    """Configure global logging settings."""
    logging_config.configure(**kwargs)


def get_logger(name: str) -> logging.Logger:
    """Get logger with current configuration."""
    return logging_config.get_logger(name)


def get_federated_logger(component: str, **context) -> logging.Logger:
    """Get federated logger with context."""
    return logging_config.get_federated_logger(component, **context)
