#!/usr/bin/env python3
"""Test Generation 2 robustness components without torch."""

import sys
import os

def test_logging_config():
    """Test logging configuration module."""
    print("Testing logging configuration...")
    try:
        # Test the logging module directly
        sys.path.insert(0, os.path.abspath('.'))

        import json
        import logging
        import time
        from fed_vit_autorl.logging_config import StructuredFormatter, FederatedLoggerAdapter

        # Test structured formatter
        formatter = StructuredFormatter()

        # Create a log record
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )

        formatted = formatter.format(record)
        data = json.loads(formatted)

        assert "timestamp" in data
        assert data["level"] == "INFO"
        assert data["message"] == "Test message"

        print("✅ Logging configuration working!")
        return True
    except Exception as e:
        print(f"❌ Logging test failed: {e}")
        return False

def test_error_handling_core():
    """Test core error handling without torch."""
    print("Testing core error handling...")
    try:
        from fed_vit_autorl.error_handling import (
            FederatedError, ClientError, ServerError, ErrorCategory, ErrorSeverity
        )

        # Test error creation
        error = FederatedError(
            "Test error",
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.HIGH
        )

        assert error.category == ErrorCategory.VALIDATION
        assert error.severity == ErrorSeverity.HIGH
        assert error.recoverable == True

        # Test client error
        client_error = ClientError("Client failed", client_id="test_client")
        assert client_error.client_id == "test_client"
        assert client_error.context["client_id"] == "test_client"

        print("✅ Error handling core working!")
        return True
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def test_validation_core():
    """Test validation without torch tensors."""
    print("Testing validation core...")
    try:
        from fed_vit_autorl.validation.input_validator import InputValidator, ValidationResult

        # Test validator creation
        validator = InputValidator(strict_mode=True)

        # Test string validation
        result = validator.validate_string_input(
            "test_string",
            max_length=50,
            name="test_input"
        )

        assert result.is_valid == True
        assert result.sanitized_input == "test_string"

        # Test numeric validation
        result = validator.validate_numeric_input(
            42.5,
            min_val=0,
            max_val=100,
            name="test_number"
        )

        assert result.is_valid == True
        assert result.sanitized_input == 42.5

        # Test invalid input
        result = validator.validate_string_input(
            "x" * 2000,  # Too long
            max_length=10
        )

        if validator.strict_mode:
            assert result.is_valid == False

        print("✅ Validation core working!")
        return True
    except Exception as e:
        print(f"❌ Validation test failed: {e}")
        return False

def test_health_monitoring_core():
    """Test health monitoring without system dependencies."""
    print("Testing health monitoring core...")
    try:
        from fed_vit_autorl.monitoring.health_checker import HealthCheck, HealthStatus

        # Test health check creation
        health_check = HealthCheck(
            name="test_check",
            status=HealthStatus.HEALTHY,
            message="All systems normal",
            timestamp=time.time(),
            metrics={"cpu_usage": 45.2},
            remediation=None
        )

        assert health_check.status == HealthStatus.HEALTHY
        assert health_check.metrics["cpu_usage"] == 45.2

        # Test status enum
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.CRITICAL.value == "critical"

        print("✅ Health monitoring core working!")
        return True
    except Exception as e:
        print(f"❌ Health monitoring test failed: {e}")
        return False

def main():
    """Run minimal Generation 2 tests."""
    print("🚀 Testing Fed-ViT-AutoRL Generation 2: Core Robustness\n")

    import time

    tests = [
        test_logging_config,
        test_error_handling_core,
        test_validation_core,
        test_health_monitoring_core,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}\n")

    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 Generation 2 core robustness components working!")
        return True
    else:
        print("⚠️  Some core tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
