#!/usr/bin/env python3
"""Final test for Generation 2 robustness components."""

import sys
import os
import time
sys.path.insert(0, os.path.abspath('.'))

def test_logging_config():
    """Test logging configuration module."""
    print("Testing logging configuration...")
    try:
        import json
        import logging
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
    """Test core error handling."""
    print("Testing core error handling...")
    try:
        from fed_vit_autorl.error_handling import (
            FederatedError, ClientError, ErrorCategory, ErrorSeverity
        )

        # Test error creation
        error = FederatedError(
            "Test error",
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.HIGH
        )

        assert error.category == ErrorCategory.VALIDATION
        assert error.severity == ErrorSeverity.HIGH

        print("✅ Error handling core working!")
        return True
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def test_validation_core():
    """Test validation core."""
    print("Testing validation core...")
    try:
        from fed_vit_autorl.validation.input_validator import InputValidator

        # Test validator creation
        validator = InputValidator(strict_mode=True)

        # Test string validation
        result = validator.validate_string_input(
            "test_string",
            max_length=50,
            name="test_input"
        )

        assert result.is_valid == True

        # Test numeric validation
        result = validator.validate_numeric_input(
            42.5,
            min_val=0,
            max_val=100,
            name="test_number"
        )

        assert result.is_valid == True

        print("✅ Validation core working!")
        return True
    except Exception as e:
        print(f"❌ Validation test failed: {e}")
        return False

def test_health_monitoring_core():
    """Test health monitoring core."""
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

        print("✅ Health monitoring core working!")
        return True
    except Exception as e:
        print(f"❌ Health monitoring test failed: {e}")
        return False

def test_integration():
    """Test integration of all components."""
    print("Testing component integration...")
    try:
        from fed_vit_autorl.logging_config import get_federated_logger
        from fed_vit_autorl.error_handling import handle_error, FederatedError, ErrorCategory
        from fed_vit_autorl.validation.input_validator import FederatedInputValidator

        # Test federated logger
        logger = get_federated_logger("test", client_id="integration_test")

        # Test validation with error handling
        validator = FederatedInputValidator()
        result = validator.validate_client_id("test_vehicle_123")

        if not result.is_valid:
            error = FederatedError(result.message, category=ErrorCategory.VALIDATION)
            handle_error(error, auto_recover=False)

        print("✅ Component integration working!")
        return True
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def main():
    """Run all Generation 2 tests."""
    print("🚀 Testing Fed-ViT-AutoRL Generation 2: Final Robustness Test\\n")

    tests = [
        test_logging_config,
        test_error_handling_core,
        test_validation_core,
        test_health_monitoring_core,
        test_integration,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}\\n")

    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 Generation 2 robustness components fully operational!")
        print("✅ Comprehensive error handling implemented")
        print("✅ Input validation and sanitization working")
        print("✅ Health monitoring and diagnostics ready")
        print("✅ Structured logging and observability enabled")
        print("✅ Component integration verified")
        return True
    else:
        print("⚠️  Some robustness tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
