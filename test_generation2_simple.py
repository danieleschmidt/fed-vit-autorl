#!/usr/bin/env python3
"""Simple test for Generation 2 robustness components."""

import sys
import os
import time
sys.path.insert(0, os.path.abspath('.'))

def test_logging_config():
    """Test logging configuration module."""
    print("Testing logging configuration...")
    try:
        from fed_vit_autorl.logging_config import StructuredFormatter

        # Test structured formatter
        formatter = StructuredFormatter()
        print("✅ Logging configuration working!")
        return True
    except Exception as e:
        print(f"❌ Logging test failed: {e}")
        return False

def test_error_handling():
    """Test error handling."""
    print("Testing error handling...")
    try:
        from fed_vit_autorl.error_handling import FederatedError, ErrorCategory, ErrorSeverity

        # Test error creation
        error = FederatedError(
            "Test error",
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.HIGH
        )

        assert error.category == ErrorCategory.VALIDATION
        print("✅ Error handling working!")
        return True
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def test_validation():
    """Test validation."""
    print("Testing validation...")
    try:
        from fed_vit_autorl.validation.input_validator import InputValidator

        # Test validator creation
        validator = InputValidator(strict_mode=True)

        # Simple string test
        result = validator.validate_string_input("test", max_length=10)
        assert result.is_valid == True

        print("✅ Validation working!")
        return True
    except Exception as e:
        print(f"❌ Validation test failed: {e}")
        return False

def test_health_monitoring():
    """Test health monitoring."""
    print("Testing health monitoring...")
    try:
        from fed_vit_autorl.monitoring.health_checker import HealthCheck, HealthStatus

        # Test health check creation
        health_check = HealthCheck(
            name="test_check",
            status=HealthStatus.HEALTHY,
            message="All systems normal",
            timestamp=time.time(),
            metrics={"test": 1}
        )

        assert health_check.status == HealthStatus.HEALTHY
        print("✅ Health monitoring working!")
        return True
    except Exception as e:
        print(f"❌ Health monitoring test failed: {e}")
        return False

def main():
    """Run simple Generation 2 tests."""
    print("🚀 Testing Fed-ViT-AutoRL Generation 2: Simple Robustness Test\\n")

    tests = [
        test_logging_config,
        test_error_handling,
        test_validation,
        test_health_monitoring,
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
        print("🎉 Generation 2 robustness components are working!")
        return True
    else:
        print("⚠️  Some robustness tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
