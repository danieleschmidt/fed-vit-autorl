#!/usr/bin/env python3
"""Test Generation 2 robustness components without external dependencies."""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

def test_logging_system():
    """Test logging configuration."""
    print("Testing logging system...")
    try:
        from fed_vit_autorl.logging_config import setup_logging, get_federated_logger

        # Setup basic logging
        logger = setup_logging(log_level='INFO', structured_logging=True)

        # Test federated logger
        fed_logger = get_federated_logger('client', client_id='test_001')
        fed_logger.info('Test federated logging message')

        print("✅ Logging system working!")
        return True
    except Exception as e:
        print(f"❌ Logging test failed: {e}")
        return False

def test_error_handling():
    """Test error handling system."""
    print("Testing error handling...")
    try:
        from fed_vit_autorl.error_handling import ErrorHandler, with_error_handling, FederatedError, ErrorCategory

        # Test error handler
        error_handler = ErrorHandler()

        # Test error classification
        test_error = ValueError("Test validation error")
        fed_error = error_handler._classify_error(test_error)

        print(f"   Error classified as: {fed_error.category.value}")

        # Test decorator
        @with_error_handling(max_retries=2, reraise=False, fallback_value='recovered')
        def test_function():
            raise ValueError('Test error')

        result = test_function()
        print(f"   Error handling result: {result}")

        print("✅ Error handling working!")
        return True
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def test_validation_system():
    """Test validation system."""
    print("Testing validation system...")
    try:
        from fed_vit_autorl.validation.input_validator import InputValidator, FederatedInputValidator

        # Test basic validator
        validator = InputValidator()

        # Test string validation
        string_result = validator.validate_string_input("test_string", max_length=20)
        print(f"   String validation: {string_result.is_valid}")

        # Test numeric validation
        numeric_result = validator.validate_numeric_input(42.0, min_val=0, max_val=100)
        print(f"   Numeric validation: {numeric_result.is_valid}")

        # Test federated validator
        fed_validator = FederatedInputValidator()
        client_result = fed_validator.validate_client_id('vehicle_001')
        print(f"   Client ID validation: {client_result.is_valid}")

        print("✅ Validation system working!")
        return True
    except Exception as e:
        print(f"❌ Validation test failed: {e}")
        return False

def test_health_checker():
    """Test health monitoring system."""
    print("Testing health checker...")
    try:
        from fed_vit_autorl.monitoring.health_checker import HealthChecker, FederatedHealthChecker, HealthStatus

        # Test basic health checker
        health_checker = HealthChecker()

        # Run system checks
        checks = health_checker.run_all_checks()
        print(f"   System health checks completed: {len(checks)} checks")

        # Check overall status
        overall_status = health_checker.get_overall_status()
        print(f"   Overall health status: {overall_status.value}")

        # Test federated health checker
        fed_checker = FederatedHealthChecker()

        # Test federation health check
        fed_health = fed_checker.check_federation_health(
            participation_rate=0.8,
            avg_communication_latency=2.5,
            privacy_budget_remaining=0.7,
            last_aggregation_time=45.0
        )
        print(f"   Federation health: {fed_health.status.value}")

        print("✅ Health checker working!")
        return True
    except Exception as e:
        print(f"❌ Health checker test failed: {e}")
        return False

def main():
    """Run all Generation 2 tests."""
    print("🚀 Testing Fed-ViT-AutoRL Generation 2: Robustness Components\n")

    tests = [
        test_logging_system,
        test_error_handling,
        test_validation_system,
        test_health_checker,
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
        print("🎉 All Generation 2 robustness components are working perfectly!")
        return True
    else:
        print("⚠️  Some tests failed - check implementation")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
