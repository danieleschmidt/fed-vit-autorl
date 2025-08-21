"""Comprehensive test runner for Fed-ViT-AutoRL."""

import unittest
import logging
import time
import sys
import os
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import importlib
import traceback
import concurrent.futures
import threading

logger = logging.getLogger(__name__)


class TestResult(Enum):
    """Test result status."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class TestReport:
    """Individual test report."""
    test_name: str
    result: TestResult
    execution_time: float
    message: str = ""
    error_traceback: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class RobustTestRunner:
    """Robust test runner with comprehensive reporting."""

    def __init__(self,
                 timeout_per_test: float = 30.0,
                 parallel_tests: bool = True,
                 max_workers: int = 4):
        """Initialize test runner.

        Args:
            timeout_per_test: Maximum time per test in seconds
            parallel_tests: Whether to run tests in parallel
            max_workers: Maximum number of parallel workers
        """
        self.timeout_per_test = timeout_per_test
        self.parallel_tests = parallel_tests
        self.max_workers = max_workers

        self.test_results: List[TestReport] = []
        self.test_functions: List[Callable] = []
        self._lock = threading.Lock()

        logger.info(f"Initialized test runner (parallel={parallel_tests}, workers={max_workers})")

    def add_test(self, test_func: Callable, test_name: Optional[str] = None) -> None:
        """Add test function to runner.

        Args:
            test_func: Test function to add
            test_name: Optional custom test name
        """
        if test_name is None:
            test_name = getattr(test_func, '__name__', 'unnamed_test')

        # Wrap function with name
        if hasattr(test_func, '__self__'):
            # It's a bound method, create a wrapper function
            def wrapper():
                return test_func()
            wrapper._test_name = test_name
            self.test_functions.append(wrapper)
        else:
            test_func._test_name = test_name
            self.test_functions.append(test_func)

        logger.debug(f"Added test: {test_name}")

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all registered tests.

        Returns:
            Comprehensive test results
        """
        start_time = time.time()
        self.test_results.clear()

        logger.info(f"Starting test run with {len(self.test_functions)} tests")

        if self.parallel_tests and len(self.test_functions) > 1:
            self._run_parallel_tests()
        else:
            self._run_sequential_tests()

        total_time = time.time() - start_time

        # Generate summary
        passed = sum(1 for r in self.test_results if r.result == TestResult.PASSED)
        failed = sum(1 for r in self.test_results if r.result == TestResult.FAILED)
        errors = sum(1 for r in self.test_results if r.result == TestResult.ERROR)
        skipped = sum(1 for r in self.test_results if r.result == TestResult.SKIPPED)

        summary = {
            'total_tests': len(self.test_results),
            'passed': passed,
            'failed': failed,
            'errors': errors,
            'skipped': skipped,
            'execution_time': total_time,
            'success_rate': (passed / max(len(self.test_results), 1)) * 100,
            'results': [
                {
                    'test_name': r.test_name,
                    'result': r.result.value,
                    'execution_time': r.execution_time,
                    'message': r.message
                }
                for r in self.test_results
            ]
        }

        logger.info(f"Test run completed: {passed}/{len(self.test_results)} passed ({summary['success_rate']:.1f}%)")

        return summary

    def _run_parallel_tests(self) -> None:
        """Run tests in parallel."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tests
            future_to_test = {
                executor.submit(self._run_single_test, test_func): test_func
                for test_func in self.test_functions
            }

            # Collect results
            for future in concurrent.futures.as_completed(future_to_test):
                try:
                    result = future.result(timeout=self.timeout_per_test)
                    with self._lock:
                        self.test_results.append(result)
                except concurrent.futures.TimeoutError:
                    test_func = future_to_test[future]
                    test_name = getattr(test_func, '_test_name', 'unknown')
                    result = TestReport(
                        test_name=test_name,
                        result=TestResult.ERROR,
                        execution_time=self.timeout_per_test,
                        message=f"Test timed out after {self.timeout_per_test} seconds"
                    )
                    with self._lock:
                        self.test_results.append(result)
                except Exception as e:
                    test_func = future_to_test[future]
                    test_name = getattr(test_func, '_test_name', 'unknown')
                    result = TestReport(
                        test_name=test_name,
                        result=TestResult.ERROR,
                        execution_time=0.0,
                        message=f"Test execution error: {str(e)}",
                        error_traceback=traceback.format_exc()
                    )
                    with self._lock:
                        self.test_results.append(result)

    def _run_sequential_tests(self) -> None:
        """Run tests sequentially."""
        for test_func in self.test_functions:
            result = self._run_single_test(test_func)
            self.test_results.append(result)

    def _run_single_test(self, test_func: Callable) -> TestReport:
        """Run a single test function.

        Args:
            test_func: Test function to run

        Returns:
            Test report
        """
        test_name = getattr(test_func, '_test_name', 'unknown')
        start_time = time.time()

        try:
            # Run the test
            result = test_func()
            execution_time = time.time() - start_time

            # Determine result status
            if result is None or result is True:
                status = TestResult.PASSED
                message = "Test passed"
            elif result is False:
                status = TestResult.FAILED
                message = "Test returned False"
            elif isinstance(result, str):
                # String result indicates failure with message
                status = TestResult.FAILED
                message = result
            else:
                status = TestResult.PASSED
                message = f"Test completed: {str(result)}"

            logger.debug(f"Test {test_name}: {status.value} ({execution_time:.3f}s)")

            return TestReport(
                test_name=test_name,
                result=status,
                execution_time=execution_time,
                message=message
            )

        except unittest.SkipTest as e:
            execution_time = time.time() - start_time
            return TestReport(
                test_name=test_name,
                result=TestResult.SKIPPED,
                execution_time=execution_time,
                message=str(e)
            )

        except AssertionError as e:
            execution_time = time.time() - start_time
            return TestReport(
                test_name=test_name,
                result=TestResult.FAILED,
                execution_time=execution_time,
                message=str(e),
                error_traceback=traceback.format_exc()
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return TestReport(
                test_name=test_name,
                result=TestResult.ERROR,
                execution_time=execution_time,
                message=f"Unexpected error: {str(e)}",
                error_traceback=traceback.format_exc()
            )

    def run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests for core functionality."""
        integration_tests = [
            self._test_basic_imports,
            self._test_error_handling,
            self._test_input_validation,
            self._test_logging_system,
            self._test_health_monitoring,
            self._test_security_features,
        ]

        # Clear existing tests and add integration tests
        original_tests = self.test_functions.copy()
        self.test_functions.clear()

        for test in integration_tests:
            self.add_test(test)

        results = self.run_all_tests()

        # Restore original tests
        self.test_functions = original_tests

        return results

    def _test_basic_imports(self) -> bool:
        """Test that basic imports work."""
        try:
            import fed_vit_autorl
            from fed_vit_autorl.error_handling import FederatedError
            from fed_vit_autorl.validation.input_validator import InputValidator
            return True
        except ImportError as e:
            raise AssertionError(f"Import failed: {e}")

    def _test_error_handling(self) -> bool:
        """Test error handling system."""
        try:
            from fed_vit_autorl.error_handling import FederatedError, ErrorCategory

            # Test error creation
            error = FederatedError("Test error", "TEST_CODE")
            assert str(error) == "Test error", "Error string representation failed"

            # Test error with category
            error_with_cat = FederatedError("Network error", "NET_001", ErrorCategory.NETWORK)
            assert error_with_cat.category == ErrorCategory.NETWORK, "Error category not set"

            return True
        except Exception as e:
            raise AssertionError(f"Error handling test failed: {e}")

    def _test_input_validation(self) -> bool:
        """Test input validation system."""
        try:
            from fed_vit_autorl.validation.input_validator import InputValidator

            validator = InputValidator()

            # Test numeric validation
            result = validator.validate_numeric_input(5.0, min_val=0.0, max_val=10.0)
            assert result.is_valid, "Valid numeric input rejected"

            # Test invalid numeric input
            result = validator.validate_numeric_input(-1.0, min_val=0.0, max_val=10.0)
            assert not result.is_valid, "Invalid numeric input accepted"

            return True
        except Exception as e:
            raise AssertionError(f"Input validation test failed: {e}")

    def _test_logging_system(self) -> bool:
        """Test logging system."""
        try:
            from fed_vit_autorl.logging_config import setup_logging

            # Test basic logging setup
            logger = setup_logging('test_logger', 'INFO', structured_logging=False)
            assert logger is not None, "Logger setup failed"

            # Test log message
            logger.info("Test log message")

            return True
        except Exception as e:
            raise AssertionError(f"Logging test failed: {e}")

    def _test_health_monitoring(self) -> bool:
        """Test health monitoring system."""
        try:
            from fed_vit_autorl.monitoring.health_checker import HealthChecker

            health_checker = HealthChecker()
            result = health_checker.check_health()

            assert 'status' in result, "Health check missing status"
            assert 'timestamp' in result, "Health check missing timestamp"

            return True
        except Exception as e:
            raise AssertionError(f"Health monitoring test failed: {e}")

    def _test_security_features(self) -> bool:
        """Test security features."""
        try:
            from fed_vit_autorl.security import SecurityManager

            security_manager = SecurityManager()

            # Test encryption/decryption
            test_data = "sensitive_data"
            encrypted = security_manager.encrypt_data(test_data)
            decrypted = security_manager.decrypt_data(encrypted)

            assert decrypted.decode() == test_data, "Encryption/decryption failed"

            # Test input validation
            safe_input = security_manager.validate_input("normal_text", "message")
            assert safe_input, "Safe input rejected"

            dangerous_input = security_manager.validate_input("<script>alert('xss')</script>", "message")
            assert not dangerous_input, "Dangerous input accepted"

            return True
        except Exception as e:
            raise AssertionError(f"Security test failed: {e}")

    def generate_test_report(self, output_file: Optional[str] = None) -> str:
        """Generate detailed test report.

        Args:
            output_file: Optional file to write report to

        Returns:
            Test report as string
        """
        report_lines = [
            "=" * 80,
            "Fed-ViT-AutoRL Test Report",
            "=" * 80,
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Tests: {len(self.test_results)}",
            ""
        ]

        # Summary
        passed = sum(1 for r in self.test_results if r.result == TestResult.PASSED)
        failed = sum(1 for r in self.test_results if r.result == TestResult.FAILED)
        errors = sum(1 for r in self.test_results if r.result == TestResult.ERROR)
        skipped = sum(1 for r in self.test_results if r.result == TestResult.SKIPPED)

        report_lines.extend([
            "Summary:",
            f"  ✅ Passed:  {passed}",
            f"  ❌ Failed:  {failed}",
            f"  💥 Errors:  {errors}",
            f"  ⏭️  Skipped: {skipped}",
            f"  📊 Success Rate: {(passed/max(len(self.test_results), 1))*100:.1f}%",
            ""
        ])

        # Detailed results
        report_lines.append("Detailed Results:")
        report_lines.append("-" * 40)

        for result in self.test_results:
            status_emoji = {
                TestResult.PASSED: "✅",
                TestResult.FAILED: "❌",
                TestResult.ERROR: "💥",
                TestResult.SKIPPED: "⏭️"
            }

            report_lines.append(
                f"{status_emoji[result.result]} {result.test_name} "
                f"({result.execution_time:.3f}s) - {result.message}"
            )

            if result.error_traceback:
                report_lines.extend([
                    "   Traceback:",
                    "   " + result.error_traceback.replace("\n", "\n   "),
                    ""
                ])

        report = "\n".join(report_lines)

        if output_file:
            try:
                with open(output_file, 'w') as f:
                    f.write(report)
                logger.info(f"Test report written to {output_file}")
            except Exception as e:
                logger.error(f"Failed to write report to {output_file}: {e}")

        return report
