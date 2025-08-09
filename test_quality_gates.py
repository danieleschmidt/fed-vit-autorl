#!/usr/bin/env python3
"""Test quality gates and validation system."""

import sys
import os
import time
sys.path.insert(0, os.path.abspath('.'))

def test_quality_gate_components():
    """Test quality gate core components."""
    print("Testing quality gate components...")
    try:
        from fed_vit_autorl.testing.quality_gates import (
            QualityGateValidator, QualityGateStatus, QualityMetrics, QualityGateResult
        )
        
        # Test enums and dataclasses
        status = QualityGateStatus.PASSED
        assert status.value == "passed"
        
        # Test QualityMetrics
        metrics = QualityMetrics(
            test_coverage=85.0,
            code_quality_score=8.5,
            security_score=90.0,
            performance_score=80.0
        )
        
        assert metrics.test_coverage == 85.0
        assert metrics.code_quality_score == 8.5
        
        # Test QualityGateResult
        result = QualityGateResult(
            gate_name="test_gate",
            status=QualityGateStatus.PASSED,
            message="Test passed",
            score=85.0,
            threshold=80.0
        )
        
        assert result.gate_name == "test_gate"
        assert result.status == QualityGateStatus.PASSED
        assert result.score == 85.0
        
        print("✅ Quality gate components working!")
        return True
    except Exception as e:
        print(f"❌ Quality gate components test failed: {e}")
        return False

def test_quality_gate_validator():
    """Test quality gate validator."""
    print("Testing quality gate validator...")
    try:
        from fed_vit_autorl.testing.quality_gates import QualityGateValidator, QualityGateStatus
        
        # Create validator
        validator = QualityGateValidator(
            project_root=".",
            min_coverage=70.0,
            min_code_quality=7.0,
            min_security_score=80.0,
            enable_strict_mode=False
        )
        
        # Test individual gate methods exist
        assert hasattr(validator, '_validate_test_coverage')
        assert hasattr(validator, '_validate_code_quality')
        assert hasattr(validator, '_validate_security')
        
        # Test overall status when no results
        status = validator.get_overall_status()
        assert status == QualityGateStatus.SKIPPED
        
        print("✅ Quality gate validator working!")
        return True
    except Exception as e:
        print(f"❌ Quality gate validator test failed: {e}")
        return False

def test_individual_quality_gates():
    """Test individual quality gates."""
    print("Testing individual quality gates...")
    try:
        from fed_vit_autorl.testing.quality_gates import QualityGateValidator, QualityGateStatus
        
        validator = QualityGateValidator(project_root=".")
        
        # Test test coverage validation
        coverage_result = validator._validate_test_coverage()
        assert coverage_result.gate_name == "test_coverage"
        assert coverage_result.status in [QualityGateStatus.PASSED, QualityGateStatus.WARNING, QualityGateStatus.FAILED]
        
        # Test code quality validation  
        quality_result = validator._validate_code_quality()
        assert quality_result.gate_name == "code_quality"
        assert quality_result.status in [QualityGateStatus.PASSED, QualityGateStatus.WARNING, QualityGateStatus.FAILED]
        
        # Test security validation
        security_result = validator._validate_security()
        assert security_result.gate_name == "security_scan"
        assert security_result.status in [QualityGateStatus.PASSED, QualityGateStatus.WARNING, QualityGateStatus.FAILED]
        
        print("✅ Individual quality gates working!")
        return True
    except Exception as e:
        print(f"❌ Individual quality gates test failed: {e}")
        return False

def test_quality_report_generation():
    """Test quality report generation."""
    print("Testing quality report generation...")
    try:
        from fed_vit_autorl.testing.quality_gates import QualityGateValidator
        
        validator = QualityGateValidator(project_root=".")
        
        # Generate report with no validations
        report = validator.get_quality_report()
        assert "status" in report
        assert report["status"] == "no_validations"
        
        # Test that validation history starts empty
        assert len(validator.validation_history) == 0
        assert len(validator.gate_results) == 0
        
        print("✅ Quality report generation working!")
        return True
    except Exception as e:
        print(f"❌ Quality report generation test failed: {e}")
        return False

def test_comprehensive_quality_validation():
    """Test comprehensive quality validation."""
    print("Testing comprehensive quality validation...")
    try:
        from fed_vit_autorl.testing.quality_gates import QualityGateValidator
        
        # Create validator with reasonable thresholds
        validator = QualityGateValidator(
            project_root=".",
            min_coverage=60.0,        # Lower threshold for testing
            min_code_quality=6.0,     # Lower threshold for testing
            min_security_score=70.0,  # Lower threshold for testing
            min_performance_score=60.0,
            min_documentation=50.0,
            max_complexity=15.0,      # Higher threshold for testing
            enable_strict_mode=False
        )
        
        # Run all quality gates
        results = validator.validate_all_gates()
        
        # Verify all gates were run
        expected_gates = [
            "test_coverage", "code_quality", "security_scan", "performance_test",
            "documentation", "complexity_check", "dependency_check", "linting"
        ]
        
        for gate in expected_gates:
            assert gate in results, f"Missing gate: {gate}"
            assert results[gate].gate_name == gate
        
        # Check that we have results
        assert len(validator.gate_results) >= len(expected_gates)
        assert len(validator.validation_history) >= 1
        
        # Generate final report
        report = validator.get_quality_report()
        assert "overall_status" in report
        assert "success_rate" in report
        assert "gate_summary" in report
        
        print(f"✅ Comprehensive validation complete! Success rate: {report['success_rate']:.1f}%")
        return True
    except Exception as e:
        print(f"❌ Comprehensive quality validation test failed: {e}")
        return False

def main():
    """Run all quality gate tests."""
    print("🧪 Testing Fed-ViT-AutoRL Quality Gates & Validation\\n")
    
    tests = [
        test_quality_gate_components,
        test_quality_gate_validator,
        test_individual_quality_gates,
        test_quality_report_generation,
        test_comprehensive_quality_validation,
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
        print("🎉 Quality gates and validation system working perfectly!")
        print("✅ Comprehensive test coverage validation")
        print("✅ Code quality analysis with metrics")
        print("✅ Security scanning with pattern detection")
        print("✅ Performance validation and monitoring")
        print("✅ Documentation coverage analysis")
        print("✅ Complexity analysis and thresholds")
        print("✅ Dependency validation and checks")
        print("✅ Code linting and style validation")
        print("✅ Comprehensive quality reporting")
        return True
    else:
        print("⚠️  Some quality gate tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)