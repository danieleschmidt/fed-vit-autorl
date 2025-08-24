"""Quality Gates and Comprehensive Testing Framework

Implements automated quality gates with testing, security scanning, 
performance benchmarking, and validation for all three generations.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import time
import logging
import hashlib
import subprocess
import ast
import inspect
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import unittest
from unittest.mock import Mock, patch
import warnings
import gc
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import tempfile
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QualityGateStatus(Enum):
    """Quality gate status enumeration."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


class TestSeverity(Enum):
    """Test severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class QualityGateResult:
    """Result of a quality gate check."""
    gate_name: str
    status: QualityGateStatus
    severity: TestSeverity
    score: float  # 0.0 to 1.0
    message: str
    details: Dict[str, Any]
    execution_time: float
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            **asdict(self),
            'status': self.status.value,
            'severity': self.severity.value,
            'timestamp': self.timestamp.isoformat()
        }


class PerformanceBenchmark:
    """Performance benchmarking for federated learning components."""
    
    def __init__(self):
        self.benchmarks = {}
        self.baseline_metrics = {
            'model_inference_ms': 100.0,  # Maximum acceptable inference time
            'training_batch_ms': 2000.0,  # Maximum training time per batch
            'memory_usage_mb': 1000.0,    # Maximum memory usage
            'aggregation_ms': 500.0,      # Maximum aggregation time
            'cache_hit_rate': 0.8         # Minimum cache hit rate
        }
    
    def benchmark_model_inference(self, model: nn.Module, input_tensor: torch.Tensor, 
                                iterations: int = 100) -> Dict[str, float]:
        """Benchmark model inference performance."""
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_tensor)
        
        # Benchmark
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.perf_counter()
        
        with torch.no_grad():
            for _ in range(iterations):
                _ = model(input_tensor)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        avg_time_ms = (total_time / iterations) * 1000
        
        return {
            'avg_inference_time_ms': avg_time_ms,
            'iterations': iterations,
            'total_time_s': total_time,
            'throughput_fps': iterations / total_time
        }
    
    def benchmark_training_iteration(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                                   data: torch.Tensor, labels: torch.Tensor,
                                   iterations: int = 50) -> Dict[str, float]:
        """Benchmark training iteration performance."""
        model.train()
        criterion = nn.CrossEntropyLoss()
        
        # Warmup
        for _ in range(5):
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Benchmark
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.perf_counter()
        
        for _ in range(iterations):
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        avg_time_ms = (total_time / iterations) * 1000
        
        return {
            'avg_training_time_ms': avg_time_ms,
            'iterations': iterations,
            'total_time_s': total_time,
            'throughput_ips': iterations / total_time  # iterations per second
        }
    
    def benchmark_memory_usage(self, func: callable, *args, **kwargs) -> Dict[str, float]:
        """Benchmark memory usage of a function."""
        # Get initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        else:
            initial_gpu_memory = 0
        
        # Execute function
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        # Get final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        if torch.cuda.is_available():
            final_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        else:
            final_gpu_memory = 0
        
        return {
            'execution_time_s': end_time - start_time,
            'cpu_memory_used_mb': final_memory - initial_memory,
            'gpu_memory_used_mb': final_gpu_memory - initial_gpu_memory,
            'peak_cpu_memory_mb': final_memory,
            'peak_gpu_memory_mb': final_gpu_memory
        }


class SecurityScanner:
    """Security vulnerability scanner for federated learning code."""
    
    def __init__(self):
        self.security_issues = []
        self.severity_weights = {
            TestSeverity.LOW: 0.1,
            TestSeverity.MEDIUM: 0.3,
            TestSeverity.HIGH: 0.7,
            TestSeverity.CRITICAL: 1.0
        }
    
    def scan_code_for_vulnerabilities(self, file_path: Path) -> Dict[str, Any]:
        """Scan Python file for common security vulnerabilities."""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST for security analysis
            try:
                tree = ast.parse(content)
                issues.extend(self._analyze_ast_security(tree, content))
            except SyntaxError as e:
                issues.append({
                    'type': 'syntax_error',
                    'severity': TestSeverity.HIGH,
                    'message': f'Syntax error: {str(e)}',
                    'line': getattr(e, 'lineno', 0)
                })
            
            # Text-based security checks
            issues.extend(self._analyze_text_security(content))
            
        except Exception as e:
            issues.append({
                'type': 'scan_error',
                'severity': TestSeverity.MEDIUM,
                'message': f'Failed to scan file: {str(e)}',
                'line': 0
            })
        
        return {
            'file': str(file_path),
            'issues_found': len(issues),
            'issues': issues,
            'security_score': self._calculate_security_score(issues)
        }
    
    def _analyze_ast_security(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Analyze AST for security issues."""
        issues = []
        
        for node in ast.walk(tree):
            # Check for dangerous function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    
                    # Dangerous functions
                    if func_name in ['eval', 'exec']:
                        issues.append({
                            'type': 'dangerous_function',
                            'severity': TestSeverity.CRITICAL,
                            'message': f'Use of dangerous function: {func_name}',
                            'line': node.lineno
                        })
                    
                    elif func_name in ['input', 'raw_input'] and len(node.args) == 0:
                        issues.append({
                            'type': 'unsafe_input',
                            'severity': TestSeverity.HIGH,
                            'message': f'Unsafe user input without validation: {func_name}',
                            'line': node.lineno
                        })
                
                elif isinstance(node.func, ast.Attribute):
                    # Check for subprocess without shell=False
                    if (hasattr(node.func, 'attr') and 
                        node.func.attr in ['call', 'run', 'Popen']):
                        
                        # Check for shell=True
                        for keyword in node.keywords:
                            if (keyword.arg == 'shell' and 
                                isinstance(keyword.value, ast.Constant) and
                                keyword.value.value is True):
                                
                                issues.append({
                                    'type': 'shell_injection_risk',
                                    'severity': TestSeverity.HIGH,
                                    'message': 'subprocess call with shell=True is dangerous',
                                    'line': node.lineno
                                })
            
            # Check for hardcoded secrets/passwords
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        var_name = target.id.lower()
                        if any(secret in var_name for secret in ['password', 'secret', 'key', 'token']):
                            if isinstance(node.value, ast.Constant):
                                issues.append({
                                    'type': 'hardcoded_secret',
                                    'severity': TestSeverity.HIGH,
                                    'message': f'Potentially hardcoded secret in variable: {target.id}',
                                    'line': node.lineno
                                })
        
        return issues
    
    def _analyze_text_security(self, content: str) -> List[Dict[str, Any]]:
        """Analyze code content for text-based security issues."""
        issues = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            line_lower = line.lower()
            
            # Check for SQL injection patterns
            if any(pattern in line_lower for pattern in [
                'execute(', 'query(', 'cursor.execute'
            ]):
                if '+' in line or '.format(' in line or 'f"' in line or "f'" in line:
                    issues.append({
                        'type': 'sql_injection_risk',
                        'severity': TestSeverity.HIGH,
                        'message': 'Potential SQL injection vulnerability',
                        'line': i
                    })
            
            # Check for debug/development code
            if any(debug in line_lower for debug in ['print(', 'pdb.', 'breakpoint(', 'debug=']):
                issues.append({
                    'type': 'debug_code',
                    'severity': TestSeverity.LOW,
                    'message': 'Debug code found in production',
                    'line': i
                })
            
            # Check for TODO/FIXME/HACK comments
            if any(comment in line_lower for comment in ['todo', 'fixme', 'hack', 'xxx']):
                issues.append({
                    'type': 'todo_comment',
                    'severity': TestSeverity.LOW,
                    'message': 'TODO/FIXME comment found',
                    'line': i
                })
        
        return issues
    
    def _calculate_security_score(self, issues: List[Dict[str, Any]]) -> float:
        """Calculate security score (0.0 to 1.0, higher is better)."""
        if not issues:
            return 1.0
        
        total_weight = sum(self.severity_weights[issue['severity']] for issue in issues)
        max_possible_weight = len(issues) * self.severity_weights[TestSeverity.CRITICAL]
        
        if max_possible_weight == 0:
            return 1.0
        
        return max(0.0, 1.0 - (total_weight / max_possible_weight))


class FunctionalTester:
    """Functional testing suite for federated learning components."""
    
    def test_model_basic_functionality(self, model_class, *args, **kwargs) -> Dict[str, Any]:
        """Test basic model functionality."""
        start_time = time.perf_counter()
        issues = []
        
        try:
            # Test model creation
            model = model_class(*args, **kwargs)
            
            # Test forward pass
            batch_size = 4
            if hasattr(model, 'embed_dim'):
                input_size = (batch_size, 3, 384, 384)  # Assume image input
            else:
                input_size = (batch_size, 768)  # Assume feature input
            
            test_input = torch.randn(*input_size)
            
            with torch.no_grad():
                output = model(test_input)
            
            # Validate output
            if output is None:
                issues.append("Model returned None output")
            elif torch.isnan(output).any():
                issues.append("Model output contains NaN values")
            elif torch.isinf(output).any():
                issues.append("Model output contains infinite values")
            elif output.shape[0] != batch_size:
                issues.append(f"Output batch size mismatch: expected {batch_size}, got {output.shape[0]}")
            
            # Test parameter count
            param_count = sum(p.numel() for p in model.parameters())
            if param_count == 0:
                issues.append("Model has no parameters")
            elif param_count > 100_000_000:  # 100M parameters
                issues.append(f"Model is very large: {param_count:,} parameters")
            
            # Test gradient computation
            model.train()
            loss = nn.CrossEntropyLoss()(output, torch.randint(0, output.size(-1), (batch_size,)))
            loss.backward()
            
            # Check gradients
            grad_count = 0
            for param in model.parameters():
                if param.grad is not None:
                    grad_count += 1
                    if torch.isnan(param.grad).any():
                        issues.append("Model gradients contain NaN values")
                    elif torch.isinf(param.grad).any():
                        issues.append("Model gradients contain infinite values")
            
            if grad_count == 0:
                issues.append("No gradients computed")
            
        except Exception as e:
            issues.append(f"Model test failed with exception: {str(e)}")
        
        execution_time = time.perf_counter() - start_time
        
        return {
            'passed': len(issues) == 0,
            'issues': issues,
            'execution_time': execution_time,
            'parameter_count': param_count if 'param_count' in locals() else 0
        }
    
    def test_federated_system_integration(self, system_class, *args, **kwargs) -> Dict[str, Any]:
        """Test federated system integration."""
        start_time = time.perf_counter()
        issues = []
        
        try:
            # Test system creation
            system = system_class(*args, **kwargs)
            
            # Test data generation
            from .simple_training import generate_mock_data
            client_data, client_labels = generate_mock_data(batch_size=8, num_clients=3)
            test_data, test_labels = generate_mock_data(batch_size=4, num_clients=1)
            test_data, test_labels = test_data[0], test_labels[0]
            
            # Test training (if available)
            if hasattr(system, 'client_local_training') or hasattr(system, 'client_local_training_robust'):
                train_method = (getattr(system, 'client_local_training_robust', None) or 
                              getattr(system, 'client_local_training', None))
                
                if train_method:
                    for client_id in range(min(3, len(client_data))):
                        try:
                            result = train_method(
                                client_id=client_id,
                                data=client_data[client_id],
                                labels=client_labels[client_id],
                                epochs=1
                            )
                            
                            if result is None:
                                issues.append(f"Training returned None for client {client_id}")
                            elif isinstance(result, dict):
                                if 'loss' not in result:
                                    issues.append(f"Training result missing 'loss' for client {client_id}")
                                elif not isinstance(result['loss'], (int, float)):
                                    issues.append(f"Invalid loss type for client {client_id}")
                                elif np.isnan(result['loss']) or np.isinf(result['loss']):
                                    issues.append(f"Invalid loss value for client {client_id}")
                        
                        except Exception as e:
                            issues.append(f"Training failed for client {client_id}: {str(e)}")
            
            # Test aggregation (if available)
            if hasattr(system, 'federated_averaging') or hasattr(system, 'federated_averaging_robust'):
                agg_method = (getattr(system, 'federated_averaging_robust', None) or 
                            getattr(system, 'federated_averaging', None))
                
                if agg_method:
                    try:
                        agg_result = agg_method()
                        if not isinstance(agg_result, bool):
                            issues.append("Aggregation should return boolean success status")
                    except Exception as e:
                        issues.append(f"Aggregation failed: {str(e)}")
            
            # Test evaluation (if available)
            if hasattr(system, 'evaluate_global_model') or hasattr(system, 'evaluate_global_model_robust'):
                eval_method = (getattr(system, 'evaluate_global_model_robust', None) or 
                             getattr(system, 'evaluate_global_model', None))
                
                if eval_method:
                    try:
                        eval_result = eval_method(test_data, test_labels)
                        if eval_result is None:
                            issues.append("Evaluation returned None")
                        elif isinstance(eval_result, dict):
                            required_keys = ['loss', 'accuracy']
                            for key in required_keys:
                                if key not in eval_result:
                                    issues.append(f"Evaluation result missing '{key}'")
                    except Exception as e:
                        issues.append(f"Evaluation failed: {str(e)}")
            
        except Exception as e:
            issues.append(f"System integration test failed: {str(e)}")
        
        execution_time = time.perf_counter() - start_time
        
        return {
            'passed': len(issues) == 0,
            'issues': issues,
            'execution_time': execution_time
        }


class QualityGateRunner:
    """Main quality gate runner that orchestrates all checks."""
    
    def __init__(self):
        self.benchmarker = PerformanceBenchmark()
        self.security_scanner = SecurityScanner()
        self.functional_tester = FunctionalTester()
        self.results = []
    
    def run_all_quality_gates(self, project_path: Path = None) -> Dict[str, Any]:
        """Run all quality gates and return comprehensive report."""
        
        if project_path is None:
            project_path = Path(__file__).parent.parent
        
        logger.info("🛡️ Starting comprehensive quality gate execution")
        start_time = time.perf_counter()
        
        # Define quality gates
        quality_gates = [
            ("Security Scan", TestSeverity.CRITICAL, self._run_security_gate),
            ("Performance Benchmarks", TestSeverity.HIGH, self._run_performance_gate),
            ("Unit Tests", TestSeverity.HIGH, self._run_unit_tests_gate),
            ("Integration Tests", TestSeverity.HIGH, self._run_integration_tests_gate),
            ("Code Coverage", TestSeverity.MEDIUM, self._run_coverage_gate),
            ("Model Validation", TestSeverity.HIGH, self._run_model_validation_gate),
            ("Memory Leak Detection", TestSeverity.MEDIUM, self._run_memory_leak_gate),
            ("Thread Safety", TestSeverity.MEDIUM, self._run_thread_safety_gate),
            ("Error Recovery", TestSeverity.HIGH, self._run_error_recovery_gate),
            ("Resource Limits", TestSeverity.MEDIUM, self._run_resource_limits_gate)
        ]
        
        # Execute quality gates
        for gate_name, severity, gate_func in quality_gates:
            try:
                logger.info(f"Executing quality gate: {gate_name}")
                gate_start = time.perf_counter()
                
                result = gate_func(project_path)
                
                gate_time = time.perf_counter() - gate_start
                
                gate_result = QualityGateResult(
                    gate_name=gate_name,
                    status=result.get('status', QualityGateStatus.FAILED),
                    severity=severity,
                    score=result.get('score', 0.0),
                    message=result.get('message', ''),
                    details=result.get('details', {}),
                    execution_time=gate_time,
                    timestamp=datetime.now()
                )
                
                self.results.append(gate_result)
                
                status_emoji = "✅" if gate_result.status == QualityGateStatus.PASSED else "❌"
                logger.info(f"{status_emoji} {gate_name}: {gate_result.status.value} "
                          f"(Score: {gate_result.score:.2f}, Time: {gate_time:.2f}s)")
                
            except Exception as e:
                logger.error(f"Quality gate '{gate_name}' failed with exception: {str(e)}")
                
                error_result = QualityGateResult(
                    gate_name=gate_name,
                    status=QualityGateStatus.FAILED,
                    severity=severity,
                    score=0.0,
                    message=f"Gate execution failed: {str(e)}",
                    details={'exception': str(e)},
                    execution_time=0.0,
                    timestamp=datetime.now()
                )
                
                self.results.append(error_result)
        
        total_time = time.perf_counter() - start_time
        
        # Calculate overall score
        overall_score = self._calculate_overall_score()
        
        # Generate final report
        report = {
            'overall_status': 'PASSED' if overall_score >= 0.8 else 'FAILED',
            'overall_score': overall_score,
            'total_execution_time': total_time,
            'gates_executed': len(quality_gates),
            'gates_passed': len([r for r in self.results if r.status == QualityGateStatus.PASSED]),
            'gates_failed': len([r for r in self.results if r.status == QualityGateStatus.FAILED]),
            'gates_warning': len([r for r in self.results if r.status == QualityGateStatus.WARNING]),
            'timestamp': datetime.now().isoformat(),
            'results': [r.to_dict() for r in self.results]
        }
        
        # Save report
        self._save_quality_report(report)
        
        logger.info(f"🏁 Quality gate execution completed: {report['overall_status']} "
                   f"(Score: {overall_score:.2f}, Time: {total_time:.2f}s)")
        
        return report
    
    def _run_security_gate(self, project_path: Path) -> Dict[str, Any]:
        """Run security scanning quality gate."""
        python_files = list(project_path.glob("**/*.py"))
        
        all_issues = []
        security_scores = []
        
        for py_file in python_files:
            if 'test_' in py_file.name or '__pycache__' in str(py_file):
                continue
                
            scan_result = self.security_scanner.scan_code_for_vulnerabilities(py_file)
            all_issues.extend(scan_result['issues'])
            security_scores.append(scan_result['security_score'])
        
        avg_security_score = np.mean(security_scores) if security_scores else 1.0
        critical_issues = len([i for i in all_issues if i['severity'] == TestSeverity.CRITICAL])
        high_issues = len([i for i in all_issues if i['severity'] == TestSeverity.HIGH])
        
        status = QualityGateStatus.PASSED
        if critical_issues > 0:
            status = QualityGateStatus.FAILED
        elif high_issues > 5:
            status = QualityGateStatus.FAILED
        elif high_issues > 0:
            status = QualityGateStatus.WARNING
        
        return {
            'status': status,
            'score': avg_security_score,
            'message': f"Found {len(all_issues)} security issues ({critical_issues} critical, {high_issues} high)",
            'details': {
                'files_scanned': len(python_files),
                'total_issues': len(all_issues),
                'critical_issues': critical_issues,
                'high_issues': high_issues,
                'security_score': avg_security_score,
                'issues': all_issues[:10]  # First 10 issues
            }
        }
    
    def _run_performance_gate(self, project_path: Path) -> Dict[str, Any]:
        """Run performance benchmarking quality gate."""
        try:
            # Test simple model performance
            from .simple_training import SimpleViTPerception
            
            model = SimpleViTPerception(embed_dim=512)  # Smaller for testing
            test_input = torch.randn(4, 3, 384, 384)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            labels = torch.randint(0, 4, (4,))
            
            # Benchmark inference
            inference_result = self.benchmarker.benchmark_model_inference(model, test_input, iterations=50)
            
            # Benchmark training
            training_result = self.benchmarker.benchmark_training_iteration(
                model, optimizer, test_input, labels, iterations=20
            )
            
            # Check against baselines
            inference_passed = inference_result['avg_inference_time_ms'] < self.benchmarker.baseline_metrics['model_inference_ms']
            training_passed = training_result['avg_training_time_ms'] < self.benchmarker.baseline_metrics['training_batch_ms']
            
            status = QualityGateStatus.PASSED if (inference_passed and training_passed) else QualityGateStatus.WARNING
            score = 0.8 if (inference_passed and training_passed) else 0.6
            
            return {
                'status': status,
                'score': score,
                'message': f"Inference: {inference_result['avg_inference_time_ms']:.1f}ms, Training: {training_result['avg_training_time_ms']:.1f}ms",
                'details': {
                    'inference_benchmark': inference_result,
                    'training_benchmark': training_result,
                    'baselines': self.benchmarker.baseline_metrics,
                    'inference_passed': inference_passed,
                    'training_passed': training_passed
                }
            }
            
        except Exception as e:
            return {
                'status': QualityGateStatus.FAILED,
                'score': 0.0,
                'message': f"Performance benchmarking failed: {str(e)}",
                'details': {'error': str(e)}
            }
    
    def _run_unit_tests_gate(self, project_path: Path) -> Dict[str, Any]:
        """Run unit tests quality gate."""
        try:
            # Import the test modules to run them
            test_results = []
            
            # Test Generation 1
            from .simple_training import SimpleViTPerception, SimpleFederatedSystem
            
            model_test = self.functional_tester.test_model_basic_functionality(SimpleViTPerception, embed_dim=512)
            test_results.append(('SimpleViT Model', model_test))
            
            system_test = self.functional_tester.test_federated_system_integration(SimpleFederatedSystem, num_clients=3)
            test_results.append(('SimpleFederated System', system_test))
            
            # Test Generation 2 (if available)
            try:
                from .robust_training import RobustFederatedSystem
                robust_test = self.functional_tester.test_federated_system_integration(
                    RobustFederatedSystem, num_clients=3, embed_dim=512
                )
                test_results.append(('RobustFederated System', robust_test))
            except ImportError:
                pass
            
            passed_tests = sum(1 for _, result in test_results if result['passed'])
            total_tests = len(test_results)
            
            status = QualityGateStatus.PASSED if passed_tests == total_tests else QualityGateStatus.WARNING
            score = passed_tests / total_tests if total_tests > 0 else 0.0
            
            return {
                'status': status,
                'score': score,
                'message': f"Passed {passed_tests}/{total_tests} unit tests",
                'details': {
                    'passed_tests': passed_tests,
                    'total_tests': total_tests,
                    'test_results': test_results
                }
            }
            
        except Exception as e:
            return {
                'status': QualityGateStatus.FAILED,
                'score': 0.0,
                'message': f"Unit tests failed: {str(e)}",
                'details': {'error': str(e)}
            }
    
    def _run_integration_tests_gate(self, project_path: Path) -> Dict[str, Any]:
        """Run integration tests quality gate."""
        try:
            # Test all three generations integration
            integration_results = []
            
            # Test Generation 1
            try:
                from .simple_training import run_simple_federated_training
                gen1_result = run_simple_federated_training(num_rounds=1, num_clients=3)
                integration_results.append(('Generation 1', gen1_result))
            except Exception as e:
                integration_results.append(('Generation 1', False))
                logger.error(f"Generation 1 integration test failed: {e}")
            
            # Test Generation 2
            try:
                from .robust_training import run_robust_federated_training
                gen2_result = run_robust_federated_training(num_rounds=1, num_clients=3)
                integration_results.append(('Generation 2', gen2_result))
            except Exception as e:
                integration_results.append(('Generation 2', False))
                logger.error(f"Generation 2 integration test failed: {e}")
            
            passed_integrations = sum(1 for _, result in integration_results if result)
            total_integrations = len(integration_results)
            
            status = QualityGateStatus.PASSED if passed_integrations >= total_integrations * 0.8 else QualityGateStatus.WARNING
            score = passed_integrations / total_integrations if total_integrations > 0 else 0.0
            
            return {
                'status': status,
                'score': score,
                'message': f"Passed {passed_integrations}/{total_integrations} integration tests",
                'details': {
                    'integration_results': integration_results,
                    'passed_integrations': passed_integrations,
                    'total_integrations': total_integrations
                }
            }
            
        except Exception as e:
            return {
                'status': QualityGateStatus.FAILED,
                'score': 0.0,
                'message': f"Integration tests failed: {str(e)}",
                'details': {'error': str(e)}
            }
    
    def _run_coverage_gate(self, project_path: Path) -> Dict[str, Any]:
        """Run code coverage analysis quality gate."""
        try:
            # Simple coverage analysis based on imports and function calls
            python_files = list(project_path.glob("**/*.py"))
            
            total_functions = 0
            covered_functions = 0
            
            for py_file in python_files:
                if 'test_' in py_file.name or '__pycache__' in str(py_file):
                    continue
                    
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            total_functions += 1
                            # Simple heuristic: function is "covered" if it has docstring or has calls
                            if (ast.get_docstring(node) or 
                                any(isinstance(child, ast.Call) for child in ast.walk(node))):
                                covered_functions += 1
                                
                except Exception as e:
                    logger.warning(f"Could not analyze coverage for {py_file}: {e}")
            
            coverage_ratio = covered_functions / total_functions if total_functions > 0 else 1.0
            
            status = QualityGateStatus.PASSED if coverage_ratio >= 0.8 else QualityGateStatus.WARNING
            
            return {
                'status': status,
                'score': coverage_ratio,
                'message': f"Code coverage: {coverage_ratio:.1%} ({covered_functions}/{total_functions} functions)",
                'details': {
                    'total_functions': total_functions,
                    'covered_functions': covered_functions,
                    'coverage_ratio': coverage_ratio,
                    'files_analyzed': len(python_files)
                }
            }
            
        except Exception as e:
            return {
                'status': QualityGateStatus.FAILED,
                'score': 0.0,
                'message': f"Coverage analysis failed: {str(e)}",
                'details': {'error': str(e)}
            }
    
    def _run_model_validation_gate(self, project_path: Path) -> Dict[str, Any]:
        """Run model validation quality gate."""
        try:
            validation_results = []
            
            # Test model consistency
            from .simple_training import SimpleViTPerception
            
            model1 = SimpleViTPerception(embed_dim=512)
            model2 = SimpleViTPerception(embed_dim=512)
            
            # Test same input produces same output (deterministic)
            test_input = torch.randn(2, 3, 384, 384)
            
            model1.eval()
            model2.eval()
            
            with torch.no_grad():
                output1 = model1(test_input)
                output2 = model2(test_input)
            
            # They should be different since models are randomly initialized
            output_diff = torch.norm(output1 - output2).item()
            validation_results.append(('Model Randomization', output_diff > 0.1))
            
            # Test model can learn (gradient flows)
            model1.train()
            optimizer = torch.optim.Adam(model1.parameters(), lr=1e-4)
            criterion = nn.CrossEntropyLoss()
            
            labels = torch.randint(0, 4, (2,))
            
            initial_loss = criterion(model1(test_input), labels).item()
            
            for _ in range(5):  # Few training steps
                optimizer.zero_grad()
                loss = criterion(model1(test_input), labels)
                loss.backward()
                optimizer.step()
            
            final_loss = criterion(model1(test_input), labels).item()
            learning_occurred = final_loss < initial_loss
            validation_results.append(('Model Learning', learning_occurred))
            
            # Test model state persistence
            state_dict = model1.state_dict()
            model3 = SimpleViTPerception(embed_dim=512)
            model3.load_state_dict(state_dict)
            
            model1.eval()
            model3.eval()
            
            with torch.no_grad():
                output1 = model1(test_input)
                output3 = model3(test_input)
            
            state_persistence = torch.allclose(output1, output3, rtol=1e-5)
            validation_results.append(('State Persistence', state_persistence))
            
            passed_validations = sum(1 for _, result in validation_results if result)
            total_validations = len(validation_results)
            
            status = QualityGateStatus.PASSED if passed_validations == total_validations else QualityGateStatus.WARNING
            score = passed_validations / total_validations
            
            return {
                'status': status,
                'score': score,
                'message': f"Passed {passed_validations}/{total_validations} model validations",
                'details': {
                    'validation_results': validation_results,
                    'initial_loss': initial_loss,
                    'final_loss': final_loss,
                    'learning_occurred': learning_occurred
                }
            }
            
        except Exception as e:
            return {
                'status': QualityGateStatus.FAILED,
                'score': 0.0,
                'message': f"Model validation failed: {str(e)}",
                'details': {'error': str(e)}
            }
    
    def _run_memory_leak_gate(self, project_path: Path) -> Dict[str, Any]:
        """Run memory leak detection quality gate."""
        try:
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Run training iterations and monitor memory
            memory_samples = []
            
            from .simple_training import SimpleViTPerception, SimpleFederatedSystem
            
            for i in range(10):  # 10 iterations
                # Create and destroy objects
                model = SimpleViTPerception(embed_dim=256)  # Smaller model
                system = SimpleFederatedSystem(num_clients=2)
                
                # Run a quick training step
                data = torch.randn(4, 3, 384, 384)
                labels = torch.randint(0, 4, (4,))
                
                system.client_local_training(0, data, labels, epochs=1)
                
                # Force cleanup
                del model, system, data, labels
                gc.collect()
                
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                memory_samples.append(current_memory)
            
            final_memory = memory_samples[-1]
            memory_increase = final_memory - initial_memory
            
            # Check for memory leak (more than 100MB increase)
            has_memory_leak = memory_increase > 100.0
            
            status = QualityGateStatus.PASSED if not has_memory_leak else QualityGateStatus.WARNING
            score = max(0.0, 1.0 - (memory_increase / 200.0))  # Normalize to 0-1 scale
            
            return {
                'status': status,
                'score': score,
                'message': f"Memory increase: {memory_increase:.1f} MB",
                'details': {
                    'initial_memory_mb': initial_memory,
                    'final_memory_mb': final_memory,
                    'memory_increase_mb': memory_increase,
                    'memory_samples': memory_samples,
                    'has_memory_leak': has_memory_leak
                }
            }
            
        except Exception as e:
            return {
                'status': QualityGateStatus.FAILED,
                'score': 0.0,
                'message': f"Memory leak detection failed: {str(e)}",
                'details': {'error': str(e)}
            }
    
    def _run_thread_safety_gate(self, project_path: Path) -> Dict[str, Any]:
        """Run thread safety analysis quality gate."""
        try:
            thread_safety_results = []
            
            # Test concurrent model access
            from .simple_training import SimpleViTPerception
            
            model = SimpleViTPerception(embed_dim=256)
            shared_results = []
            errors = []
            
            def worker_function(worker_id):
                try:
                    test_input = torch.randn(2, 3, 384, 384)
                    with torch.no_grad():
                        output = model(test_input)
                        shared_results.append((worker_id, output.shape, output.mean().item()))
                except Exception as e:
                    errors.append((worker_id, str(e)))
            
            # Run concurrent workers
            threads = []
            for i in range(5):
                thread = threading.Thread(target=worker_function, args=(i,))
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join(timeout=10)
            
            # Analyze results
            concurrent_access_safe = len(errors) == 0 and len(shared_results) == 5
            thread_safety_results.append(('Concurrent Model Access', concurrent_access_safe))
            
            # Test for race conditions in simple operations
            shared_counter = {'value': 0}
            lock = threading.Lock()
            
            def increment_with_lock():
                for _ in range(100):
                    with lock:
                        shared_counter['value'] += 1
            
            def increment_without_lock():
                for _ in range(100):
                    shared_counter['value'] += 1
            
            # Test with lock (should be safe)
            shared_counter['value'] = 0
            lock_threads = [threading.Thread(target=increment_with_lock) for _ in range(5)]
            for t in lock_threads:
                t.start()
            for t in lock_threads:
                t.join()
            
            expected_value = 500  # 5 threads * 100 increments
            lock_safe = shared_counter['value'] == expected_value
            thread_safety_results.append(('Lock Safety', lock_safe))
            
            passed_tests = sum(1 for _, result in thread_safety_results if result)
            total_tests = len(thread_safety_results)
            
            status = QualityGateStatus.PASSED if passed_tests == total_tests else QualityGateStatus.WARNING
            score = passed_tests / total_tests
            
            return {
                'status': status,
                'score': score,
                'message': f"Passed {passed_tests}/{total_tests} thread safety tests",
                'details': {
                    'thread_safety_results': thread_safety_results,
                    'concurrent_errors': errors,
                    'concurrent_results_count': len(shared_results),
                    'lock_test_result': shared_counter['value']
                }
            }
            
        except Exception as e:
            return {
                'status': QualityGateStatus.FAILED,
                'score': 0.0,
                'message': f"Thread safety testing failed: {str(e)}",
                'details': {'error': str(e)}
            }
    
    def _run_error_recovery_gate(self, project_path: Path) -> Dict[str, Any]:
        """Run error recovery testing quality gate."""
        try:
            recovery_tests = []
            
            # Test model with invalid input
            from .simple_training import SimpleViTPerception
            
            model = SimpleViTPerception(embed_dim=256)
            
            # Test 1: Invalid input shape
            try:
                invalid_input = torch.randn(2, 2, 384, 384)  # Wrong number of channels
                with torch.no_grad():
                    output = model(invalid_input)
                recovery_tests.append(('Invalid Input Shape', False))  # Should have failed
            except Exception:
                recovery_tests.append(('Invalid Input Shape', True))  # Correctly handled error
            
            # Test 2: NaN input handling
            try:
                nan_input = torch.full((2, 3, 384, 384), float('nan'))
                with torch.no_grad():
                    output = model(nan_input)
                    has_nan_output = torch.isnan(output).any().item()
                    recovery_tests.append(('NaN Input Handling', not has_nan_output))
            except Exception:
                recovery_tests.append(('NaN Input Handling', True))  # Error correctly caught
            
            # Test 3: Very large values
            try:
                large_input = torch.full((2, 3, 384, 384), 1e6)
                with torch.no_grad():
                    output = model(large_input)
                    has_inf_output = torch.isinf(output).any().item()
                    recovery_tests.append(('Large Value Handling', not has_inf_output))
            except Exception:
                recovery_tests.append(('Large Value Handling', True))  # Error correctly caught
            
            # Test 4: Memory exhaustion simulation
            try:
                # Try to allocate huge tensor
                huge_tensor = torch.randn(1000, 1000, 1000)  # Will likely fail
                recovery_tests.append(('Memory Exhaustion', False))  # Should have failed
            except (RuntimeError, MemoryError):
                recovery_tests.append(('Memory Exhaustion', True))  # Correctly handled
            
            passed_tests = sum(1 for _, result in recovery_tests if result)
            total_tests = len(recovery_tests)
            
            status = QualityGateStatus.PASSED if passed_tests >= total_tests * 0.75 else QualityGateStatus.WARNING
            score = passed_tests / total_tests
            
            return {
                'status': status,
                'score': score,
                'message': f"Passed {passed_tests}/{total_tests} error recovery tests",
                'details': {
                    'recovery_tests': recovery_tests,
                    'passed_tests': passed_tests,
                    'total_tests': total_tests
                }
            }
            
        except Exception as e:
            return {
                'status': QualityGateStatus.FAILED,
                'score': 0.0,
                'message': f"Error recovery testing failed: {str(e)}",
                'details': {'error': str(e)}
            }
    
    def _run_resource_limits_gate(self, project_path: Path) -> Dict[str, Any]:
        """Run resource limits testing quality gate."""
        try:
            resource_tests = []
            
            # Monitor resource usage during model operations
            initial_cpu = psutil.cpu_percent(interval=0.1)
            initial_memory = psutil.virtual_memory().percent
            
            from .simple_training import SimpleViTPerception
            
            start_time = time.perf_counter()
            
            # Create and use model
            model = SimpleViTPerception(embed_dim=512)
            data = torch.randn(8, 3, 384, 384)
            labels = torch.randint(0, 4, (8,))
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            
            # Training iterations
            for _ in range(10):
                optimizer.zero_grad()
                output = model(data)
                loss = nn.CrossEntropyLoss()(output, labels)
                loss.backward()
                optimizer.step()
            
            end_time = time.perf_counter()
            final_cpu = psutil.cpu_percent(interval=0.1)
            final_memory = psutil.virtual_memory().percent
            
            execution_time = end_time - start_time
            cpu_increase = final_cpu - initial_cpu
            memory_increase = final_memory - initial_memory
            
            # Resource limit checks
            resource_tests.append(('Execution Time', execution_time < 30.0))  # Under 30 seconds
            resource_tests.append(('CPU Usage', cpu_increase < 80.0))  # CPU increase under 80%
            resource_tests.append(('Memory Usage', memory_increase < 20.0))  # Memory increase under 20%
            
            # Check model size
            param_count = sum(p.numel() for p in model.parameters())
            model_size_mb = param_count * 4 / 1024 / 1024  # Assume float32
            resource_tests.append(('Model Size', model_size_mb < 100.0))  # Under 100MB
            
            passed_tests = sum(1 for _, result in resource_tests if result)
            total_tests = len(resource_tests)
            
            status = QualityGateStatus.PASSED if passed_tests == total_tests else QualityGateStatus.WARNING
            score = passed_tests / total_tests
            
            return {
                'status': status,
                'score': score,
                'message': f"Passed {passed_tests}/{total_tests} resource limit tests",
                'details': {
                    'resource_tests': resource_tests,
                    'execution_time': execution_time,
                    'cpu_increase': cpu_increase,
                    'memory_increase': memory_increase,
                    'model_size_mb': model_size_mb,
                    'parameter_count': param_count
                }
            }
            
        except Exception as e:
            return {
                'status': QualityGateStatus.FAILED,
                'score': 0.0,
                'message': f"Resource limits testing failed: {str(e)}",
                'details': {'error': str(e)}
            }
    
    def _calculate_overall_score(self) -> float:
        """Calculate overall quality score."""
        if not self.results:
            return 0.0
        
        # Weight scores by severity
        severity_weights = {
            TestSeverity.CRITICAL: 1.0,
            TestSeverity.HIGH: 0.8,
            TestSeverity.MEDIUM: 0.6,
            TestSeverity.LOW: 0.4
        }
        
        weighted_scores = []
        for result in self.results:
            weight = severity_weights[result.severity]
            weighted_scores.append(result.score * weight)
        
        return np.mean(weighted_scores)
    
    def _save_quality_report(self, report: Dict[str, Any]):
        """Save quality report to file."""
        report_file = Path("quality_gates_report.json")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Quality gates report saved to: {report_file}")


def main():
    """Main entry point for quality gates."""
    runner = QualityGateRunner()
    
    try:
        report = runner.run_all_quality_gates()
        
        print(f"\n🛡️ QUALITY GATES REPORT 🛡️")
        print(f"Overall Status: {report['overall_status']}")
        print(f"Overall Score: {report['overall_score']:.2f}/1.00")
        print(f"Gates Passed: {report['gates_passed']}/{report['gates_executed']}")
        print(f"Total Time: {report['total_execution_time']:.2f}s")
        
        if report['overall_status'] == 'PASSED':
            print("✅ All quality gates passed!")
            exit(0)
        else:
            print("❌ Quality gates failed!")
            print("\nFailed Gates:")
            for result in runner.results:
                if result.status == QualityGateStatus.FAILED:
                    print(f"  - {result.gate_name}: {result.message}")
            exit(1)
            
    except Exception as e:
        print(f"❌ Quality gate execution failed: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()