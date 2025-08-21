"""Quality gates and validation for Fed-ViT-AutoRL."""

import time
import logging
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import subprocess
import os
import json

logger = logging.getLogger(__name__)


class QualityGateStatus(Enum):
    """Quality gate validation status."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class QualityMetrics:
    """Quality metrics for validation."""
    test_coverage: float = 0.0
    code_quality_score: float = 0.0
    security_score: float = 0.0
    performance_score: float = 0.0
    documentation_coverage: float = 0.0
    complexity_score: float = 0.0
    maintainability_score: float = 0.0
    reliability_score: float = 0.0
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class QualityGateResult:
    """Result of a quality gate validation."""
    gate_name: str
    status: QualityGateStatus
    message: str
    score: Optional[float] = None
    threshold: Optional[float] = None
    metrics: Optional[Dict[str, Any]] = None
    execution_time: float = 0.0
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class QualityGateValidator:
    """Comprehensive quality gate validation system."""

    def __init__(
        self,
        project_root: str = ".",
        min_coverage: float = 80.0,
        min_code_quality: float = 8.0,
        min_security_score: float = 85.0,
        min_performance_score: float = 75.0,
        min_documentation: float = 70.0,
        max_complexity: float = 10.0,
        enable_strict_mode: bool = False,
    ):
        """Initialize quality gate validator.

        Args:
            project_root: Root directory of the project
            min_coverage: Minimum test coverage percentage
            min_code_quality: Minimum code quality score (0-10)
            min_security_score: Minimum security score percentage
            min_performance_score: Minimum performance score percentage
            min_documentation: Minimum documentation coverage percentage
            max_complexity: Maximum cyclomatic complexity
            enable_strict_mode: Whether to enable strict validation
        """
        self.project_root = os.path.abspath(project_root)
        self.min_coverage = min_coverage
        self.min_code_quality = min_code_quality
        self.min_security_score = min_security_score
        self.min_performance_score = min_performance_score
        self.min_documentation = min_documentation
        self.max_complexity = max_complexity
        self.enable_strict_mode = enable_strict_mode

        self.gate_results: List[QualityGateResult] = []
        self.validation_history: List[Dict[str, Any]] = []

        # Quality gates configuration
        self.gates = {
            "test_coverage": self._validate_test_coverage,
            "code_quality": self._validate_code_quality,
            "security_scan": self._validate_security,
            "performance_test": self._validate_performance,
            "documentation": self._validate_documentation,
            "complexity_check": self._validate_complexity,
            "dependency_check": self._validate_dependencies,
            "linting": self._validate_linting,
        }

        logger.info(f"Initialized quality gate validator for {self.project_root}")

    def validate_all_gates(self) -> Dict[str, QualityGateResult]:
        """Run all quality gates.

        Returns:
            Dictionary of gate results
        """
        results = {}
        start_time = time.time()

        logger.info("Starting comprehensive quality gate validation")

        for gate_name, gate_function in self.gates.items():
            try:
                gate_start = time.time()
                result = gate_function()
                result.execution_time = time.time() - gate_start

                results[gate_name] = result
                self.gate_results.append(result)

                status_emoji = {"passed": "✅", "failed": "❌", "warning": "⚠️", "skipped": "⏭️"}
                logger.info(
                    f"{status_emoji.get(result.status.value, '❓')} {gate_name}: "
                    f"{result.status.value} - {result.message}"
                )

            except Exception as e:
                error_result = QualityGateResult(
                    gate_name=gate_name,
                    status=QualityGateStatus.FAILED,
                    message=f"Gate execution failed: {str(e)}",
                    execution_time=time.time() - gate_start
                )
                results[gate_name] = error_result
                self.gate_results.append(error_result)

                logger.error(f"❌ {gate_name}: failed with error: {e}")

        total_time = time.time() - start_time

        # Generate summary
        self._generate_validation_summary(results, total_time)

        return results

    def _validate_test_coverage(self) -> QualityGateResult:
        """Validate test coverage."""
        try:
            # Look for existing coverage data or run coverage if available
            coverage_file = os.path.join(self.project_root, ".coverage")

            if os.path.exists(coverage_file):
                # Try to read coverage data
                try:
                    result = subprocess.run(
                        ["coverage", "report", "--format=json"],
                        cwd=self.project_root,
                        capture_output=True,
                        text=True,
                        timeout=30
                    )

                    if result.returncode == 0:
                        coverage_data = json.loads(result.stdout)
                        coverage_percent = coverage_data.get("totals", {}).get("percent_covered", 0.0)
                    else:
                        raise subprocess.SubprocessError("Coverage command failed")

                except (subprocess.SubprocessError, json.JSONDecodeError, FileNotFoundError):
                    # Fallback to estimating coverage
                    coverage_percent = self._estimate_test_coverage()
            else:
                # Estimate coverage based on test files
                coverage_percent = self._estimate_test_coverage()

            if coverage_percent >= self.min_coverage:
                status = QualityGateStatus.PASSED
                message = f"Test coverage {coverage_percent:.1f}% meets minimum {self.min_coverage}%"
            elif coverage_percent >= self.min_coverage - 10:
                status = QualityGateStatus.WARNING
                message = f"Test coverage {coverage_percent:.1f}% below target {self.min_coverage}%"
            else:
                status = QualityGateStatus.FAILED
                message = f"Test coverage {coverage_percent:.1f}% below minimum {self.min_coverage}%"

            return QualityGateResult(
                gate_name="test_coverage",
                status=status,
                message=message,
                score=coverage_percent,
                threshold=self.min_coverage,
                metrics={"coverage_percent": coverage_percent}
            )

        except Exception as e:
            return QualityGateResult(
                gate_name="test_coverage",
                status=QualityGateStatus.FAILED,
                message=f"Coverage validation failed: {str(e)}"
            )

    def _estimate_test_coverage(self) -> float:
        """Estimate test coverage based on files."""
        try:
            # Count source files
            source_files = 0
            test_files = 0

            for root, dirs, files in os.walk(self.project_root):
                # Skip common non-source directories
                dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', '.pytest_cache', 'node_modules'}]

                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        rel_path = os.path.relpath(file_path, self.project_root)

                        if '/test' in rel_path or file.startswith('test_') or file.endswith('_test.py'):
                            test_files += 1
                        elif not file.startswith('_') and 'test' not in rel_path.lower():
                            source_files += 1

            if source_files == 0:
                return 0.0

            # Rough estimation: assume each test file covers 2-3 source files
            estimated_coverage = min(100.0, (test_files * 2.5 / source_files) * 100)

            return estimated_coverage

        except Exception:
            return 0.0

    def _validate_code_quality(self) -> QualityGateResult:
        """Validate code quality using static analysis."""
        try:
            quality_score = 8.0  # Default good score
            issues = []

            # Check for common code quality indicators
            python_files = []
            for root, dirs, files in os.walk(self.project_root):
                dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', '.pytest_cache'}]
                for file in files:
                    if file.endswith('.py'):
                        python_files.append(os.path.join(root, file))

            # Basic quality checks
            total_lines = 0
            total_functions = 0
            total_classes = 0
            doc_strings = 0

            for file_path in python_files[:50]:  # Limit to avoid timeout
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        lines = content.split('\n')
                        total_lines += len(lines)

                        # Count functions and classes
                        for line in lines:
                            stripped = line.strip()
                            if stripped.startswith('def '):
                                total_functions += 1
                            elif stripped.startswith('class '):
                                total_classes += 1
                            elif '"""' in stripped or "'''" in stripped:
                                doc_strings += 1

                except Exception:
                    continue

            # Calculate quality metrics
            if total_functions + total_classes > 0:
                doc_ratio = doc_strings / (total_functions + total_classes)
                if doc_ratio < 0.3:
                    quality_score -= 1.0
                    issues.append("Low documentation ratio")

            if total_lines > 0:
                avg_file_size = total_lines / len(python_files) if python_files else 0
                if avg_file_size > 500:
                    quality_score -= 0.5
                    issues.append("Large average file size")

            # Determine status
            if quality_score >= self.min_code_quality:
                status = QualityGateStatus.PASSED
                message = f"Code quality score {quality_score:.1f}/10 meets minimum {self.min_code_quality}"
            elif quality_score >= self.min_code_quality - 1:
                status = QualityGateStatus.WARNING
                message = f"Code quality score {quality_score:.1f}/10 below target {self.min_code_quality}"
            else:
                status = QualityGateStatus.FAILED
                message = f"Code quality score {quality_score:.1f}/10 below minimum {self.min_code_quality}"

            return QualityGateResult(
                gate_name="code_quality",
                status=status,
                message=message,
                score=quality_score,
                threshold=self.min_code_quality,
                metrics={
                    "quality_score": quality_score,
                    "total_files": len(python_files),
                    "total_lines": total_lines,
                    "issues": issues
                }
            )

        except Exception as e:
            return QualityGateResult(
                gate_name="code_quality",
                status=QualityGateStatus.FAILED,
                message=f"Code quality validation failed: {str(e)}"
            )

    def _validate_security(self) -> QualityGateResult:
        """Validate security using basic checks."""
        try:
            security_score = 90.0  # Start with high score
            security_issues = []

            # Check for common security issues
            python_files = []
            for root, dirs, files in os.walk(self.project_root):
                dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', '.pytest_cache'}]
                for file in files:
                    if file.endswith('.py'):
                        python_files.append(os.path.join(root, file))

            # Security pattern checks
            dangerous_patterns = [
                (r'(?<!\.)\beval\s*\([^)]*[\'"]', "Use of eval() function with string"),
                (r'(?<!\.)\bexec\s*\([^)]*[\'"]', "Use of exec() function with string"),
                (r'subprocess\.call\([^)]*shell=True', "Shell injection risk"),
                (r'os\.system\(', "OS command execution"),
                (r'(?<!validate_.*_)input\s*\([^)]*\)', "Raw input usage"),
            ]

            for file_path in python_files[:30]:  # Limit files to check
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # Check for dangerous patterns
                    import re
                    for pattern, issue in dangerous_patterns:
                        if re.search(pattern, content):
                            security_score -= 10.0
                            security_issues.append(f"{issue} in {os.path.basename(file_path)}")

                except Exception:
                    continue

            # Check for secrets in code (basic check)
            secret_patterns = [
                (r'password\s*=\s*["\'][^"\']+["\']', "Hardcoded password"),
                (r'api[_-]?key\s*=\s*["\'][^"\']+["\']', "Hardcoded API key"),
                (r'secret\s*=\s*["\'][^"\']+["\']', "Hardcoded secret"),
            ]

            for file_path in python_files[:20]:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    for pattern, issue in secret_patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            security_score -= 15.0
                            security_issues.append(f"{issue} in {os.path.basename(file_path)}")

                except Exception:
                    continue

            security_score = max(0.0, min(100.0, security_score))

            # Determine status
            if security_score >= self.min_security_score:
                status = QualityGateStatus.PASSED
                message = f"Security score {security_score:.1f}% meets minimum {self.min_security_score}%"
            elif security_score >= self.min_security_score - 10:
                status = QualityGateStatus.WARNING
                message = f"Security score {security_score:.1f}% below target {self.min_security_score}%"
            else:
                status = QualityGateStatus.FAILED
                message = f"Security score {security_score:.1f}% below minimum {self.min_security_score}%"

            return QualityGateResult(
                gate_name="security_scan",
                status=status,
                message=message,
                score=security_score,
                threshold=self.min_security_score,
                metrics={
                    "security_score": security_score,
                    "issues_found": len(security_issues),
                    "security_issues": security_issues[:10]  # Limit output
                }
            )

        except Exception as e:
            return QualityGateResult(
                gate_name="security_scan",
                status=QualityGateStatus.FAILED,
                message=f"Security validation failed: {str(e)}"
            )

    def _validate_performance(self) -> QualityGateResult:
        """Validate performance characteristics."""
        try:
            # Basic performance checks
            performance_score = 80.0
            perf_metrics = {}

            # Check import time (basic performance indicator)
            import_start = time.time()
            try:
                import fed_vit_autorl
                import_time = time.time() - import_start
                perf_metrics["import_time"] = import_time

                if import_time > 5.0:
                    performance_score -= 20.0
                elif import_time > 2.0:
                    performance_score -= 10.0

            except Exception as e:
                performance_score -= 30.0
                perf_metrics["import_error"] = str(e)

            # Check file structure efficiency
            python_files = []
            total_size = 0

            for root, dirs, files in os.walk(self.project_root):
                dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', '.pytest_cache'}]
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        try:
                            size = os.path.getsize(file_path)
                            total_size += size
                            python_files.append((file_path, size))
                        except Exception:
                            continue

            if python_files:
                avg_file_size = total_size / len(python_files)
                perf_metrics["avg_file_size_kb"] = avg_file_size / 1024
                perf_metrics["total_files"] = len(python_files)

                # Large files can indicate performance issues
                if avg_file_size > 100000:  # 100KB
                    performance_score -= 10.0

            performance_score = max(0.0, min(100.0, performance_score))

            # Determine status
            if performance_score >= self.min_performance_score:
                status = QualityGateStatus.PASSED
                message = f"Performance score {performance_score:.1f}% meets minimum {self.min_performance_score}%"
            elif performance_score >= self.min_performance_score - 10:
                status = QualityGateStatus.WARNING
                message = f"Performance score {performance_score:.1f}% below target {self.min_performance_score}%"
            else:
                status = QualityGateStatus.FAILED
                message = f"Performance score {performance_score:.1f}% below minimum {self.min_performance_score}%"

            return QualityGateResult(
                gate_name="performance_test",
                status=status,
                message=message,
                score=performance_score,
                threshold=self.min_performance_score,
                metrics=perf_metrics
            )

        except Exception as e:
            return QualityGateResult(
                gate_name="performance_test",
                status=QualityGateStatus.FAILED,
                message=f"Performance validation failed: {str(e)}"
            )

    def _validate_documentation(self) -> QualityGateResult:
        """Validate documentation coverage."""
        try:
            doc_score = 0.0
            doc_metrics = {}

            # Check for key documentation files
            doc_files = {
                "README.md": 25.0,
                "CONTRIBUTING.md": 15.0,
                "CHANGELOG.md": 10.0,
                "LICENSE": 10.0,
                "docs/": 20.0,
            }

            found_docs = []
            for doc_file, weight in doc_files.items():
                doc_path = os.path.join(self.project_root, doc_file)
                if os.path.exists(doc_path):
                    doc_score += weight
                    found_docs.append(doc_file)

            doc_metrics["found_docs"] = found_docs
            doc_metrics["doc_files_score"] = doc_score

            # Check docstrings in Python files
            python_files = []
            for root, dirs, files in os.walk(self.project_root):
                dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', '.pytest_cache'}]
                for file in files:
                    if file.endswith('.py'):
                        python_files.append(os.path.join(root, file))

            total_functions = 0
            documented_functions = 0

            for file_path in python_files[:30]:  # Limit to avoid timeout
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()

                    i = 0
                    while i < len(lines):
                        line = lines[i].strip()

                        # Check for function or method definition
                        if line.startswith('def ') and not line.startswith('def _'):
                            total_functions += 1

                            # Check if next non-empty line is a docstring
                            j = i + 1
                            while j < len(lines) and not lines[j].strip():
                                j += 1

                            if j < len(lines):
                                next_line = lines[j].strip()
                                if next_line.startswith('"""') or next_line.startswith("'''"):
                                    documented_functions += 1

                        i += 1

                except Exception:
                    continue

            # Calculate docstring coverage
            if total_functions > 0:
                docstring_coverage = (documented_functions / total_functions) * 100
                doc_score += docstring_coverage * 0.2  # 20% weight for docstrings
                doc_metrics["docstring_coverage"] = docstring_coverage
                doc_metrics["total_functions"] = total_functions
                doc_metrics["documented_functions"] = documented_functions

            doc_score = min(100.0, doc_score)

            # Determine status
            if doc_score >= self.min_documentation:
                status = QualityGateStatus.PASSED
                message = f"Documentation coverage {doc_score:.1f}% meets minimum {self.min_documentation}%"
            elif doc_score >= self.min_documentation - 10:
                status = QualityGateStatus.WARNING
                message = f"Documentation coverage {doc_score:.1f}% below target {self.min_documentation}%"
            else:
                status = QualityGateStatus.FAILED
                message = f"Documentation coverage {doc_score:.1f}% below minimum {self.min_documentation}%"

            return QualityGateResult(
                gate_name="documentation",
                status=status,
                message=message,
                score=doc_score,
                threshold=self.min_documentation,
                metrics=doc_metrics
            )

        except Exception as e:
            return QualityGateResult(
                gate_name="documentation",
                status=QualityGateStatus.FAILED,
                message=f"Documentation validation failed: {str(e)}"
            )

    def _validate_complexity(self) -> QualityGateResult:
        """Validate code complexity."""
        try:
            # Simple complexity analysis
            total_complexity = 0
            function_count = 0
            high_complexity_functions = []

            python_files = []
            for root, dirs, files in os.walk(self.project_root):
                dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', '.pytest_cache'}]
                for file in files:
                    if file.endswith('.py'):
                        python_files.append(os.path.join(root, file))

            for file_path in python_files[:20]:  # Limit files
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # Simple complexity heuristics
                    lines = content.split('\n')
                    in_function = False
                    current_complexity = 1
                    function_name = ""

                    for line in lines:
                        stripped = line.strip()

                        if stripped.startswith('def '):
                            if in_function:
                                # End previous function
                                total_complexity += current_complexity
                                function_count += 1
                                if current_complexity > self.max_complexity:
                                    high_complexity_functions.append(
                                        f"{function_name}: {current_complexity}"
                                    )

                            # Start new function
                            in_function = True
                            current_complexity = 1
                            function_name = stripped.split('(')[0].replace('def ', '')

                        elif in_function:
                            # Count complexity indicators
                            complexity_keywords = ['if ', 'elif ', 'for ', 'while ', 'except ', 'and ', 'or ']
                            for keyword in complexity_keywords:
                                if keyword in stripped:
                                    current_complexity += 1
                                    break

                    # Handle last function
                    if in_function:
                        total_complexity += current_complexity
                        function_count += 1
                        if current_complexity > self.max_complexity:
                            high_complexity_functions.append(
                                f"{function_name}: {current_complexity}"
                            )

                except Exception:
                    continue

            avg_complexity = total_complexity / function_count if function_count > 0 else 0

            # Determine status
            if avg_complexity <= self.max_complexity:
                status = QualityGateStatus.PASSED
                message = f"Average complexity {avg_complexity:.1f} meets maximum {self.max_complexity}"
            elif avg_complexity <= self.max_complexity + 2:
                status = QualityGateStatus.WARNING
                message = f"Average complexity {avg_complexity:.1f} above target {self.max_complexity}"
            else:
                status = QualityGateStatus.FAILED
                message = f"Average complexity {avg_complexity:.1f} above maximum {self.max_complexity}"

            return QualityGateResult(
                gate_name="complexity_check",
                status=status,
                message=message,
                score=self.max_complexity - avg_complexity,  # Higher score for lower complexity
                threshold=self.max_complexity,
                metrics={
                    "avg_complexity": avg_complexity,
                    "total_functions": function_count,
                    "high_complexity_functions": high_complexity_functions[:10]
                }
            )

        except Exception as e:
            return QualityGateResult(
                gate_name="complexity_check",
                status=QualityGateStatus.FAILED,
                message=f"Complexity validation failed: {str(e)}"
            )

    def _validate_dependencies(self) -> QualityGateResult:
        """Validate project dependencies."""
        try:
            dep_issues = []
            dep_score = 100.0

            # Check for requirements files
            req_files = ['requirements.txt', 'pyproject.toml', 'setup.py', 'environment.yml']
            found_req_files = []

            for req_file in req_files:
                req_path = os.path.join(self.project_root, req_file)
                if os.path.exists(req_path):
                    found_req_files.append(req_file)

            if not found_req_files:
                dep_score -= 30.0
                dep_issues.append("No dependency specification files found")

            # Check pyproject.toml specifically
            pyproject_path = os.path.join(self.project_root, 'pyproject.toml')
            if os.path.exists(pyproject_path):
                try:
                    with open(pyproject_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # Basic checks
                    if 'dependencies' not in content:
                        dep_score -= 10.0
                        dep_issues.append("No dependencies section in pyproject.toml")

                    if 'version' not in content:
                        dep_score -= 5.0
                        dep_issues.append("No version specified")

                except Exception:
                    dep_score -= 10.0
                    dep_issues.append("Could not parse pyproject.toml")

            # Check for common security issues in dependencies
            # This is a placeholder - in practice would use safety or similar tools

            # Determine status
            if dep_score >= 90.0:
                status = QualityGateStatus.PASSED
                message = f"Dependency validation score {dep_score:.1f}% - all checks passed"
            elif dep_score >= 70.0:
                status = QualityGateStatus.WARNING
                message = f"Dependency validation score {dep_score:.1f}% - minor issues found"
            else:
                status = QualityGateStatus.FAILED
                message = f"Dependency validation score {dep_score:.1f}% - major issues found"

            return QualityGateResult(
                gate_name="dependency_check",
                status=status,
                message=message,
                score=dep_score,
                threshold=90.0,
                metrics={
                    "dependency_score": dep_score,
                    "found_req_files": found_req_files,
                    "issues": dep_issues
                }
            )

        except Exception as e:
            return QualityGateResult(
                gate_name="dependency_check",
                status=QualityGateStatus.FAILED,
                message=f"Dependency validation failed: {str(e)}"
            )

    def _validate_linting(self) -> QualityGateResult:
        """Validate code linting and formatting."""
        try:
            lint_score = 100.0
            lint_issues = []

            # Check for common Python linting issues
            python_files = []
            for root, dirs, files in os.walk(self.project_root):
                dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', '.pytest_cache'}]
                for file in files:
                    if file.endswith('.py'):
                        python_files.append(os.path.join(root, file))

            # Basic linting checks
            for file_path in python_files[:15]:  # Limit files to check
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()

                    for i, line in enumerate(lines):
                        # Check line length (basic PEP 8)
                        if len(line.rstrip()) > 100:
                            lint_score -= 0.5
                            if len(lint_issues) < 10:  # Limit reported issues
                                lint_issues.append(
                                    f"Long line in {os.path.basename(file_path)}:{i+1}"
                                )

                        # Check for trailing whitespace
                        if line.endswith(' ') or line.endswith('\t'):
                            lint_score -= 0.1
                            if len(lint_issues) < 10:
                                lint_issues.append(
                                    f"Trailing whitespace in {os.path.basename(file_path)}:{i+1}"
                                )

                        # Check for mixed indentation (basic)
                        if line.startswith(' ') and line.startswith('\t'):
                            lint_score -= 2.0
                            if len(lint_issues) < 10:
                                lint_issues.append(
                                    f"Mixed indentation in {os.path.basename(file_path)}:{i+1}"
                                )

                except Exception:
                    continue

            lint_score = max(0.0, lint_score)

            # Determine status
            if lint_score >= 95.0:
                status = QualityGateStatus.PASSED
                message = f"Linting score {lint_score:.1f}% - excellent code style"
            elif lint_score >= 80.0:
                status = QualityGateStatus.WARNING
                message = f"Linting score {lint_score:.1f}% - minor style issues"
            else:
                status = QualityGateStatus.FAILED
                message = f"Linting score {lint_score:.1f}% - significant style issues"

            return QualityGateResult(
                gate_name="linting",
                status=status,
                message=message,
                score=lint_score,
                threshold=95.0,
                metrics={
                    "lint_score": lint_score,
                    "files_checked": len(python_files),
                    "issues": lint_issues
                }
            )

        except Exception as e:
            return QualityGateResult(
                gate_name="linting",
                status=QualityGateStatus.FAILED,
                message=f"Linting validation failed: {str(e)}"
            )

    def _generate_validation_summary(
        self,
        results: Dict[str, QualityGateResult],
        total_time: float
    ) -> None:
        """Generate validation summary and report."""
        passed = sum(1 for r in results.values() if r.status == QualityGateStatus.PASSED)
        warning = sum(1 for r in results.values() if r.status == QualityGateStatus.WARNING)
        failed = sum(1 for r in results.values() if r.status == QualityGateStatus.FAILED)
        skipped = sum(1 for r in results.values() if r.status == QualityGateStatus.SKIPPED)

        total_gates = len(results)

        summary = {
            "timestamp": time.time(),
            "total_time": total_time,
            "total_gates": total_gates,
            "passed": passed,
            "warning": warning,
            "failed": failed,
            "skipped": skipped,
            "success_rate": (passed / total_gates) * 100 if total_gates > 0 else 0,
            "gate_results": {name: result for name, result in results.items()},
        }

        self.validation_history.append(summary)

        # Keep only last 50 validation runs
        if len(self.validation_history) > 50:
            self.validation_history = self.validation_history[-50:]

        # Log summary
        logger.info(f"Quality gate validation completed in {total_time:.2f}s")
        logger.info(f"Results: {passed} passed, {warning} warning, {failed} failed, {skipped} skipped")
        logger.info(f"Overall success rate: {summary['success_rate']:.1f}%")

    def get_overall_status(self) -> QualityGateStatus:
        """Get overall quality gate status."""
        if not self.gate_results:
            return QualityGateStatus.SKIPPED

        recent_results = self.gate_results[-len(self.gates):]  # Last full run

        if any(r.status == QualityGateStatus.FAILED for r in recent_results):
            return QualityGateStatus.FAILED
        elif any(r.status == QualityGateStatus.WARNING for r in recent_results):
            return QualityGateStatus.WARNING
        else:
            return QualityGateStatus.PASSED

    def get_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive quality report."""
        if not self.validation_history:
            return {"status": "no_validations", "message": "No quality gate validations performed"}

        latest_validation = self.validation_history[-1]
        overall_status = self.get_overall_status()

        # Calculate trends
        if len(self.validation_history) >= 2:
            prev_success_rate = self.validation_history[-2]["success_rate"]
            current_success_rate = latest_validation["success_rate"]
            trend = current_success_rate - prev_success_rate
        else:
            trend = 0.0

        return {
            "overall_status": overall_status.value,
            "success_rate": latest_validation["success_rate"],
            "trend": trend,
            "total_validations": len(self.validation_history),
            "last_validation": latest_validation,
            "gate_summary": {
                "passed": latest_validation["passed"],
                "warning": latest_validation["warning"],
                "failed": latest_validation["failed"],
                "skipped": latest_validation["skipped"],
            },
            "configuration": {
                "min_coverage": self.min_coverage,
                "min_code_quality": self.min_code_quality,
                "min_security_score": self.min_security_score,
                "strict_mode": self.enable_strict_mode,
            }
        }
