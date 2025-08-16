#!/usr/bin/env python3
"""Quality gates validation for Fed-ViT-AutoRL project."""

import os
import sys
import ast
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class QualityGate:
    """Base class for quality gates."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.passed = False
        self.messages = []
        self.score = 0.0
    
    def validate(self, project_root: Path) -> bool:
        """Validate the quality gate."""
        raise NotImplementedError
    
    def add_message(self, message: str, level: str = "INFO"):
        """Add a validation message."""
        self.messages.append(f"[{level}] {message}")
    
    def get_report(self) -> Dict:
        """Get validation report."""
        return {
            'name': self.name,
            'description': self.description,
            'passed': self.passed,
            'score': self.score,
            'messages': self.messages
        }


class CodeQualityGate(QualityGate):
    """Validate code quality standards."""
    
    def __init__(self):
        super().__init__(
            "Code Quality",
            "Validate Python code quality, structure, and best practices"
        )
    
    def validate(self, project_root: Path) -> bool:
        """Validate code quality."""
        total_score = 0.0
        max_score = 0.0
        
        # Check Python files structure
        py_files = list(project_root.rglob("*.py"))
        if not py_files:
            self.add_message("No Python files found", "ERROR")
            return False
        
        self.add_message(f"Found {len(py_files)} Python files")
        
        # Validate imports and structure
        import_score, import_max = self._validate_imports(py_files)
        total_score += import_score
        max_score += import_max
        
        # Validate docstrings
        doc_score, doc_max = self._validate_docstrings(py_files)
        total_score += doc_score
        max_score += doc_max
        
        # Validate type hints
        type_score, type_max = self._validate_type_hints(py_files)
        total_score += type_score
        max_score += type_max
        
        # Validate error handling
        error_score, error_max = self._validate_error_handling(py_files)
        total_score += error_score
        max_score += error_max
        
        self.score = total_score / max_score if max_score > 0 else 0.0
        self.passed = self.score >= 0.8  # 80% threshold
        
        self.add_message(f"Overall code quality score: {self.score:.2%}")
        
        return self.passed
    
    def _validate_imports(self, py_files: List[Path]) -> Tuple[float, float]:
        """Validate import statements."""
        total_files = len(py_files)
        files_with_proper_imports = 0
        
        for file_path in py_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                # Check for proper import organization
                has_proper_imports = True
                imports = []
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        imports.append(node)
                
                if imports:
                    files_with_proper_imports += 1
                    
            except Exception as e:
                self.add_message(f"Failed to parse {file_path}: {e}", "WARNING")
        
        score = files_with_proper_imports / total_files if total_files > 0 else 0
        self.add_message(f"Import quality: {score:.1%} ({files_with_proper_imports}/{total_files})")
        
        return score * 25, 25  # 25% weight for imports
    
    def _validate_docstrings(self, py_files: List[Path]) -> Tuple[float, float]:
        """Validate docstring coverage."""
        total_functions = 0
        functions_with_docstrings = 0
        
        for file_path in py_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        total_functions += 1
                        
                        # Check if has docstring
                        if (node.body and 
                            isinstance(node.body[0], ast.Expr) and
                            isinstance(node.body[0].value, ast.Constant) and
                            isinstance(node.body[0].value.value, str)):
                            functions_with_docstrings += 1
                            
            except Exception as e:
                self.add_message(f"Failed to analyze docstrings in {file_path}: {e}", "WARNING")
        
        score = functions_with_docstrings / total_functions if total_functions > 0 else 0
        self.add_message(f"Docstring coverage: {score:.1%} ({functions_with_docstrings}/{total_functions})")
        
        return score * 25, 25  # 25% weight for docstrings
    
    def _validate_type_hints(self, py_files: List[Path]) -> Tuple[float, float]:
        """Validate type hint coverage."""
        total_functions = 0
        functions_with_hints = 0
        
        for file_path in py_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        total_functions += 1
                        
                        # Check for return type annotation
                        has_return_hint = node.returns is not None
                        
                        # Check for argument type annotations
                        has_arg_hints = any(arg.annotation is not None for arg in node.args.args)
                        
                        if has_return_hint or has_arg_hints:
                            functions_with_hints += 1
                            
            except Exception as e:
                self.add_message(f"Failed to analyze type hints in {file_path}: {e}", "WARNING")
        
        score = functions_with_hints / total_functions if total_functions > 0 else 0
        self.add_message(f"Type hint coverage: {score:.1%} ({functions_with_hints}/{total_functions})")
        
        return score * 25, 25  # 25% weight for type hints
    
    def _validate_error_handling(self, py_files: List[Path]) -> Tuple[float, float]:
        """Validate error handling patterns."""
        total_files = 0
        files_with_error_handling = 0
        
        for file_path in py_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                total_files += 1
                
                # Check for try-except blocks
                has_error_handling = any(
                    isinstance(node, ast.Try) for node in ast.walk(tree)
                )
                
                if has_error_handling:
                    files_with_error_handling += 1
                    
            except Exception as e:
                self.add_message(f"Failed to analyze error handling in {file_path}: {e}", "WARNING")
        
        score = files_with_error_handling / total_files if total_files > 0 else 0
        self.add_message(f"Error handling coverage: {score:.1%} ({files_with_error_handling}/{total_files})")
        
        return score * 25, 25  # 25% weight for error handling


class ArchitectureGate(QualityGate):
    """Validate system architecture and design patterns."""
    
    def __init__(self):
        super().__init__(
            "Architecture Quality",
            "Validate system architecture, modularity, and design patterns"
        )
    
    def validate(self, project_root: Path) -> bool:
        """Validate architecture quality."""
        total_score = 0.0
        max_score = 0.0
        
        # Check project structure
        structure_score, structure_max = self._validate_project_structure(project_root)
        total_score += structure_score
        max_score += structure_max
        
        # Check module organization
        module_score, module_max = self._validate_module_organization(project_root)
        total_score += module_score
        max_score += module_max
        
        # Check design patterns
        pattern_score, pattern_max = self._validate_design_patterns(project_root)
        total_score += pattern_score
        max_score += pattern_max
        
        self.score = total_score / max_score if max_score > 0 else 0.0
        self.passed = self.score >= 0.7  # 70% threshold
        
        self.add_message(f"Overall architecture score: {self.score:.2%}")
        
        return self.passed
    
    def _validate_project_structure(self, project_root: Path) -> Tuple[float, float]:
        """Validate project structure."""
        required_dirs = [
            "fed_vit_autorl",
            "tests",
            "configs",
            "scripts"
        ]
        
        required_files = [
            "pyproject.toml",
            "README.md",
            "Dockerfile",
            "docker-compose.yml"
        ]
        
        structure_score = 0
        max_structure_score = len(required_dirs) + len(required_files)
        
        # Check directories
        for dir_name in required_dirs:
            dir_path = project_root / dir_name
            if dir_path.exists() and dir_path.is_dir():
                structure_score += 1
                self.add_message(f"✓ Found required directory: {dir_name}")
            else:
                self.add_message(f"✗ Missing required directory: {dir_name}", "WARNING")
        
        # Check files
        for file_name in required_files:
            file_path = project_root / file_name
            if file_path.exists() and file_path.is_file():
                structure_score += 1
                self.add_message(f"✓ Found required file: {file_name}")
            else:
                self.add_message(f"✗ Missing required file: {file_name}", "WARNING")
        
        score = structure_score / max_structure_score
        self.add_message(f"Project structure score: {score:.1%}")
        
        return score * 40, 40  # 40% weight
    
    def _validate_module_organization(self, project_root: Path) -> Tuple[float, float]:
        """Validate module organization."""
        fed_vit_dir = project_root / "fed_vit_autorl"
        
        if not fed_vit_dir.exists():
            self.add_message("Main package directory not found", "ERROR")
            return 0, 30
        
        expected_modules = [
            "models",
            "federated", 
            "evaluation",
            "edge",
            "reinforcement",
            "simulation",
            "safety",
            "optimization"
        ]
        
        found_modules = 0
        for module in expected_modules:
            module_path = fed_vit_dir / module
            if module_path.exists():
                found_modules += 1
                
                # Check for __init__.py
                init_file = module_path / "__init__.py"
                if init_file.exists():
                    self.add_message(f"✓ Module {module} properly initialized")
                else:
                    self.add_message(f"⚠ Module {module} missing __init__.py", "WARNING")
            else:
                self.add_message(f"✗ Missing module: {module}", "WARNING")
        
        score = found_modules / len(expected_modules)
        self.add_message(f"Module organization score: {score:.1%}")
        
        return score * 30, 30  # 30% weight
    
    def _validate_design_patterns(self, project_root: Path) -> Tuple[float, float]:
        """Validate design patterns usage."""
        patterns_found = 0
        total_patterns = 5
        
        # Check for factory pattern
        if self._check_pattern(project_root, r"class.*Factory"):
            patterns_found += 1
            self.add_message("✓ Factory pattern detected")
        
        # Check for singleton pattern
        if self._check_pattern(project_root, r"_instance.*=.*None"):
            patterns_found += 1
            self.add_message("✓ Singleton pattern detected")
        
        # Check for observer pattern
        if self._check_pattern(project_root, r"def notify|class.*Observer"):
            patterns_found += 1
            self.add_message("✓ Observer pattern detected")
        
        # Check for strategy pattern
        if self._check_pattern(project_root, r"class.*Strategy"):
            patterns_found += 1
            self.add_message("✓ Strategy pattern detected")
        
        # Check for dependency injection
        if self._check_pattern(project_root, r"def __init__.*:.*\w+"):
            patterns_found += 1
            self.add_message("✓ Dependency injection pattern detected")
        
        score = patterns_found / total_patterns
        self.add_message(f"Design patterns score: {score:.1%}")
        
        return score * 30, 30  # 30% weight
    
    def _check_pattern(self, project_root: Path, pattern: str) -> bool:
        """Check if a pattern exists in the codebase."""
        py_files = list(project_root.rglob("*.py"))
        
        for file_path in py_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if re.search(pattern, content, re.IGNORECASE):
                    return True
                    
            except Exception:
                continue
        
        return False


class SecurityGate(QualityGate):
    """Validate security practices."""
    
    def __init__(self):
        super().__init__(
            "Security",
            "Validate security practices and potential vulnerabilities"
        )
    
    def validate(self, project_root: Path) -> bool:
        """Validate security practices."""
        total_score = 0.0
        max_score = 0.0
        
        # Check for security anti-patterns
        security_score, security_max = self._check_security_patterns(project_root)
        total_score += security_score
        max_score += security_max
        
        # Check configuration security
        config_score, config_max = self._check_config_security(project_root)
        total_score += config_score
        max_score += config_max
        
        self.score = total_score / max_score if max_score > 0 else 0.0
        self.passed = self.score >= 0.8  # 80% threshold for security
        
        self.add_message(f"Overall security score: {self.score:.2%}")
        
        return self.passed
    
    def _check_security_patterns(self, project_root: Path) -> Tuple[float, float]:
        """Check for security anti-patterns."""
        py_files = list(project_root.rglob("*.py"))
        security_issues = 0
        total_checks = 0
        
        dangerous_patterns = [
            (r"(?<!\.)\beval\s*\([^)]*['\"]", "Use of eval() function with string"),
            (r"(?<!\.)\bexec\s*\([^)]*['\"]", "Use of exec() function with string"),
            (r"os\.system\s*\(", "Use of os.system()"),
            (r"subprocess\.call.*shell=True", "Shell injection risk"),
            (r"password\s*=\s*[\"'][^\"']+[\"']", "Hardcoded password"),
            (r"api_key\s*=\s*[\"'][^\"']+[\"']", "Hardcoded API key"),
        ]
        
        for file_path in py_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for pattern, description in dangerous_patterns:
                    total_checks += 1
                    if re.search(pattern, content, re.IGNORECASE):
                        security_issues += 1
                        self.add_message(f"Security issue in {file_path}: {description}", "WARNING")
                
            except Exception as e:
                self.add_message(f"Failed to analyze {file_path}: {e}", "WARNING")
        
        # Score is inverted (fewer issues = higher score)
        score = 1.0 - (security_issues / max(total_checks, 1))
        self.add_message(f"Security pattern analysis: {security_issues} issues found")
        
        return score * 60, 60  # 60% weight
    
    def _check_config_security(self, project_root: Path) -> Tuple[float, float]:
        """Check configuration security."""
        config_files = list(project_root.rglob("*.yaml")) + list(project_root.rglob("*.yml"))
        secure_configs = 0
        total_configs = len(config_files)
        
        if total_configs == 0:
            self.add_message("No configuration files found", "WARNING")
            return 0, 40
        
        for config_file in config_files:
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for security best practices
                has_security_section = "security:" in content.lower()
                has_auth_config = "auth" in content.lower()
                has_ssl_config = "ssl" in content.lower() or "tls" in content.lower()
                
                if has_security_section or has_auth_config or has_ssl_config:
                    secure_configs += 1
                    self.add_message(f"✓ Security configuration found in {config_file.name}")
                else:
                    self.add_message(f"⚠ No security configuration in {config_file.name}", "WARNING")
                
            except Exception as e:
                self.add_message(f"Failed to analyze {config_file}: {e}", "WARNING")
        
        score = secure_configs / total_configs
        self.add_message(f"Configuration security score: {score:.1%}")
        
        return score * 40, 40  # 40% weight


class DocumentationGate(QualityGate):
    """Validate documentation quality and coverage."""
    
    def __init__(self):
        super().__init__(
            "Documentation",
            "Validate documentation quality, coverage, and usefulness"
        )
    
    def validate(self, project_root: Path) -> bool:
        """Validate documentation."""
        total_score = 0.0
        max_score = 0.0
        
        # Check README quality
        readme_score, readme_max = self._validate_readme(project_root)
        total_score += readme_score
        max_score += readme_max
        
        # Check documentation structure
        docs_score, docs_max = self._validate_docs_structure(project_root)
        total_score += docs_score
        max_score += docs_max
        
        # Check code documentation
        code_docs_score, code_docs_max = self._validate_code_documentation(project_root)
        total_score += code_docs_score
        max_score += code_docs_max
        
        self.score = total_score / max_score if max_score > 0 else 0.0
        self.passed = self.score >= 0.7  # 70% threshold
        
        self.add_message(f"Overall documentation score: {self.score:.2%}")
        
        return self.passed
    
    def _validate_readme(self, project_root: Path) -> Tuple[float, float]:
        """Validate README.md quality."""
        readme_path = project_root / "README.md"
        
        if not readme_path.exists():
            self.add_message("README.md not found", "ERROR")
            return 0, 40
        
        try:
            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            score = 0
            max_score = 6
            
            # Check for essential sections
            sections = [
                ("installation", r"##?\s*install"),
                ("usage", r"##?\s*usage|##?\s*quick\s*start"),
                ("features", r"##?\s*features|##?\s*overview"),
                ("examples", r"##?\s*examples"),
                ("contributing", r"##?\s*contribut"),
                ("license", r"##?\s*license")
            ]
            
            for section_name, pattern in sections:
                if re.search(pattern, content, re.IGNORECASE):
                    score += 1
                    self.add_message(f"✓ README has {section_name} section")
                else:
                    self.add_message(f"⚠ README missing {section_name} section", "WARNING")
            
            # Check content length (reasonable documentation)
            if len(content) > 1000:
                self.add_message("✓ README has substantial content")
            else:
                self.add_message("⚠ README content seems brief", "WARNING")
            
            final_score = score / max_score
            self.add_message(f"README quality score: {final_score:.1%}")
            
            return final_score * 40, 40  # 40% weight
            
        except Exception as e:
            self.add_message(f"Failed to analyze README: {e}", "ERROR")
            return 0, 40
    
    def _validate_docs_structure(self, project_root: Path) -> Tuple[float, float]:
        """Validate documentation structure."""
        docs_dir = project_root / "docs"
        
        if not docs_dir.exists():
            self.add_message("docs/ directory not found", "WARNING")
            return 0.5, 30  # Partial credit if docs in README
        
        expected_docs = [
            "ARCHITECTURE.md",
            "DEVELOPMENT.md", 
            "IMPLEMENTATION_SUMMARY.md",
            "ROADMAP.md"
        ]
        
        found_docs = 0
        for doc in expected_docs:
            doc_path = docs_dir / doc
            if doc_path.exists():
                found_docs += 1
                self.add_message(f"✓ Found documentation: {doc}")
            else:
                self.add_message(f"⚠ Missing documentation: {doc}", "WARNING")
        
        score = found_docs / len(expected_docs)
        self.add_message(f"Documentation structure score: {score:.1%}")
        
        return score * 30, 30  # 30% weight
    
    def _validate_code_documentation(self, project_root: Path) -> Tuple[float, float]:
        """Validate code documentation."""
        py_files = list(project_root.rglob("*.py"))
        
        if not py_files:
            return 0, 30
        
        documented_classes = 0
        total_classes = 0
        
        for file_path in py_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        total_classes += 1
                        
                        # Check for class docstring
                        if (node.body and 
                            isinstance(node.body[0], ast.Expr) and
                            isinstance(node.body[0].value, ast.Constant) and
                            isinstance(node.body[0].value.value, str)):
                            documented_classes += 1
                            
            except Exception as e:
                self.add_message(f"Failed to analyze {file_path}: {e}", "WARNING")
        
        score = documented_classes / total_classes if total_classes > 0 else 1.0
        self.add_message(f"Code documentation score: {score:.1%} ({documented_classes}/{total_classes} classes)")
        
        return score * 30, 30  # 30% weight


class QualityGateRunner:
    """Run all quality gates and generate report."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.gates = [
            CodeQualityGate(),
            ArchitectureGate(),
            SecurityGate(),
            DocumentationGate(),
        ]
        self.results = []
    
    def run_all_gates(self) -> bool:
        """Run all quality gates."""
        logger.info("Starting quality gate validation...")
        
        all_passed = True
        
        for gate in self.gates:
            logger.info(f"Running {gate.name} gate...")
            try:
                passed = gate.validate(self.project_root)
                self.results.append(gate.get_report())
                
                if passed:
                    logger.info(f"✅ {gate.name} gate PASSED (Score: {gate.score:.1%})")
                else:
                    logger.error(f"❌ {gate.name} gate FAILED (Score: {gate.score:.1%})")
                    all_passed = False
                    
            except Exception as e:
                logger.error(f"❌ {gate.name} gate ERROR: {e}")
                all_passed = False
                
                # Add error result
                gate.passed = False
                gate.add_message(f"Gate execution failed: {e}", "ERROR")
                self.results.append(gate.get_report())
        
        return all_passed
    
    def generate_report(self) -> Dict:
        """Generate comprehensive quality report."""
        total_score = sum(result['score'] for result in self.results)
        avg_score = total_score / len(self.results) if self.results else 0
        
        passed_gates = sum(1 for result in self.results if result['passed'])
        
        return {
            'overall_passed': all(result['passed'] for result in self.results),
            'overall_score': avg_score,
            'gates_passed': f"{passed_gates}/{len(self.results)}",
            'results': self.results,
            'summary': {
                'total_gates': len(self.results),
                'passed_gates': passed_gates,
                'failed_gates': len(self.results) - passed_gates,
                'average_score': avg_score,
            }
        }
    
    def save_report(self, output_path: Path) -> None:
        """Save quality report to file."""
        report = self.generate_report()
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Quality report saved to: {output_path}")


def main():
    """Main entry point."""
    project_root = Path(__file__).parent.parent
    
    logger.info(f"Running quality gates for project: {project_root}")
    
    runner = QualityGateRunner(project_root)
    all_passed = runner.run_all_gates()
    
    # Generate and save report
    report_path = project_root / "quality_report.json"
    runner.save_report(report_path)
    
    # Print summary
    report = runner.generate_report()
    print("\n" + "="*60)
    print("QUALITY GATES SUMMARY")
    print("="*60)
    print(f"Overall Status: {'✅ PASSED' if report['overall_passed'] else '❌ FAILED'}")
    print(f"Overall Score: {report['overall_score']:.1%}")
    print(f"Gates Passed: {report['gates_passed']}")
    print("\nDetailed Results:")
    
    for result in report['results']:
        status = "✅ PASSED" if result['passed'] else "❌ FAILED"
        print(f"  {result['name']}: {status} (Score: {result['score']:.1%})")
        
        # Show critical messages
        for message in result['messages'][-3:]:  # Last 3 messages
            if "[ERROR]" in message or "[WARNING]" in message:
                print(f"    {message}")
    
    print("="*60)
    
    # Exit with proper code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()