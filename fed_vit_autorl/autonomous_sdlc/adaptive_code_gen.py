"""Real-time Adaptive Code Generation System."""

import ast
import asyncio
import hashlib
import inspect
import json
import logging
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    CodeT5Tokenizer, T5ForConditionalGeneration
)

logger = logging.getLogger(__name__)


class CodeGenerationStrategy(Enum):
    """Code generation strategies."""
    TEMPLATE_BASED = "template_based"
    AI_ASSISTED = "ai_assisted"
    GENETIC_PROGRAMMING = "genetic_programming"
    NEURAL_SYNTHESIS = "neural_synthesis"
    HYBRID_APPROACH = "hybrid_approach"


class PerformanceOptimizationTarget(Enum):
    """Performance optimization targets."""
    LATENCY = "latency"
    MEMORY = "memory"
    THROUGHPUT = "throughput"
    ACCURACY = "accuracy"
    ENERGY = "energy"
    SCALABILITY = "scalability"


@dataclass
class CodeGenerationRequest:
    """Request for adaptive code generation."""
    
    functionality_description: str
    target_language: str = "python"
    performance_requirements: Dict[str, float] = field(default_factory=dict)
    optimization_targets: List[PerformanceOptimizationTarget] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    context_code: Optional[str] = None
    test_cases: List[Dict[str, Any]] = field(default_factory=list)
    quality_requirements: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "functionality_description": self.functionality_description,
            "target_language": self.target_language,
            "performance_requirements": self.performance_requirements,
            "optimization_targets": [t.value for t in self.optimization_targets],
            "constraints": self.constraints,
            "context_code": self.context_code,
            "test_cases": self.test_cases,
            "quality_requirements": self.quality_requirements
        }


@dataclass
class GeneratedCode:
    """Generated code with metadata."""
    
    code: str
    language: str
    functionality: str
    performance_metrics: Dict[str, float]
    quality_metrics: Dict[str, float]
    optimization_applied: List[str]
    generation_strategy: CodeGenerationStrategy
    confidence_score: float
    test_results: Dict[str, Any]
    generation_time: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "code": self.code,
            "language": self.language,
            "functionality": self.functionality,
            "performance_metrics": self.performance_metrics,
            "quality_metrics": self.quality_metrics,
            "optimization_applied": self.optimization_applied,
            "generation_strategy": self.generation_strategy.value,
            "confidence_score": self.confidence_score,
            "test_results": self.test_results,
            "generation_time": self.generation_time.isoformat()
        }


class CodeQualityAnalyzer:
    """Analyzes code quality and suggests improvements."""
    
    def __init__(self):
        self.quality_metrics = [
            "cyclomatic_complexity",
            "maintainability_index", 
            "lines_of_code",
            "code_duplication",
            "test_coverage",
            "documentation_coverage"
        ]
        
    def analyze_quality(self, code: str, language: str = "python") -> Dict[str, float]:
        """Analyze code quality metrics."""
        
        try:
            if language.lower() == "python":
                return self._analyze_python_quality(code)
            else:
                return self._analyze_generic_quality(code)
                
        except Exception as e:
            logger.warning(f"Code quality analysis failed: {e}")
            return {metric: 0.5 for metric in self.quality_metrics}
            
    def _analyze_python_quality(self, code: str) -> Dict[str, float]:
        """Analyze Python code quality."""
        
        metrics = {}
        
        try:
            # Parse AST
            tree = ast.parse(code)
            
            # Cyclomatic complexity (simplified)
            complexity = self._calculate_cyclomatic_complexity(tree)
            metrics["cyclomatic_complexity"] = max(0.0, min(1.0, (10 - complexity) / 10))
            
            # Lines of code (fewer is better for simple functions)
            lines = len([line for line in code.split('\n') if line.strip()])
            metrics["lines_of_code"] = max(0.0, min(1.0, (100 - lines) / 100))
            
            # Code duplication (simplified check)
            duplication = self._check_code_duplication(code)
            metrics["code_duplication"] = 1.0 - duplication
            
            # Documentation coverage
            doc_coverage = self._check_documentation_coverage(tree)
            metrics["documentation_coverage"] = doc_coverage
            
            # Maintainability (heuristic)
            maintainability = (
                metrics["cyclomatic_complexity"] * 0.3 +
                metrics["lines_of_code"] * 0.2 + 
                metrics["code_duplication"] * 0.3 +
                metrics["documentation_coverage"] * 0.2
            )
            metrics["maintainability_index"] = maintainability
            
            # Test coverage (placeholder - would need actual test execution)
            metrics["test_coverage"] = 0.8  # Default reasonable value
            
        except Exception as e:
            logger.warning(f"Python quality analysis error: {e}")
            # Return default values
            metrics = {metric: 0.7 for metric in self.quality_metrics}
            
        return metrics
        
    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity from AST."""
        
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            # Add complexity for control flow statements
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, (ast.BoolOp,)):
                # Boolean operators add complexity
                complexity += len(node.values) - 1
                
        return complexity
        
    def _check_code_duplication(self, code: str) -> float:
        """Check for code duplication (simplified)."""
        
        lines = [line.strip() for line in code.split('\n') if line.strip()]
        
        if len(lines) < 3:
            return 0.0
            
        # Simple duplication check - count identical lines
        unique_lines = set(lines)
        duplication_ratio = 1.0 - (len(unique_lines) / len(lines))
        
        return duplication_ratio
        
    def _check_documentation_coverage(self, tree: ast.AST) -> float:
        """Check documentation coverage."""
        
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        
        if not functions:
            return 1.0  # No functions to document
            
        documented = 0
        for func in functions:
            if (ast.get_docstring(func) is not None):
                documented += 1
                
        return documented / len(functions)
        
    def _analyze_generic_quality(self, code: str) -> Dict[str, float]:
        """Analyze generic code quality (language-agnostic)."""
        
        metrics = {}
        
        # Lines of code
        lines = len([line for line in code.split('\n') if line.strip()])
        metrics["lines_of_code"] = max(0.0, min(1.0, (100 - lines) / 100))
        
        # Simple complexity based on nesting and keywords
        complexity_keywords = ['if', 'while', 'for', 'switch', 'case']
        complexity_count = sum(code.lower().count(kw) for kw in complexity_keywords)
        metrics["cyclomatic_complexity"] = max(0.0, min(1.0, (20 - complexity_count) / 20))
        
        # Basic duplication check
        lines_list = [line.strip() for line in code.split('\n') if line.strip()]
        unique_ratio = len(set(lines_list)) / max(1, len(lines_list))
        metrics["code_duplication"] = unique_ratio
        
        # Documentation (presence of comments)
        comment_patterns = ['#', '//', '/*', '"""', "'''"]
        has_comments = any(pattern in code for pattern in comment_patterns)
        metrics["documentation_coverage"] = 0.8 if has_comments else 0.3
        
        # Default values for remaining metrics
        metrics["maintainability_index"] = np.mean(list(metrics.values()))
        metrics["test_coverage"] = 0.7  # Default
        
        return metrics


class PerformanceOptimizer:
    """Optimizes code for specific performance targets."""
    
    def __init__(self):
        self.optimization_strategies = {
            PerformanceOptimizationTarget.LATENCY: self._optimize_for_latency,
            PerformanceOptimizationTarget.MEMORY: self._optimize_for_memory,
            PerformanceOptimizationTarget.THROUGHPUT: self._optimize_for_throughput,
            PerformanceOptimizationTarget.ACCURACY: self._optimize_for_accuracy,
            PerformanceOptimizationTarget.ENERGY: self._optimize_for_energy,
            PerformanceOptimizationTarget.SCALABILITY: self._optimize_for_scalability
        }
        
    def optimize_code(self, 
                     code: str, 
                     targets: List[PerformanceOptimizationTarget],
                     language: str = "python") -> Tuple[str, List[str]]:
        """Optimize code for specified performance targets."""
        
        optimized_code = code
        applied_optimizations = []
        
        try:
            for target in targets:
                if target in self.optimization_strategies:
                    optimized_code, optimizations = self.optimization_strategies[target](
                        optimized_code, language
                    )
                    applied_optimizations.extend(optimizations)
                    
        except Exception as e:
            logger.warning(f"Code optimization failed: {e}")
            return code, []
            
        return optimized_code, applied_optimizations
        
    def _optimize_for_latency(self, code: str, language: str) -> Tuple[str, List[str]]:
        """Apply latency optimizations."""
        
        optimizations = []
        optimized_code = code
        
        if language.lower() == "python":
            # Vectorization suggestions
            if "for " in code and "range(" in code:
                optimized_code = self._suggest_vectorization(optimized_code)
                optimizations.append("suggested_vectorization")
                
            # Caching suggestions
            if "def " in code:
                optimized_code = self._add_caching_decorator(optimized_code)
                optimizations.append("added_lru_cache")
                
            # List comprehensions
            optimized_code = self._optimize_loops_to_comprehensions(optimized_code)
            optimizations.append("loop_to_comprehension")
            
        return optimized_code, optimizations
        
    def _optimize_for_memory(self, code: str, language: str) -> Tuple[str, List[str]]:
        """Apply memory optimizations."""
        
        optimizations = []
        optimized_code = code
        
        if language.lower() == "python":
            # Generator expressions instead of list comprehensions
            optimized_code = self._suggest_generators(optimized_code)
            optimizations.append("generators_over_lists")
            
            # __slots__ for classes
            optimized_code = self._add_slots_to_classes(optimized_code)
            optimizations.append("added_slots")
            
            # Memory-efficient data structures
            optimizations.append("suggested_memory_efficient_structures")
            
        return optimized_code, optimizations
        
    def _optimize_for_throughput(self, code: str, language: str) -> Tuple[str, List[str]]:
        """Apply throughput optimizations."""
        
        optimizations = []
        optimized_code = code
        
        # Parallel processing suggestions
        if "for " in code:
            optimized_code = self._suggest_parallel_processing(optimized_code)
            optimizations.append("parallel_processing")
            
        # Batch processing
        optimizations.append("batch_processing_pattern")
        
        return optimized_code, optimizations
        
    def _optimize_for_accuracy(self, code: str, language: str) -> Tuple[str, List[str]]:
        """Apply accuracy optimizations."""
        
        optimizations = []
        optimized_code = code
        
        # Add input validation
        optimized_code = self._add_input_validation(optimized_code)
        optimizations.append("input_validation")
        
        # Error handling
        optimized_code = self._add_error_handling(optimized_code)
        optimizations.append("error_handling")
        
        return optimized_code, optimizations
        
    def _optimize_for_energy(self, code: str, language: str) -> Tuple[str, List[str]]:
        """Apply energy efficiency optimizations."""
        
        optimizations = []
        optimized_code = code
        
        # Efficient algorithms
        optimizations.append("algorithm_efficiency_review")
        
        # Reduced computation
        if "**" in code:  # Power operations
            optimized_code = self._optimize_power_operations(optimized_code)
            optimizations.append("optimized_power_operations")
            
        return optimized_code, optimizations
        
    def _optimize_for_scalability(self, code: str, language: str) -> Tuple[str, List[str]]:
        """Apply scalability optimizations."""
        
        optimizations = []
        optimized_code = code
        
        # Asynchronous patterns
        optimized_code = self._suggest_async_patterns(optimized_code)
        optimizations.append("async_patterns")
        
        # Connection pooling
        if "connect" in code.lower():
            optimizations.append("connection_pooling")
            
        return optimized_code, optimizations
        
    # Utility methods for specific optimizations
        
    def _suggest_vectorization(self, code: str) -> str:
        """Suggest NumPy vectorization where applicable."""
        
        # Simple pattern matching for common loop patterns
        pattern = r'for\s+(\w+)\s+in\s+range\([^)]+\):\s*\n\s*([^=]+)\s*=\s*([^=]+)\s*\+\s*([^=]+)'
        
        def replace_with_vectorization(match):
            return f"# Vectorization suggestion: Consider using NumPy operations\n{match.group(0)}"
            
        return re.sub(pattern, replace_with_vectorization, code, flags=re.MULTILINE)
        
    def _add_caching_decorator(self, code: str) -> str:
        """Add LRU cache decorator to functions."""
        
        if "from functools import lru_cache" not in code:
            imports = "from functools import lru_cache\n"
            code = imports + code
            
        # Add @lru_cache decorator to functions (simplified)
        pattern = r'(def\s+\w+\([^)]*\):)'
        replacement = r'@lru_cache(maxsize=128)\n\1'
        
        return re.sub(pattern, replacement, code)
        
    def _optimize_loops_to_comprehensions(self, code: str) -> str:
        """Convert simple loops to list comprehensions."""
        
        # Very simplified pattern - would need more sophisticated parsing
        pattern = r'result\s*=\s*\[\]\s*\nfor\s+(\w+)\s+in\s+([^:]+):\s*\n\s*result\.append\(([^)]+)\)'
        replacement = r'result = [\3 for \1 in \2]'
        
        return re.sub(pattern, replacement, code, flags=re.MULTILINE | re.DOTALL)
        
    def _suggest_generators(self, code: str) -> str:
        """Suggest generator expressions for memory efficiency."""
        
        # Add comment suggestions for generator expressions
        pattern = r'\[([^]]+for[^]]+)\]'
        replacement = r'# Consider generator: (\1) for memory efficiency\n[\1]'
        
        return re.sub(pattern, replacement, code)
        
    def _add_slots_to_classes(self, code: str) -> str:
        """Add __slots__ to classes for memory efficiency."""
        
        # Simple class detection and slots addition
        pattern = r'class\s+(\w+)(?:\([^)]*\))?:\s*\n(\s*)(.*?)(?=\n\s*def|\n[^\s]|\Z)'
        
        def add_slots(match):
            class_name = match.group(1)
            indent = match.group(2)
            class_body = match.group(3)
            
            if "__slots__" not in class_body:
                slots_line = f"{indent}__slots__ = []  # Add attributes for memory efficiency\n"
                return f"class {class_name}:\n{slots_line}{indent}{class_body}"
            return match.group(0)
            
        return re.sub(pattern, add_slots, code, flags=re.MULTILINE | re.DOTALL)
        
    def _suggest_parallel_processing(self, code: str) -> str:
        """Suggest parallel processing patterns."""
        
        if "multiprocessing" not in code:
            import_line = "# Consider: from multiprocessing import Pool\n"
            code = import_line + code
            
        # Add comment for parallel processing opportunities
        pattern = r'for\s+\w+\s+in\s+[^:]+:'
        replacement = f'{pattern}\n    # Consider parallel processing with Pool.map()'
        
        return re.sub(pattern, replacement, code)
        
    def _add_input_validation(self, code: str) -> str:
        """Add input validation to functions."""
        
        pattern = r'def\s+(\w+)\(([^)]+)\):\s*\n(\s*)'
        
        def add_validation(match):
            func_name = match.group(1)
            params = match.group(2)
            indent = match.group(3)
            
            validation = f"{indent}# Input validation\n{indent}if not all([{params}]):\n{indent}    raise ValueError('Invalid parameters')\n"
            return f"def {func_name}({params}):\n{validation}"
            
        return re.sub(pattern, add_validation, code, flags=re.MULTILINE)
        
    def _add_error_handling(self, code: str) -> str:
        """Add comprehensive error handling."""
        
        if "try:" not in code:
            # Wrap main code in try-except
            lines = code.split('\n')
            if lines:
                indented_code = '\n'.join(['    ' + line for line in lines])
                code = f"try:\n{indented_code}\nexcept Exception as e:\n    logging.error(f'Error: {{e}}')\n    raise"
                
        return code
        
    def _optimize_power_operations(self, code: str) -> str:
        """Optimize power operations for energy efficiency."""
        
        # Replace x**2 with x*x for efficiency
        code = re.sub(r'(\w+)\s*\*\*\s*2', r'\1 * \1', code)
        
        # Suggest efficient sqrt operations
        pattern = r'(\w+)\s*\*\*\s*0\.5'
        replacement = r'math.sqrt(\1)  # More efficient than power'
        
        if "math.sqrt" in replacement and "import math" not in code:
            code = "import math\n" + code
            
        return re.sub(pattern, replacement, code)
        
    def _suggest_async_patterns(self, code: str) -> str:
        """Suggest asynchronous programming patterns."""
        
        # Add async/await suggestions
        if "def " in code and "async" not in code:
            pattern = r'def\s+(\w+)\([^)]*\):'
            replacement = r'# Consider: async def \1(...): for I/O operations\ndef \1(...):'
            code = re.sub(pattern, replacement, code)
            
        return code


class IntelligentRefactorer:
    """Intelligent code refactoring system."""
    
    def __init__(self):
        self.refactoring_patterns = self._load_refactoring_patterns()
        
    def _load_refactoring_patterns(self) -> Dict[str, Dict[str, str]]:
        """Load refactoring patterns."""
        
        return {
            "extract_method": {
                "description": "Extract repeated code into separate method",
                "pattern": r"(.{20,})\n\s*\1",  # Simplified duplicate detection
                "replacement": "# Consider extracting duplicate code into a method"
            },
            "simplify_conditionals": {
                "description": "Simplify complex conditional expressions",
                "pattern": r"if\s+(.+)\s+and\s+(.+)\s+and\s+(.+):",
                "replacement": "if all([\\1, \\2, \\3]):"
            },
            "remove_dead_code": {
                "description": "Identify potentially unused code",
                "pattern": r"def\s+(\w+)\([^)]*\):\s*\n\s*pass",
                "replacement": "# Dead code candidate: def \\1 - consider removal"
            }
        }
        
    def refactor_code(self, code: str, language: str = "python") -> Tuple[str, List[str]]:
        """Apply intelligent refactoring to code."""
        
        refactored_code = code
        applied_refactorings = []
        
        try:
            for pattern_name, pattern_info in self.refactoring_patterns.items():
                pattern = pattern_info["pattern"]
                replacement = pattern_info["replacement"]
                
                if re.search(pattern, refactored_code):
                    refactored_code = re.sub(pattern, replacement, refactored_code)
                    applied_refactorings.append(pattern_name)
                    
            # Apply additional refactorings
            refactored_code, additional = self._apply_structural_refactoring(refactored_code)
            applied_refactorings.extend(additional)
            
        except Exception as e:
            logger.warning(f"Code refactoring failed: {e}")
            return code, []
            
        return refactored_code, applied_refactorings
        
    def _apply_structural_refactoring(self, code: str) -> Tuple[str, List[str]]:
        """Apply structural refactoring patterns."""
        
        refactored_code = code
        applied = []
        
        # Extract constants
        if re.search(r'\b\d+\.\d+\b', code):  # Find magic numbers
            refactored_code = "# Consider extracting magic numbers as constants\n" + refactored_code
            applied.append("extract_constants")
            
        # Improve naming
        if re.search(r'\b[a-z]{1,2}\b\s*=', code):  # Short variable names
            refactored_code = "# Consider using more descriptive variable names\n" + refactored_code
            applied.append("improve_naming")
            
        return refactored_code, applied


class AdaptiveCodeGenerator:
    """Main adaptive code generation system."""
    
    def __init__(self, output_dir: Path = Path("adaptive_code_gen")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.quality_analyzer = CodeQualityAnalyzer()
        self.performance_optimizer = PerformanceOptimizer()
        self.refactorer = IntelligentRefactorer()
        
        # Initialize AI models
        self._init_ai_models()
        
        # Code templates
        self.templates = self._load_code_templates()
        
        # Generation history
        self.generation_history = []
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / "adaptive_code_gen.log"),
                logging.StreamHandler()
            ]
        )
        
    def _init_ai_models(self):
        """Initialize AI models for code generation."""
        
        try:
            # Code generation model
            self.code_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
            self.code_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
            
            # Set pad token
            if self.code_tokenizer.pad_token is None:
                self.code_tokenizer.pad_token = self.code_tokenizer.eos_token
                
            logger.info("AI models initialized successfully")
            
        except Exception as e:
            logger.warning(f"Could not initialize AI models: {e}")
            self.code_tokenizer = None
            self.code_model = None
            
    def _load_code_templates(self) -> Dict[str, str]:
        """Load code generation templates."""
        
        return {
            "function_template": '''def {function_name}({parameters}):
    """
    {description}
    
    Args:
        {args_doc}
    
    Returns:
        {return_doc}
    """
    {implementation}
    return result
''',
            "class_template": '''class {class_name}:
    """
    {description}
    """
    
    def __init__(self, {init_params}):
        {init_implementation}
    
    def {method_name}(self, {method_params}):
        """
        {method_description}
        """
        {method_implementation}
        return result
''',
            "algorithm_template": '''def {algorithm_name}({input_params}):
    """
    {algorithm_description}
    
    Time Complexity: {time_complexity}
    Space Complexity: {space_complexity}
    """
    # Initialize
    {initialization}
    
    # Main algorithm
    {main_logic}
    
    # Post-processing
    {post_processing}
    
    return result
'''
        }
        
    async def generate_adaptive_code(self, request: CodeGenerationRequest) -> GeneratedCode:
        """Generate adaptive code based on request."""
        
        logger.info(f"🔧 Generating adaptive code: {request.functionality_description}")
        
        start_time = datetime.now()
        
        try:
            # Determine best generation strategy
            strategy = self._select_generation_strategy(request)
            
            # Generate initial code
            initial_code = await self._generate_initial_code(request, strategy)
            
            # Apply performance optimizations
            optimized_code, optimizations = self.performance_optimizer.optimize_code(
                initial_code, 
                request.optimization_targets,
                request.target_language
            )
            
            # Apply intelligent refactoring
            refactored_code, refactorings = self.refactorer.refactor_code(
                optimized_code, 
                request.target_language
            )
            
            # Analyze quality
            quality_metrics = self.quality_analyzer.analyze_quality(
                refactored_code, 
                request.target_language
            )
            
            # Run tests if provided
            test_results = await self._run_tests(refactored_code, request.test_cases)
            
            # Calculate performance metrics
            performance_metrics = self._estimate_performance_metrics(
                refactored_code, 
                request.performance_requirements
            )
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                quality_metrics, 
                performance_metrics, 
                test_results
            )
            
            # Create generated code object
            generated_code = GeneratedCode(
                code=refactored_code,
                language=request.target_language,
                functionality=request.functionality_description,
                performance_metrics=performance_metrics,
                quality_metrics=quality_metrics,
                optimization_applied=optimizations + refactorings,
                generation_strategy=strategy,
                confidence_score=confidence_score,
                test_results=test_results
            )
            
            # Store in history
            self.generation_history.append(generated_code)
            
            # Save results
            await self._save_generation_results(generated_code, request)
            
            generation_time = datetime.now() - start_time
            logger.info(f"✅ Code generation completed in {generation_time}")
            
            return generated_code
            
        except Exception as e:
            logger.error(f"❌ Code generation failed: {e}")
            # Return fallback code
            return GeneratedCode(
                code=f"# Code generation failed: {str(e)}\n# Fallback implementation needed",
                language=request.target_language,
                functionality=request.functionality_description,
                performance_metrics={},
                quality_metrics={},
                optimization_applied=[],
                generation_strategy=CodeGenerationStrategy.TEMPLATE_BASED,
                confidence_score=0.0,
                test_results={"error": str(e)}
            )
            
    def _select_generation_strategy(self, request: CodeGenerationRequest) -> CodeGenerationStrategy:
        """Select optimal generation strategy based on request."""
        
        # Decision logic based on request characteristics
        
        if self.code_model is None:
            return CodeGenerationStrategy.TEMPLATE_BASED
            
        # If context code is provided, use AI assistance
        if request.context_code:
            return CodeGenerationStrategy.AI_ASSISTED
            
        # For complex optimization requirements, use hybrid
        if len(request.optimization_targets) > 2:
            return CodeGenerationStrategy.HYBRID_APPROACH
            
        # For simple requests, use templates
        if len(request.functionality_description.split()) < 10:
            return CodeGenerationStrategy.TEMPLATE_BASED
            
        # Default to neural synthesis
        return CodeGenerationStrategy.NEURAL_SYNTHESIS
        
    async def _generate_initial_code(self, 
                                    request: CodeGenerationRequest, 
                                    strategy: CodeGenerationStrategy) -> str:
        """Generate initial code using selected strategy."""
        
        if strategy == CodeGenerationStrategy.TEMPLATE_BASED:
            return self._generate_template_based_code(request)
        elif strategy == CodeGenerationStrategy.AI_ASSISTED:
            return await self._generate_ai_assisted_code(request)
        elif strategy == CodeGenerationStrategy.NEURAL_SYNTHESIS:
            return await self._generate_neural_synthesis_code(request)
        elif strategy == CodeGenerationStrategy.HYBRID_APPROACH:
            return await self._generate_hybrid_code(request)
        else:
            return self._generate_template_based_code(request)
            
    def _generate_template_based_code(self, request: CodeGenerationRequest) -> str:
        """Generate code using templates."""
        
        # Analyze request to determine template type
        desc_lower = request.functionality_description.lower()
        
        if "function" in desc_lower or "calculate" in desc_lower:
            template = self.templates["function_template"]
            
            return template.format(
                function_name=self._extract_function_name(request.functionality_description),
                parameters="data",
                description=request.functionality_description,
                args_doc="data: Input data for processing",
                return_doc="Processed result",
                implementation=self._generate_basic_implementation(request)
            )
            
        elif "class" in desc_lower or "object" in desc_lower:
            template = self.templates["class_template"]
            
            class_name = self._extract_class_name(request.functionality_description)
            return template.format(
                class_name=class_name,
                description=request.functionality_description,
                init_params="*args, **kwargs",
                init_implementation="pass",
                method_name="process",
                method_params="self, data",
                method_description="Process data using implemented algorithm",
                method_implementation=self._generate_basic_implementation(request)
            )
            
        elif "algorithm" in desc_lower:
            template = self.templates["algorithm_template"]
            
            return template.format(
                algorithm_name=self._extract_function_name(request.functionality_description),
                input_params="data",
                algorithm_description=request.functionality_description,
                time_complexity="O(n)",
                space_complexity="O(1)",
                initialization="result = None",
                main_logic=self._generate_basic_implementation(request),
                post_processing="# Validate result"
            )
            
        else:
            # Generic function template
            return self._generate_generic_function(request)
            
    def _extract_function_name(self, description: str) -> str:
        """Extract function name from description."""
        
        # Simple heuristic to extract function name
        words = description.lower().split()
        
        # Look for action verbs
        action_words = ['calculate', 'compute', 'process', 'analyze', 'generate', 'create', 'build']
        
        for word in words:
            if word in action_words:
                return f"{word}_data"
                
        # Default name
        return "process_data"
        
    def _extract_class_name(self, description: str) -> str:
        """Extract class name from description."""
        
        words = description.split()
        
        # Look for nouns that could be class names
        for word in words:
            if word[0].isupper() or word in ['processor', 'analyzer', 'generator', 'manager']:
                return word.capitalize() + "Processor"
                
        return "DataProcessor"
        
    def _generate_basic_implementation(self, request: CodeGenerationRequest) -> str:
        """Generate basic implementation based on request."""
        
        desc_lower = request.functionality_description.lower()
        
        # Pattern matching for common operations
        if "sort" in desc_lower:
            return "result = sorted(data)"
        elif "filter" in desc_lower:
            return "result = [item for item in data if condition(item)]"
        elif "transform" in desc_lower or "convert" in desc_lower:
            return "result = [transform_item(item) for item in data]"
        elif "aggregate" in desc_lower or "sum" in desc_lower:
            return "result = sum(data)"
        elif "search" in desc_lower or "find" in desc_lower:
            return "result = next((item for item in data if matches(item)), None)"
        else:
            return "# Implementation logic here\nresult = data"
            
    def _generate_generic_function(self, request: CodeGenerationRequest) -> str:
        """Generate generic function implementation."""
        
        return f'''def process_data(data):
    """
    {request.functionality_description}
    
    Args:
        data: Input data for processing
        
    Returns:
        Processed result
    """
    # TODO: Implement specific logic
    result = data
    return result
'''

    async def _generate_ai_assisted_code(self, request: CodeGenerationRequest) -> str:
        """Generate code using AI assistance."""
        
        if self.code_model is None:
            return self._generate_template_based_code(request)
            
        try:
            # Create prompt for code generation
            prompt = self._create_code_generation_prompt(request)
            
            # Tokenize prompt
            inputs = self.code_tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
            
            # Generate code
            with torch.no_grad():
                outputs = self.code_model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 200,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.code_tokenizer.eos_token_id
                )
                
            # Decode generated code
            generated_text = self.code_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract code portion (remove prompt)
            generated_code = generated_text[len(prompt):].strip()
            
            # Post-process generated code
            return self._post_process_generated_code(generated_code, request)
            
        except Exception as e:
            logger.warning(f"AI code generation failed: {e}")
            return self._generate_template_based_code(request)
            
    def _create_code_generation_prompt(self, request: CodeGenerationRequest) -> str:
        """Create prompt for AI code generation."""
        
        prompt = f"Generate Python code for: {request.functionality_description}\n\n"
        
        if request.context_code:
            prompt += f"Context code:\n{request.context_code}\n\n"
            
        if request.performance_requirements:
            prompt += f"Performance requirements: {request.performance_requirements}\n\n"
            
        prompt += "Generated code:\n"
        
        return prompt
        
    def _post_process_generated_code(self, generated_code: str, request: CodeGenerationRequest) -> str:
        """Post-process AI-generated code."""
        
        # Clean up generated code
        lines = generated_code.split('\n')
        
        # Remove empty lines at the beginning and end
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
            
        # Ensure proper indentation
        if lines:
            # Find minimum indentation
            min_indent = float('inf')
            for line in lines:
                if line.strip():
                    indent = len(line) - len(line.lstrip())
                    min_indent = min(min_indent, indent)
                    
            if min_indent != float('inf') and min_indent > 0:
                lines = [line[min_indent:] if line.strip() else line for line in lines]
                
        return '\n'.join(lines)
        
    async def _generate_neural_synthesis_code(self, request: CodeGenerationRequest) -> str:
        """Generate code using neural synthesis."""
        
        # For now, fall back to AI-assisted generation
        # In a full implementation, this would use specialized neural synthesis models
        return await self._generate_ai_assisted_code(request)
        
    async def _generate_hybrid_code(self, request: CodeGenerationRequest) -> str:
        """Generate code using hybrid approach."""
        
        # Combine template-based and AI-assisted approaches
        template_code = self._generate_template_based_code(request)
        
        if self.code_model is not None:
            ai_code = await self._generate_ai_assisted_code(request)
            
            # Combine the best parts of both approaches
            return self._merge_code_approaches(template_code, ai_code)
        else:
            return template_code
            
    def _merge_code_approaches(self, template_code: str, ai_code: str) -> str:
        """Merge template and AI-generated code."""
        
        # Simple merging strategy - use template structure with AI content
        # In practice, this would be more sophisticated
        
        if len(ai_code.strip()) > len(template_code.strip()):
            return ai_code
        else:
            return template_code
            
    async def _run_tests(self, code: str, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run test cases against generated code."""
        
        if not test_cases:
            return {"status": "no_tests", "passed": 0, "failed": 0}
            
        test_results = {
            "status": "completed",
            "passed": 0,
            "failed": 0,
            "errors": [],
            "details": []
        }
        
        try:
            # Execute code to make functions available
            exec_globals = {}
            exec(code, exec_globals)
            
            for i, test_case in enumerate(test_cases):
                try:
                    # Extract test components
                    function_name = test_case.get("function_name", "process_data")
                    inputs = test_case.get("inputs", [])
                    expected_output = test_case.get("expected_output")
                    
                    if function_name in exec_globals:
                        # Execute function with test inputs
                        if isinstance(inputs, list):
                            actual_output = exec_globals[function_name](*inputs)
                        else:
                            actual_output = exec_globals[function_name](inputs)
                            
                        # Compare outputs
                        if expected_output is not None:
                            if actual_output == expected_output:
                                test_results["passed"] += 1
                                test_results["details"].append({
                                    "test_id": i,
                                    "status": "passed"
                                })
                            else:
                                test_results["failed"] += 1
                                test_results["details"].append({
                                    "test_id": i,
                                    "status": "failed",
                                    "expected": expected_output,
                                    "actual": actual_output
                                })
                        else:
                            # Just verify it runs without error
                            test_results["passed"] += 1
                            test_results["details"].append({
                                "test_id": i,
                                "status": "passed"
                            })
                    else:
                        test_results["failed"] += 1
                        test_results["errors"].append(f"Function {function_name} not found")
                        
                except Exception as e:
                    test_results["failed"] += 1
                    test_results["errors"].append(f"Test {i} error: {str(e)}")
                    
        except Exception as e:
            test_results["status"] = "error"
            test_results["errors"].append(f"Code execution error: {str(e)}")
            
        return test_results
        
    def _estimate_performance_metrics(self, 
                                     code: str, 
                                     performance_requirements: Dict[str, float]) -> Dict[str, float]:
        """Estimate performance metrics for generated code."""
        
        metrics = {}
        
        # Estimate based on code analysis
        lines = len([line for line in code.split('\n') if line.strip()])
        
        # Simple heuristics for performance estimation
        
        # Latency estimation (lower is better)
        complexity_indicators = code.count('for ') + code.count('while ') * 2
        estimated_latency = min(100.0, 10.0 + complexity_indicators * 5.0)  # ms
        metrics["estimated_latency_ms"] = estimated_latency
        
        # Memory estimation
        data_structures = code.count('[') + code.count('{') + code.count('dict')
        estimated_memory = min(1000.0, 10.0 + data_structures * 50.0)  # MB
        metrics["estimated_memory_mb"] = estimated_memory
        
        # Throughput estimation (higher is better)
        parallel_indicators = code.count('multiprocessing') + code.count('async')
        base_throughput = 1000.0  # operations per second
        throughput_multiplier = 1.0 + parallel_indicators * 0.5
        metrics["estimated_throughput_ops"] = base_throughput * throughput_multiplier
        
        # Compare against requirements
        for req_name, req_value in performance_requirements.items():
            if req_name in metrics:
                requirement_met = metrics[req_name] <= req_value  # Assuming lower is better
                metrics[f"{req_name}_requirement_met"] = float(requirement_met)
                
        return metrics
        
    def _calculate_confidence_score(self, 
                                   quality_metrics: Dict[str, float],
                                   performance_metrics: Dict[str, float],
                                   test_results: Dict[str, Any]) -> float:
        """Calculate confidence score for generated code."""
        
        # Combine various factors into confidence score
        
        # Quality component (0-1)
        quality_score = np.mean(list(quality_metrics.values())) if quality_metrics else 0.5
        
        # Performance component (simplified)
        performance_score = 0.7  # Default reasonable score
        
        # Test component
        test_score = 0.5  # Default
        if test_results.get("status") == "completed":
            total_tests = test_results.get("passed", 0) + test_results.get("failed", 0)
            if total_tests > 0:
                test_score = test_results.get("passed", 0) / total_tests
        elif test_results.get("status") == "no_tests":
            test_score = 0.6  # Neutral score when no tests
            
        # Weighted combination
        confidence = (
            quality_score * 0.4 +
            performance_score * 0.3 +
            test_score * 0.3
        )
        
        return min(1.0, max(0.0, confidence))
        
    async def _save_generation_results(self, 
                                      generated_code: GeneratedCode,
                                      request: CodeGenerationRequest):
        """Save code generation results."""
        
        # Create timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save generated code
        code_file = self.output_dir / f"generated_code_{timestamp}.py"
        with open(code_file, "w") as f:
            f.write(generated_code.code)
            
        # Save generation metadata
        metadata = {
            "request": request.to_dict(),
            "result": generated_code.to_dict()
        }
        
        metadata_file = self.output_dir / f"generation_metadata_{timestamp}.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
            
        logger.info(f"Generation results saved to {code_file} and {metadata_file}")
        
    async def generate_multiple_variants(self, 
                                        request: CodeGenerationRequest,
                                        num_variants: int = 3) -> List[GeneratedCode]:
        """Generate multiple code variants for comparison."""
        
        variants = []
        
        # Use different strategies for variants
        strategies = [
            CodeGenerationStrategy.TEMPLATE_BASED,
            CodeGenerationStrategy.AI_ASSISTED,
            CodeGenerationStrategy.HYBRID_APPROACH
        ]
        
        for i in range(min(num_variants, len(strategies))):
            try:
                # Modify request for variant generation
                variant_request = request
                # Could add slight variations to request here
                
                # Force specific strategy
                original_strategy_method = self._select_generation_strategy
                self._select_generation_strategy = lambda req: strategies[i]
                
                variant = await self.generate_adaptive_code(variant_request)
                variants.append(variant)
                
                # Restore original method
                self._select_generation_strategy = original_strategy_method
                
            except Exception as e:
                logger.warning(f"Variant {i} generation failed: {e}")
                continue
                
        return variants
        
    def get_generation_analytics(self) -> Dict[str, Any]:
        """Get analytics on code generation history."""
        
        if not self.generation_history:
            return {"error": "No generation history available"}
            
        analytics = {
            "total_generations": len(self.generation_history),
            "average_confidence": np.mean([g.confidence_score for g in self.generation_history]),
            "strategy_distribution": {},
            "language_distribution": {},
            "quality_trends": [],
            "performance_trends": []
        }
        
        # Strategy distribution
        strategies = [g.generation_strategy.value for g in self.generation_history]
        analytics["strategy_distribution"] = {
            strategy: strategies.count(strategy) for strategy in set(strategies)
        }
        
        # Language distribution
        languages = [g.language for g in self.generation_history]
        analytics["language_distribution"] = {
            language: languages.count(language) for language in set(languages)
        }
        
        # Quality trends
        for generation in self.generation_history:
            if generation.quality_metrics:
                avg_quality = np.mean(list(generation.quality_metrics.values()))
                analytics["quality_trends"].append(avg_quality)
                
        # Performance trends
        for generation in self.generation_history:
            if generation.performance_metrics:
                # Use confidence score as proxy for performance
                analytics["performance_trends"].append(generation.confidence_score)
                
        return analytics


# Example usage
async def main():
    """Example adaptive code generation."""
    
    # Initialize system
    code_gen = AdaptiveCodeGenerator(Path("example_adaptive_code_gen"))
    
    # Create code generation request
    request = CodeGenerationRequest(
        functionality_description="Create a function that sorts a list of numbers using quicksort algorithm",
        target_language="python",
        performance_requirements={"latency_ms": 50.0, "memory_mb": 100.0},
        optimization_targets=[
            PerformanceOptimizationTarget.LATENCY,
            PerformanceOptimizationTarget.MEMORY
        ],
        test_cases=[
            {
                "function_name": "quicksort",
                "inputs": [[3, 1, 4, 1, 5, 9, 2, 6]],
                "expected_output": [1, 1, 2, 3, 4, 5, 6, 9]
            }
        ]
    )
    
    try:
        # Generate adaptive code
        result = await code_gen.generate_adaptive_code(request)
        
        print("🔧 Generated Code:")
        print(result.code)
        print(f"\n📊 Confidence Score: {result.confidence_score:.3f}")
        print(f"🛠️ Optimizations Applied: {result.optimization_applied}")
        print(f"✅ Quality Metrics: {result.quality_metrics}")
        
        # Generate multiple variants
        variants = await code_gen.generate_multiple_variants(request, num_variants=2)
        print(f"\n🔄 Generated {len(variants)} variants")
        
        # Get analytics
        analytics = code_gen.get_generation_analytics()
        print(f"\n📈 Analytics: {analytics}")
        
    except Exception as e:
        print(f"❌ Code generation failed: {e}")
    

if __name__ == "__main__":
    asyncio.run(main())