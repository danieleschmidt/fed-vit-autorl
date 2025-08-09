"""Input validation for federated learning components."""

import re
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False
    np = None
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of input validation."""
    is_valid: bool
    message: str
    sanitized_input: Any = None
    warnings: List[str] = None


class InputValidator:
    """Base input validation for ML components."""
    
    def __init__(self, strict_mode: bool = True):
        """Initialize validator.
        
        Args:
            strict_mode: Whether to be strict about validation
        """
        self.strict_mode = strict_mode
        self.validation_stats = {
            "total_validations": 0,
            "failed_validations": 0,
            "sanitizations": 0,
        }
        
        logger.info(f"Initialized input validator (strict_mode={strict_mode})")
    
    def validate_tensor_input(
        self,
        tensor,
        expected_shape: Optional[Tuple[int, ...]] = None,
        expected_dtype = None,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        name: str = "tensor",
    ) -> ValidationResult:
        """Validate tensor inputs."""
        self.validation_stats["total_validations"] += 1
        warnings = []
        
        try:
            # Check if torch is available
            if not _TORCH_AVAILABLE:
                return ValidationResult(
                    is_valid=False,
                    message=f"{name} validation requires PyTorch"
                )
            
            # Check if input is tensor
            if not isinstance(tensor, torch.Tensor):
                self.validation_stats["failed_validations"] += 1
                return ValidationResult(
                    is_valid=False,
                    message=f"{name} must be a torch.Tensor, got {type(tensor)}"
                )
            
            # Check for NaN or infinite values
            if torch.isnan(tensor).any():
                if self.strict_mode:
                    self.validation_stats["failed_validations"] += 1
                    return ValidationResult(
                        is_valid=False,
                        message=f"{name} contains NaN values"
                    )
                else:
                    warnings.append(f"{name} contains NaN values")
                    # Replace NaN with zeros
                    tensor = torch.nan_to_num(tensor, nan=0.0)
                    self.validation_stats["sanitizations"] += 1
            
            if torch.isinf(tensor).any():
                if self.strict_mode:
                    self.validation_stats["failed_validations"] += 1
                    return ValidationResult(
                        is_valid=False,
                        message=f"{name} contains infinite values"
                    )
                else:
                    warnings.append(f"{name} contains infinite values")
                    # Replace infinite values
                    tensor = torch.nan_to_num(tensor, posinf=1e6, neginf=-1e6)
                    self.validation_stats["sanitizations"] += 1
            
            # Check expected shape
            if expected_shape is not None:
                if tensor.shape != expected_shape:
                    # Allow flexible batch dimension (first dimension)
                    if len(expected_shape) > 1 and tensor.shape[1:] == expected_shape[1:]:
                        warnings.append(f"{name} batch size differs: {tensor.shape[0]} vs expected pattern")
                    else:
                        self.validation_stats["failed_validations"] += 1
                        return ValidationResult(
                            is_valid=False,
                            message=f"{name} shape mismatch: {tensor.shape} vs expected {expected_shape}"
                        )
            
            # Check expected dtype
            if expected_dtype is not None and tensor.dtype != expected_dtype:
                if self.strict_mode:
                    self.validation_stats["failed_validations"] += 1
                    return ValidationResult(
                        is_valid=False,
                        message=f"{name} dtype mismatch: {tensor.dtype} vs expected {expected_dtype}"
                    )
                else:
                    warnings.append(f"{name} dtype converted: {tensor.dtype} -> {expected_dtype}")
                    tensor = tensor.to(expected_dtype)
                    self.validation_stats["sanitizations"] += 1
            
            # Check value ranges
            if min_val is not None and tensor.min() < min_val:
                if self.strict_mode:
                    self.validation_stats["failed_validations"] += 1
                    return ValidationResult(
                        is_valid=False,
                        message=f"{name} contains values below minimum {min_val}: {tensor.min()}"
                    )
                else:
                    warnings.append(f"{name} clamped to minimum {min_val}")
                    tensor = torch.clamp(tensor, min=min_val)
                    self.validation_stats["sanitizations"] += 1
            
            if max_val is not None and tensor.max() > max_val:
                if self.strict_mode:
                    self.validation_stats["failed_validations"] += 1
                    return ValidationResult(
                        is_valid=False,
                        message=f"{name} contains values above maximum {max_val}: {tensor.max()}"
                    )
                else:
                    warnings.append(f"{name} clamped to maximum {max_val}")
                    tensor = torch.clamp(tensor, max=max_val)
                    self.validation_stats["sanitizations"] += 1
            
            return ValidationResult(
                is_valid=True,
                message=f"{name} validation passed",
                sanitized_input=tensor,
                warnings=warnings
            )
            
        except Exception as e:
            self.validation_stats["failed_validations"] += 1
            logger.error(f"Tensor validation error: {e}")
            return ValidationResult(
                is_valid=False,
                message=f"{name} validation failed: {str(e)}"
            )
    
    def validate_string_input(
        self,
        text: str,
        max_length: int = 1000,
        allowed_pattern: Optional[str] = None,
        forbidden_patterns: Optional[List[str]] = None,
        name: str = "string",
    ) -> ValidationResult:
        """Validate string inputs."""
        self.validation_stats["total_validations"] += 1
        warnings = []
        
        try:
            # Check if input is string
            if not isinstance(text, str):
                self.validation_stats["failed_validations"] += 1
                return ValidationResult(
                    is_valid=False,
                    message=f"{name} must be a string, got {type(text)}"
                )
            
            # Check length
            if len(text) > max_length:
                if self.strict_mode:
                    self.validation_stats["failed_validations"] += 1
                    return ValidationResult(
                        is_valid=False,
                        message=f"{name} exceeds maximum length {max_length}: {len(text)}"
                    )
                else:
                    warnings.append(f"{name} truncated to {max_length} characters")
                    text = text[:max_length]
                    self.validation_stats["sanitizations"] += 1
            
            # Check allowed pattern
            if allowed_pattern is not None:
                if not re.match(allowed_pattern, text):
                    self.validation_stats["failed_validations"] += 1
                    return ValidationResult(
                        is_valid=False,
                        message=f"{name} does not match allowed pattern: {allowed_pattern}"
                    )
            
            # Check forbidden patterns
            if forbidden_patterns is not None:
                for pattern in forbidden_patterns:
                    if re.search(pattern, text, re.IGNORECASE):
                        self.validation_stats["failed_validations"] += 1
                        return ValidationResult(
                            is_valid=False,
                            message=f"{name} contains forbidden pattern: {pattern}"
                        )
            
            # Basic sanitization - remove control characters
            sanitized_text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
            if sanitized_text != text:
                warnings.append(f"{name} control characters removed")
                text = sanitized_text
                self.validation_stats["sanitizations"] += 1
            
            return ValidationResult(
                is_valid=True,
                message=f"{name} validation passed",
                sanitized_input=text,
                warnings=warnings
            )
            
        except Exception as e:
            self.validation_stats["failed_validations"] += 1
            logger.error(f"String validation error: {e}")
            return ValidationResult(
                is_valid=False,
                message=f"{name} validation failed: {str(e)}"
            )
    
    def validate_numeric_input(
        self,
        value: Union[int, float],
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        integer_only: bool = False,
        name: str = "number",
    ) -> ValidationResult:
        """Validate numeric inputs."""
        self.validation_stats["total_validations"] += 1
        warnings = []
        
        try:
            # Define numeric types based on available packages
            numeric_types = [int, float]
            if _NUMPY_AVAILABLE and np is not None:
                numeric_types.extend([np.integer, np.floating])
            
            # Check if input is numeric
            if not isinstance(value, tuple(numeric_types)):
                self.validation_stats["failed_validations"] += 1
                return ValidationResult(
                    is_valid=False,
                    message=f"{name} must be numeric, got {type(value)}"
                )
            
            # Convert numpy types to Python types if numpy is available
            if _NUMPY_AVAILABLE and np is not None:
                if isinstance(value, np.integer):
                    value = int(value)
                elif isinstance(value, np.floating):
                    value = float(value)
            
            # Check for NaN or infinite
            if isinstance(value, float):
                if _NUMPY_AVAILABLE and np is not None:
                    is_nan = np.isnan(value)
                    is_inf = np.isinf(value)
                else:
                    # Fallback for when numpy is not available
                    is_nan = str(value).lower() == 'nan'
                    is_inf = str(value).lower() in ['inf', '-inf']
                
                if is_nan:
                    self.validation_stats["failed_validations"] += 1
                    return ValidationResult(
                        is_valid=False,
                        message=f"{name} is NaN"
                    )
                
                if is_inf:
                    self.validation_stats["failed_validations"] += 1
                    return ValidationResult(
                        is_valid=False,
                        message=f"{name} is infinite"
                    )
            
            # Check integer requirement
            if integer_only and not isinstance(value, int):
                if self.strict_mode:
                    self.validation_stats["failed_validations"] += 1
                    return ValidationResult(
                        is_valid=False,
                        message=f"{name} must be integer, got {type(value)}"
                    )
                else:
                    warnings.append(f"{name} converted to integer")
                    value = int(value)
                    self.validation_stats["sanitizations"] += 1
            
            # Check value ranges
            if min_val is not None and value < min_val:
                if self.strict_mode:
                    self.validation_stats["failed_validations"] += 1
                    return ValidationResult(
                        is_valid=False,
                        message=f"{name} below minimum {min_val}: {value}"
                    )
                else:
                    warnings.append(f"{name} clamped to minimum {min_val}")
                    value = min_val
                    self.validation_stats["sanitizations"] += 1
            
            if max_val is not None and value > max_val:
                if self.strict_mode:
                    self.validation_stats["failed_validations"] += 1
                    return ValidationResult(
                        is_valid=False,
                        message=f"{name} above maximum {max_val}: {value}"
                    )
                else:
                    warnings.append(f"{name} clamped to maximum {max_val}")
                    value = max_val
                    self.validation_stats["sanitizations"] += 1
            
            return ValidationResult(
                is_valid=True,
                message=f"{name} validation passed",
                sanitized_input=value,
                warnings=warnings
            )
            
        except Exception as e:
            self.validation_stats["failed_validations"] += 1
            logger.error(f"Numeric validation error: {e}")
            return ValidationResult(
                is_valid=False,
                message=f"{name} validation failed: {str(e)}"
            )
    
    def validate_dict_input(
        self,
        data: Dict[str, Any],
        required_keys: Optional[List[str]] = None,
        optional_keys: Optional[List[str]] = None,
        max_keys: int = 100,
        name: str = "dictionary",
    ) -> ValidationResult:
        """Validate dictionary inputs."""
        self.validation_stats["total_validations"] += 1
        warnings = []
        
        try:
            # Check if input is dictionary
            if not isinstance(data, dict):
                self.validation_stats["failed_validations"] += 1
                return ValidationResult(
                    is_valid=False,
                    message=f"{name} must be a dictionary, got {type(data)}"
                )
            
            # Check number of keys
            if len(data) > max_keys:
                self.validation_stats["failed_validations"] += 1
                return ValidationResult(
                    is_valid=False,
                    message=f"{name} has too many keys: {len(data)} > {max_keys}"
                )
            
            # Check required keys
            if required_keys is not None:
                missing_keys = set(required_keys) - set(data.keys())
                if missing_keys:
                    self.validation_stats["failed_validations"] += 1
                    return ValidationResult(
                        is_valid=False,
                        message=f"{name} missing required keys: {missing_keys}"
                    )
            
            # Check for unexpected keys
            if required_keys is not None and optional_keys is not None:
                allowed_keys = set(required_keys) | set(optional_keys)
                unexpected_keys = set(data.keys()) - allowed_keys
                
                if unexpected_keys:
                    if self.strict_mode:
                        self.validation_stats["failed_validations"] += 1
                        return ValidationResult(
                            is_valid=False,
                            message=f"{name} contains unexpected keys: {unexpected_keys}"
                        )
                    else:
                        warnings.append(f"{name} unexpected keys removed: {unexpected_keys}")
                        data = {k: v for k, v in data.items() if k in allowed_keys}
                        self.validation_stats["sanitizations"] += 1
            
            return ValidationResult(
                is_valid=True,
                message=f"{name} validation passed",
                sanitized_input=data,
                warnings=warnings
            )
            
        except Exception as e:
            self.validation_stats["failed_validations"] += 1
            logger.error(f"Dictionary validation error: {e}")
            return ValidationResult(
                is_valid=False,
                message=f"{name} validation failed: {str(e)}"
            )
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        total = self.validation_stats["total_validations"]
        failed = self.validation_stats["failed_validations"]
        sanitized = self.validation_stats["sanitizations"]
        
        return {
            **self.validation_stats,
            "success_rate": (total - failed) / total if total > 0 else 0.0,
            "sanitization_rate": sanitized / total if total > 0 else 0.0,
        }


class FederatedInputValidator(InputValidator):
    """Input validator for federated learning specific inputs."""
    
    def __init__(self, **kwargs):
        """Initialize federated input validator."""
        super().__init__(**kwargs)
        
        # Federated learning specific patterns
        self.client_id_pattern = r"^[a-zA-Z0-9_-]{1,50}$"
        self.forbidden_client_patterns = [
            r"admin", r"server", r"system", r"root", r"test"
        ]
        
        logger.info("Initialized federated input validator")
    
    def validate_client_id(self, client_id: str) -> ValidationResult:
        """Validate client ID format."""
        return self.validate_string_input(
            client_id,
            max_length=50,
            allowed_pattern=self.client_id_pattern,
            forbidden_patterns=self.forbidden_client_patterns,
            name="client_id"
        )
    
    def validate_model_update(
        self,
        update: Dict[str, Any],
        expected_keys: Optional[List[str]] = None,
    ) -> ValidationResult:
        """Validate federated model update."""
        self.validation_stats["total_validations"] += 1
        warnings = []
        
        try:
            # Check if update is dictionary
            if not isinstance(update, dict):
                self.validation_stats["failed_validations"] += 1
                return ValidationResult(
                    is_valid=False,
                    message="Model update must be a dictionary"
                )
            
            # Check for empty update
            if not update:
                self.validation_stats["failed_validations"] += 1
                return ValidationResult(
                    is_valid=False,
                    message="Model update cannot be empty"
                )
            
            # Validate each parameter tensor
            sanitized_update = {}
            for name, tensor in update.items():
                tensor_result = self.validate_tensor_input(
                    tensor,
                    name=f"parameter_{name}"
                )
                
                if not tensor_result.is_valid:
                    self.validation_stats["failed_validations"] += 1
                    return ValidationResult(
                        is_valid=False,
                        message=f"Invalid parameter {name}: {tensor_result.message}"
                    )
                
                if tensor_result.warnings:
                    warnings.extend([f"{name}: {w}" for w in tensor_result.warnings])
                
                sanitized_update[name] = tensor_result.sanitized_input
            
            # Check expected parameter names
            if expected_keys is not None:
                missing_keys = set(expected_keys) - set(update.keys())
                if missing_keys:
                    warnings.append(f"Missing expected parameters: {missing_keys}")
                
                unexpected_keys = set(update.keys()) - set(expected_keys)
                if unexpected_keys:
                    warnings.append(f"Unexpected parameters: {unexpected_keys}")
            
            return ValidationResult(
                is_valid=True,
                message="Model update validation passed",
                sanitized_input=sanitized_update,
                warnings=warnings
            )
            
        except Exception as e:
            self.validation_stats["failed_validations"] += 1
            logger.error(f"Model update validation error: {e}")
            return ValidationResult(
                is_valid=False,
                message=f"Model update validation failed: {str(e)}"
            )
    
    def validate_privacy_parameters(
        self,
        epsilon: float,
        delta: float,
        sensitivity: float = 1.0,
    ) -> ValidationResult:
        """Validate differential privacy parameters."""
        self.validation_stats["total_validations"] += 1
        
        # Validate epsilon
        epsilon_result = self.validate_numeric_input(
            epsilon,
            min_val=0.001,
            max_val=10.0,
            name="epsilon"
        )
        
        if not epsilon_result.is_valid:
            self.validation_stats["failed_validations"] += 1
            return epsilon_result
        
        # Validate delta
        delta_result = self.validate_numeric_input(
            delta,
            min_val=1e-10,
            max_val=0.1,
            name="delta"
        )
        
        if not delta_result.is_valid:
            self.validation_stats["failed_validations"] += 1
            return delta_result
        
        # Validate sensitivity
        sensitivity_result = self.validate_numeric_input(
            sensitivity,
            min_val=0.001,
            max_val=100.0,
            name="sensitivity"
        )
        
        if not sensitivity_result.is_valid:
            self.validation_stats["failed_validations"] += 1
            return sensitivity_result
        
        warnings = []
        if epsilon_result.warnings:
            warnings.extend(epsilon_result.warnings)
        if delta_result.warnings:
            warnings.extend(delta_result.warnings)
        if sensitivity_result.warnings:
            warnings.extend(sensitivity_result.warnings)
        
        # Check privacy budget reasonableness
        if epsilon > 5.0:
            warnings.append("High epsilon value may compromise privacy")
        
        if delta > 1e-5:
            warnings.append("High delta value may compromise privacy")
        
        return ValidationResult(
            is_valid=True,
            message="Privacy parameters validation passed",
            sanitized_input={
                "epsilon": epsilon_result.sanitized_input,
                "delta": delta_result.sanitized_input,
                "sensitivity": sensitivity_result.sanitized_input,
            },
            warnings=warnings
        )
    
    def validate_federated_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate federated learning configuration."""
        required_keys = ["num_clients", "aggregation_method", "local_epochs"]
        optional_keys = [
            "privacy_budget", "compression_ratio", "client_fraction",
            "min_clients", "max_clients", "communication_rounds"
        ]
        
        # First validate as dictionary
        dict_result = self.validate_dict_input(
            config,
            required_keys=required_keys,
            optional_keys=optional_keys,
            name="federated_config"
        )
        
        if not dict_result.is_valid:
            return dict_result
        
        config = dict_result.sanitized_input
        warnings = dict_result.warnings or []
        
        # Validate specific federated parameters
        num_clients_result = self.validate_numeric_input(
            config["num_clients"],
            min_val=2,
            max_val=10000,
            integer_only=True,
            name="num_clients"
        )
        
        if not num_clients_result.is_valid:
            return num_clients_result
        
        # Validate aggregation method
        aggregation_result = self.validate_string_input(
            config["aggregation_method"],
            max_length=20,
            allowed_pattern=r"^(fedavg|fedprox|adaptive)$",
            name="aggregation_method"
        )
        
        if not aggregation_result.is_valid:
            return aggregation_result
        
        # Validate local epochs
        epochs_result = self.validate_numeric_input(
            config["local_epochs"],
            min_val=1,
            max_val=100,
            integer_only=True,
            name="local_epochs"
        )
        
        if not epochs_result.is_valid:
            return epochs_result
        
        # Combine all warnings
        if num_clients_result.warnings:
            warnings.extend(num_clients_result.warnings)
        if aggregation_result.warnings:
            warnings.extend(aggregation_result.warnings)
        if epochs_result.warnings:
            warnings.extend(epochs_result.warnings)
        
        return ValidationResult(
            is_valid=True,
            message="Federated configuration validation passed",
            sanitized_input=config,
            warnings=warnings
        )