"""Model optimization techniques for edge deployment."""

import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
from collections import OrderedDict
import copy


logger = logging.getLogger(__name__)


class ModelPruner:
    """Neural network pruning for edge deployment optimization."""
    
    def __init__(
        self,
        target_sparsity: float = 0.5,
        structured: bool = False,
        importance_scores: Optional[str] = "magnitude",
    ):
        """Initialize model pruner.
        
        Args:
            target_sparsity: Target sparsity level (0.0 to 1.0)
            structured: Whether to use structured pruning
            importance_scores: Method for computing importance scores
        """
        self.target_sparsity = target_sparsity
        self.structured = structured
        self.importance_scores = importance_scores
        
        # Pruning statistics
        self.pruning_history = []
        self.original_size = 0
        self.compressed_size = 0
        
        logger.info(
            f"Initialized model pruner: sparsity={target_sparsity}, "
            f"structured={structured}"
        )
    
    def compute_importance_scores(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        device: str = "cpu",
    ) -> Dict[str, torch.Tensor]:
        """Compute importance scores for model parameters.
        
        Args:
            model: Model to analyze
            dataloader: Data for computing importance
            criterion: Loss function
            device: Compute device
            
        Returns:
            Dictionary mapping parameter names to importance scores
        """
        model.eval()
        importance_scores = {}
        
        if self.importance_scores == "magnitude":
            # L2 magnitude-based importance
            for name, param in model.named_parameters():
                if param.requires_grad and len(param.shape) > 1:  # Skip biases
                    importance_scores[name] = torch.abs(param.data)
        
        elif self.importance_scores == "gradient":
            # Gradient-based importance (Fisher Information approximation)
            model.zero_grad()
            
            # Accumulate gradients over batch
            for batch_idx, (data, target) in enumerate(dataloader):
                if batch_idx >= 10:  # Limit batches for efficiency
                    break
                
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
            
            # Compute importance as gradient magnitude
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None and len(param.shape) > 1:
                    importance_scores[name] = torch.abs(param.grad.data)
            
            model.zero_grad()
        
        elif self.importance_scores == "taylor":
            # Taylor expansion-based importance
            model.zero_grad()
            
            for batch_idx, (data, target) in enumerate(dataloader):
                if batch_idx >= 5:
                    break
                
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
            
            # Taylor importance: |param * grad|
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None and len(param.shape) > 1:
                    importance_scores[name] = torch.abs(param.data * param.grad.data)
            
            model.zero_grad()
        
        else:
            raise ValueError(f"Unknown importance score method: {self.importance_scores}")
        
        return importance_scores
    
    def prune_unstructured(
        self,
        model: nn.Module,
        importance_scores: Dict[str, torch.Tensor],
    ) -> nn.Module:
        """Apply unstructured (magnitude-based) pruning.
        
        Args:
            model: Model to prune
            importance_scores: Parameter importance scores
            
        Returns:
            Pruned model
        """
        # Global magnitude pruning
        parameters_to_prune = []
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                parameters_to_prune.append((module, 'weight'))
        
        # Apply global unstructured pruning
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=self.target_sparsity,
        )
        
        # Make pruning permanent
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)
        
        return model
    
    def prune_structured(
        self,
        model: nn.Module,
        importance_scores: Dict[str, torch.Tensor],
    ) -> nn.Module:
        """Apply structured pruning (remove entire channels/filters).
        
        Args:
            model: Model to prune
            importance_scores: Parameter importance scores
            
        Returns:
            Pruned model
        """
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                # Channel pruning for convolutional layers
                weight = module.weight.data
                num_filters = weight.shape[0]
                num_to_prune = int(num_filters * self.target_sparsity)
                
                if num_to_prune > 0:
                    # Compute filter importance (L2 norm across spatial dimensions)
                    filter_importance = torch.norm(weight.view(weight.shape[0], -1), dim=1)
                    
                    # Get indices of least important filters
                    _, indices_to_prune = torch.topk(
                        filter_importance, num_to_prune, largest=False
                    )
                    
                    # Create mask
                    mask = torch.ones(num_filters, dtype=torch.bool)
                    mask[indices_to_prune] = False
                    
                    # Apply structured pruning
                    prune.structured(module, name='weight', amount=num_to_prune, dim=0)
                    prune.remove(module, 'weight')
        
        return model
    
    def prune_model(
        self,
        model: nn.Module,
        dataloader: Optional[torch.utils.data.DataLoader] = None,
        criterion: Optional[nn.Module] = None,
        device: str = "cpu",
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """Prune model to reduce size and computation.
        
        Args:
            model: Model to prune
            dataloader: Data for computing importance (optional)
            criterion: Loss function for importance computation (optional)
            device: Compute device
            
        Returns:
            Tuple of (pruned_model, pruning_stats)
        """
        start_time = time.time()
        
        # Calculate original model size
        self.original_size = sum(p.numel() for p in model.parameters())
        
        # Compute importance scores if data is provided
        if dataloader is not None and criterion is not None:
            importance_scores = self.compute_importance_scores(
                model, dataloader, criterion, device
            )
        else:
            importance_scores = {}
        
        # Apply pruning
        if self.structured:
            pruned_model = self.prune_structured(model, importance_scores)
        else:
            pruned_model = self.prune_unstructured(model, importance_scores)
        
        # Calculate compressed model size
        self.compressed_size = sum(
            (p != 0).sum().item() if p.numel() > 1 else p.numel()
            for p in pruned_model.parameters()
        )
        
        # Calculate statistics
        actual_sparsity = 1.0 - (self.compressed_size / self.original_size)
        compression_ratio = self.original_size / self.compressed_size
        pruning_time = time.time() - start_time
        
        pruning_stats = {
            "original_size": self.original_size,
            "compressed_size": self.compressed_size,
            "target_sparsity": self.target_sparsity,
            "actual_sparsity": actual_sparsity,
            "compression_ratio": compression_ratio,
            "pruning_time": pruning_time,
            "structured": self.structured,
        }
        
        self.pruning_history.append(pruning_stats)
        
        logger.info(
            f"Model pruned: {actual_sparsity:.2%} sparsity, "
            f"{compression_ratio:.2f}x compression in {pruning_time:.2f}s"
        )
        
        return pruned_model, pruning_stats


class ModelQuantizer:
    """Model quantization for edge deployment."""
    
    def __init__(
        self,
        quantization_bits: int = 8,
        quantization_mode: str = "dynamic",
        calibration_method: str = "minmax",
    ):
        """Initialize model quantizer.
        
        Args:
            quantization_bits: Number of bits for quantization (8, 16)
            quantization_mode: Quantization mode ("static", "dynamic", "qat")
            calibration_method: Calibration method for static quantization
        """
        self.quantization_bits = quantization_bits
        self.quantization_mode = quantization_mode
        self.calibration_method = calibration_method
        
        # Quantization statistics
        self.quantization_history = []
        
        logger.info(
            f"Initialized model quantizer: {quantization_bits}-bit, "
            f"mode={quantization_mode}"
        )
    
    def prepare_model_for_quantization(self, model: nn.Module) -> nn.Module:
        """Prepare model for quantization.
        
        Args:
            model: Model to prepare
            
        Returns:
            Prepared model
        """
        # Set quantization configuration
        if self.quantization_bits == 8:
            qconfig = torch.quantization.get_default_qconfig('fbgemm')
        elif self.quantization_bits == 16:
            # Use custom 16-bit quantization config
            qconfig = torch.quantization.QConfig(
                activation=torch.quantization.MinMaxObserver.with_args(
                    dtype=torch.float16
                ),
                weight=torch.quantization.MinMaxObserver.with_args(
                    dtype=torch.float16
                )
            )
        else:
            raise ValueError(f"Unsupported quantization bits: {self.quantization_bits}")
        
        model.qconfig = qconfig
        
        # Prepare model
        if self.quantization_mode == "static":
            prepared_model = torch.quantization.prepare(model)
        elif self.quantization_mode == "dynamic":
            prepared_model = model  # Dynamic quantization doesn't need preparation
        elif self.quantization_mode == "qat":
            prepared_model = torch.quantization.prepare_qat(model)
        else:
            raise ValueError(f"Unknown quantization mode: {self.quantization_mode}")
        
        return prepared_model
    
    def calibrate_model(
        self,
        model: nn.Module,
        calibration_loader: torch.utils.data.DataLoader,
        device: str = "cpu",
    ) -> nn.Module:
        """Calibrate model for static quantization.
        
        Args:
            model: Prepared model
            calibration_loader: Calibration data
            device: Compute device
            
        Returns:
            Calibrated model
        """
        model.eval()
        
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(calibration_loader):
                if batch_idx >= 100:  # Limit calibration batches
                    break
                
                data = data.to(device)
                model(data)
        
        return model
    
    def quantize_model(
        self,
        model: nn.Module,
        calibration_loader: Optional[torch.utils.data.DataLoader] = None,
        device: str = "cpu",
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """Quantize model for edge deployment.
        
        Args:
            model: Model to quantize
            calibration_loader: Calibration data for static quantization
            device: Compute device
            
        Returns:
            Tuple of (quantized_model, quantization_stats)
        """
        start_time = time.time()
        
        # Calculate original model size
        original_size = sum(p.numel() * 4 for p in model.parameters())  # 32-bit floats
        
        # Prepare model for quantization
        if self.quantization_mode in ["static", "qat"]:
            prepared_model = self.prepare_model_for_quantization(model)
            
            # Calibrate for static quantization
            if self.quantization_mode == "static" and calibration_loader is not None:
                prepared_model = self.calibrate_model(prepared_model, calibration_loader, device)
            
            # Convert to quantized model
            quantized_model = torch.quantization.convert(prepared_model)
        
        elif self.quantization_mode == "dynamic":
            # Dynamic quantization
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear, nn.Conv2d},
                dtype=torch.qint8 if self.quantization_bits == 8 else torch.float16
            )
        
        else:
            raise ValueError(f"Unknown quantization mode: {self.quantization_mode}")
        
        # Calculate quantized model size
        if self.quantization_bits == 8:
            quantized_size = sum(p.numel() for p in quantized_model.parameters())
        elif self.quantization_bits == 16:
            quantized_size = sum(p.numel() * 2 for p in quantized_model.parameters())
        else:
            quantized_size = original_size
        
        # Calculate statistics
        compression_ratio = original_size / quantized_size
        quantization_time = time.time() - start_time
        
        quantization_stats = {
            "original_size_bytes": original_size,
            "quantized_size_bytes": quantized_size,
            "compression_ratio": compression_ratio,
            "quantization_bits": self.quantization_bits,
            "quantization_mode": self.quantization_mode,
            "quantization_time": quantization_time,
        }
        
        self.quantization_history.append(quantization_stats)
        
        logger.info(
            f"Model quantized: {self.quantization_bits}-bit, "
            f"{compression_ratio:.2f}x compression in {quantization_time:.2f}s"
        )
        
        return quantized_model, quantization_stats


class TensorRTOptimizer:
    """TensorRT optimization for NVIDIA edge devices."""
    
    def __init__(
        self,
        precision: str = "fp16",
        max_workspace_size: int = 1 << 30,  # 1GB
        max_batch_size: int = 1,
    ):
        """Initialize TensorRT optimizer.
        
        Args:
            precision: Precision mode ("fp32", "fp16", "int8")
            max_workspace_size: Maximum workspace size in bytes
            max_batch_size: Maximum batch size for optimization
        """
        self.precision = precision
        self.max_workspace_size = max_workspace_size
        self.max_batch_size = max_batch_size
        
        # Check TensorRT availability
        try:
            import tensorrt as trt
            self.trt_available = True
            logger.info(f"TensorRT available: version {trt.__version__}")
        except ImportError:
            self.trt_available = False
            logger.warning("TensorRT not available, optimization will be skipped")
    
    def optimize_with_tensorrt(
        self,
        onnx_model_path: str,
        output_engine_path: str,
        input_shape: Tuple[int, ...],
        calibration_loader: Optional[torch.utils.data.DataLoader] = None,
    ) -> Dict[str, Any]:
        """Optimize ONNX model with TensorRT.
        
        Args:
            onnx_model_path: Path to ONNX model
            output_engine_path: Path to save TensorRT engine
            input_shape: Input tensor shape (batch_size, channels, height, width)
            calibration_loader: Calibration data for INT8 optimization
            
        Returns:
            Optimization statistics
        """
        if not self.trt_available:
            logger.error("TensorRT not available")
            return {"error": "TensorRT not available"}
        
        try:
            import tensorrt as trt
            
            # Create TensorRT logger
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            
            # Create builder and network
            builder = trt.Builder(TRT_LOGGER)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, TRT_LOGGER)
            
            # Parse ONNX model
            with open(onnx_model_path, 'rb') as model:
                if not parser.parse(model.read()):
                    logger.error("Failed to parse ONNX model")
                    return {"error": "ONNX parsing failed"}
            
            # Create builder config
            config = builder.create_builder_config()
            config.max_workspace_size = self.max_workspace_size
            
            # Set precision
            if self.precision == "fp16":
                config.set_flag(trt.BuilderFlag.FP16)
            elif self.precision == "int8":
                config.set_flag(trt.BuilderFlag.INT8)
                
                if calibration_loader is not None:
                    # Set up INT8 calibrator
                    calibrator = TensorRTCalibrator(calibration_loader, input_shape)
                    config.int8_calibrator = calibrator
            
            # Build engine
            start_time = time.time()
            engine = builder.build_engine(network, config)
            optimization_time = time.time() - start_time
            
            if engine is None:
                logger.error("Failed to build TensorRT engine")
                return {"error": "Engine build failed"}
            
            # Serialize and save engine
            with open(output_engine_path, 'wb') as f:
                f.write(engine.serialize())
            
            # Get optimization statistics
            stats = {
                "optimization_time": optimization_time,
                "precision": self.precision,
                "max_workspace_size": self.max_workspace_size,
                "max_batch_size": self.max_batch_size,
                "engine_path": output_engine_path,
                "success": True,
            }
            
            logger.info(
                f"TensorRT optimization completed in {optimization_time:.2f}s, "
                f"engine saved to {output_engine_path}"
            )
            
            return stats
            
        except Exception as e:
            logger.error(f"TensorRT optimization failed: {e}")
            return {"error": str(e), "success": False}


class TensorRTCalibrator:
    """INT8 calibrator for TensorRT."""
    
    def __init__(
        self,
        dataloader: torch.utils.data.DataLoader,
        input_shape: Tuple[int, ...],
        cache_file: str = "calibration.cache",
    ):
        """Initialize calibrator.
        
        Args:
            dataloader: Calibration data
            input_shape: Input tensor shape
            cache_file: Cache file for calibration
        """
        self.dataloader = dataloader
        self.input_shape = input_shape
        self.cache_file = cache_file
        self.current_index = 0
        self.data_iter = iter(dataloader)
        
        try:
            import tensorrt as trt
            self.calibrator_class = trt.IInt8EntropyCalibrator2
        except ImportError:
            self.calibrator_class = None
    
    def get_batch_size(self) -> int:
        """Get batch size for calibration."""
        return self.input_shape[0]
    
    def get_batch(self, names):
        """Get next batch for calibration."""
        try:
            data, _ = next(self.data_iter)
            # Convert to numpy and flatten
            batch = data.numpy().astype(np.float32)
            return [batch]
        except StopIteration:
            return None
    
    def read_calibration_cache(self):
        """Read calibration cache if available."""
        try:
            with open(self.cache_file, "rb") as f:
                return f.read()
        except FileNotFoundError:
            return None
    
    def write_calibration_cache(self, cache):
        """Write calibration cache."""
        with open(self.cache_file, "wb") as f:
            f.write(cache)


class EdgeOptimizer:
    """Unified edge optimization pipeline."""
    
    def __init__(
        self,
        target_latency_ms: float = 100.0,
        target_memory_mb: float = 500.0,
        target_power_w: float = 15.0,
        device_type: str = "jetson_xavier_nx",
    ):
        """Initialize edge optimizer.
        
        Args:
            target_latency_ms: Target inference latency in milliseconds
            target_memory_mb: Target memory usage in megabytes
            target_power_w: Target power consumption in watts
            device_type: Target edge device type
        """
        self.target_latency = target_latency_ms
        self.target_memory = target_memory_mb
        self.target_power = target_power_w
        self.device_type = device_type
        
        # Initialize optimizers
        self.pruner = ModelPruner()
        self.quantizer = ModelQuantizer()
        self.tensorrt_optimizer = TensorRTOptimizer()
        
        logger.info(
            f"Initialized edge optimizer for {device_type}: "
            f"latency<{target_latency_ms}ms, memory<{target_memory_mb}MB"
        )
    
    def optimize_for_edge(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        calibration_loader: Optional[torch.utils.data.DataLoader] = None,
        criterion: Optional[nn.Module] = None,
        output_dir: str = "./optimized_models",
    ) -> Dict[str, Any]:
        """Complete edge optimization pipeline.
        
        Args:
            model: Model to optimize
            sample_input: Sample input for testing
            calibration_loader: Calibration data
            criterion: Loss function for pruning
            output_dir: Directory to save optimized models
            
        Returns:
            Optimization results and statistics
        """
        optimization_results = {
            "original_model": self._benchmark_model(model, sample_input),
            "optimization_steps": [],
        }
        
        current_model = model
        
        # Step 1: Pruning
        if calibration_loader is not None and criterion is not None:
            logger.info("Starting model pruning...")
            pruned_model, pruning_stats = self.pruner.prune_model(
                current_model, calibration_loader, criterion
            )
            
            pruning_benchmark = self._benchmark_model(pruned_model, sample_input)
            optimization_results["optimization_steps"].append({
                "step": "pruning",
                "stats": pruning_stats,
                "benchmark": pruning_benchmark,
            })
            
            current_model = pruned_model
        
        # Step 2: Quantization
        logger.info("Starting model quantization...")
        quantized_model, quantization_stats = self.quantizer.quantize_model(
            current_model, calibration_loader
        )
        
        quantization_benchmark = self._benchmark_model(quantized_model, sample_input)
        optimization_results["optimization_steps"].append({
            "step": "quantization",
            "stats": quantization_stats,
            "benchmark": quantization_benchmark,
        })
        
        current_model = quantized_model
        
        # Step 3: TensorRT optimization (if available)
        if self.tensorrt_optimizer.trt_available:
            logger.info("Starting TensorRT optimization...")
            
            # Export to ONNX first
            onnx_path = f"{output_dir}/model.onnx"
            self._export_to_onnx(current_model, sample_input, onnx_path)
            
            # Optimize with TensorRT
            engine_path = f"{output_dir}/model.trt"
            tensorrt_stats = self.tensorrt_optimizer.optimize_with_tensorrt(
                onnx_path, engine_path, sample_input.shape, calibration_loader
            )
            
            optimization_results["optimization_steps"].append({
                "step": "tensorrt",
                "stats": tensorrt_stats,
            })
        
        # Final results
        optimization_results["final_model"] = self._benchmark_model(current_model, sample_input)
        optimization_results["meets_targets"] = self._check_targets(
            optimization_results["final_model"]
        )
        
        logger.info("Edge optimization pipeline completed")
        return optimization_results
    
    def _benchmark_model(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        num_runs: int = 100,
    ) -> Dict[str, float]:
        """Benchmark model performance.
        
        Args:
            model: Model to benchmark
            sample_input: Sample input tensor
            num_runs: Number of benchmark runs
            
        Returns:
            Benchmark results
        """
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(sample_input)
        
        # Benchmark latency
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(sample_input)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        avg_latency_ms = (end_time - start_time) / num_runs * 1000
        
        # Calculate model size
        model_size_mb = sum(p.numel() * 4 for p in model.parameters()) / (1024 * 1024)
        
        return {
            "latency_ms": avg_latency_ms,
            "model_size_mb": model_size_mb,
            "parameters": sum(p.numel() for p in model.parameters()),
        }
    
    def _export_to_onnx(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        output_path: str,
    ) -> None:
        """Export model to ONNX format.
        
        Args:
            model: Model to export
            sample_input: Sample input for tracing
            output_path: Output ONNX file path
        """
        model.eval()
        
        torch.onnx.export(
            model,
            sample_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        logger.info(f"Model exported to ONNX: {output_path}")
    
    def _check_targets(self, benchmark_results: Dict[str, float]) -> Dict[str, bool]:
        """Check if optimization meets target constraints.
        
        Args:
            benchmark_results: Benchmark results
            
        Returns:
            Dictionary of constraint satisfaction
        """
        return {
            "latency_target": benchmark_results["latency_ms"] <= self.target_latency,
            "memory_target": benchmark_results["model_size_mb"] <= self.target_memory,
            "overall_success": (
                benchmark_results["latency_ms"] <= self.target_latency and
                benchmark_results["model_size_mb"] <= self.target_memory
            ),
        }