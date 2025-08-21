"""Performance optimization utilities for federated learning and edge deployment."""

from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import time
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass
import logging
import os
try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False
    psutil = None
import gc
import math

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    import torch.multiprocessing as mp
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None


logger = logging.getLogger(__name__)


@dataclass
class PerformanceProfile:
    """Performance profiling results."""
    inference_time_ms: float
    memory_usage_mb: float
    gpu_utilization: float
    cpu_utilization: float
    throughput_fps: float
    energy_consumption: Optional[float] = None


class ModelOptimizer:
    """Comprehensive model optimization for edge deployment."""

    def __init__(self, target_latency_ms: float = 50.0, target_memory_mb: float = 512.0):
        """Initialize model optimizer.

        Args:
            target_latency_ms: Target inference latency in milliseconds
            target_memory_mb: Target memory usage in megabytes
        """
        self.target_latency_ms = target_latency_ms
        self.target_memory_mb = target_memory_mb
        self.optimization_history = []

    def optimize_for_edge(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        optimization_level: int = 2,
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """Optimize model for edge deployment.

        Args:
            model: PyTorch model to optimize
            sample_input: Sample input tensor for calibration
            optimization_level: 0=basic, 1=moderate, 2=aggressive

        Returns:
            Tuple of (optimized_model, optimization_report)
        """
        optimization_report = {
            'original_params': sum(p.numel() for p in model.parameters()),
            'optimizations_applied': [],
            'performance_improvement': {},
        }

        # Baseline measurement
        original_profile = self._profile_model(model, sample_input)
        optimization_report['original_performance'] = original_profile

        optimized_model = model

        # Apply optimizations based on level
        if optimization_level >= 1:
            # Basic optimizations
            optimized_model = self._apply_basic_optimizations(optimized_model, sample_input)
            optimization_report['optimizations_applied'].extend(['fusion', 'constant_folding'])

        if optimization_level >= 2:
            # Moderate optimizations
            optimized_model = self._apply_quantization(optimized_model, sample_input)
            optimization_report['optimizations_applied'].append('quantization')

            optimized_model = self._apply_pruning(optimized_model, sparsity=0.3)
            optimization_report['optimizations_applied'].append('pruning_30')

        if optimization_level >= 3:
            # Aggressive optimizations
            optimized_model = self._apply_knowledge_distillation(optimized_model, model, sample_input)
            optimization_report['optimizations_applied'].append('knowledge_distillation')

            optimized_model = self._apply_neural_architecture_search(optimized_model, sample_input)
            optimization_report['optimizations_applied'].append('architecture_search')

        # Final measurement
        final_profile = self._profile_model(optimized_model, sample_input)
        optimization_report['final_performance'] = final_profile

        # Calculate improvements
        optimization_report['performance_improvement'] = {
            'latency_reduction': (original_profile.inference_time_ms - final_profile.inference_time_ms) / original_profile.inference_time_ms,
            'memory_reduction': (original_profile.memory_usage_mb - final_profile.memory_usage_mb) / original_profile.memory_usage_mb,
            'throughput_improvement': (final_profile.throughput_fps - original_profile.throughput_fps) / original_profile.throughput_fps,
        }

        optimization_report['final_params'] = sum(p.numel() for p in optimized_model.parameters())
        optimization_report['compression_ratio'] = optimization_report['original_params'] / optimization_report['final_params']

        self.optimization_history.append(optimization_report)

        return optimized_model, optimization_report

    def _profile_model(self, model: nn.Module, sample_input: torch.Tensor) -> PerformanceProfile:
        """Profile model performance."""
        model.eval()
        device = next(model.parameters()).device
        sample_input = sample_input.to(device)

        # Warm up
        with torch.no_grad():
            for _ in range(10):
                _ = model(sample_input)

        torch.cuda.synchronize() if device.type == 'cuda' else None

        # Measure inference time
        start_time = time.perf_counter()
        with torch.no_grad():
            for _ in range(100):
                _ = model(sample_input)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        end_time = time.perf_counter()

        inference_time_ms = (end_time - start_time) * 1000 / 100

        # Measure memory usage
        if device.type == 'cuda':
            memory_usage_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            torch.cuda.reset_peak_memory_stats()
        else:
            process = psutil.Process(os.getpid())
            memory_usage_mb = process.memory_info().rss / (1024 ** 2)

        # Calculate throughput
        throughput_fps = 1000.0 / inference_time_ms if inference_time_ms > 0 else 0.0

        return PerformanceProfile(
            inference_time_ms=inference_time_ms,
            memory_usage_mb=memory_usage_mb,
            gpu_utilization=0.0,  # Would require nvidia-ml-py for real measurement
            cpu_utilization=psutil.cpu_percent(),
            throughput_fps=throughput_fps,
        )

    def _apply_basic_optimizations(self, model: nn.Module, sample_input: torch.Tensor) -> nn.Module:
        """Apply basic optimizations like operator fusion."""
        # Trace and optimize the model
        model.eval()
        with torch.no_grad():
            traced_model = torch.jit.trace(model, sample_input)
            optimized_model = torch.jit.optimize_for_inference(traced_model)

        return optimized_model

    def _apply_quantization(self, model: nn.Module, sample_input: torch.Tensor) -> nn.Module:
        """Apply dynamic quantization."""
        # Prepare model for quantization
        model.eval()

        # Dynamic quantization for Linear and LSTM layers
        quantized_model = torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )

        return quantized_model

    def _apply_pruning(self, model: nn.Module, sparsity: float = 0.3) -> nn.Module:
        """Apply structured pruning to reduce model size."""
        import torch.nn.utils.prune as prune

        # Apply magnitude-based pruning to linear layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=sparsity)

        # Remove pruning reparameterization
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                try:
                    prune.remove(module, 'weight')
                except ValueError:
                    pass  # Pruning not applied

        return model

    def _apply_knowledge_distillation(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        sample_input: torch.Tensor
    ) -> nn.Module:
        """Apply knowledge distillation to compress the model."""
        # This is a simplified version - in practice, you'd need training data
        # and a proper distillation training loop

        # For now, just return the student model
        # In a real implementation, this would:
        # 1. Create a smaller student model
        # 2. Train it to mimic the teacher's outputs
        # 3. Use soft targets and temperature scaling

        logger.info("Knowledge distillation placeholder - returning original model")
        return student_model

    def _apply_neural_architecture_search(self, model: nn.Module, sample_input: torch.Tensor) -> nn.Module:
        """Apply neural architecture search for optimal model design."""
        # This is a placeholder for NAS
        # In practice, this would:
        # 1. Define a search space of architectures
        # 2. Use evolutionary algorithms or reinforcement learning
        # 3. Evaluate architectures on target hardware
        # 4. Select the best architecture within constraints

        logger.info("Neural architecture search placeholder - returning original model")
        return model


class DistributedTrainingManager:
    """Manage distributed training across multiple GPUs and nodes."""

    def __init__(self, world_size: int = 1, rank: int = 0):
        """Initialize distributed training manager.

        Args:
            world_size: Total number of processes
            rank: Rank of current process
        """
        self.world_size = world_size
        self.rank = rank
        self.is_initialized = False

    def setup_distributed(self, backend: str = 'nccl') -> None:
        """Setup distributed training environment."""
        if self.world_size > 1 and not self.is_initialized:
            dist.init_process_group(
                backend=backend,
                rank=self.rank,
                world_size=self.world_size
            )
            self.is_initialized = True
            logger.info(f"Distributed training initialized: rank {self.rank}/{self.world_size}")

    def wrap_model(self, model: nn.Module) -> nn.Module:
        """Wrap model for distributed training."""
        if self.world_size > 1 and self.is_initialized:
            return DDP(model, device_ids=[self.rank % torch.cuda.device_count()])
        return model

    def cleanup_distributed(self) -> None:
        """Cleanup distributed training."""
        if self.is_initialized:
            dist.destroy_process_group()
            self.is_initialized = False


class MemoryManager:
    """Advanced memory management for large-scale training."""

    def __init__(self, max_memory_gb: float = 8.0):
        """Initialize memory manager.

        Args:
            max_memory_gb: Maximum memory usage in GB
        """
        self.max_memory_bytes = max_memory_gb * (1024 ** 3)
        self.memory_history = []
        self.gc_threshold = 0.8  # Trigger GC at 80% memory usage

    def monitor_memory(self) -> Dict[str, float]:
        """Monitor current memory usage."""
        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated()
            gpu_reserved = torch.cuda.memory_reserved()
            gpu_free = torch.cuda.get_device_properties(0).total_memory - gpu_reserved

            memory_info = {
                'gpu_allocated_gb': gpu_allocated / (1024 ** 3),
                'gpu_reserved_gb': gpu_reserved / (1024 ** 3),
                'gpu_free_gb': gpu_free / (1024 ** 3),
                'gpu_utilization': gpu_allocated / torch.cuda.get_device_properties(0).total_memory,
            }
        else:
            memory_info = {'gpu_allocated_gb': 0, 'gpu_reserved_gb': 0, 'gpu_free_gb': 0, 'gpu_utilization': 0}

        # CPU memory
        process = psutil.Process(os.getpid())
        cpu_memory = process.memory_info().rss
        memory_info.update({
            'cpu_memory_gb': cpu_memory / (1024 ** 3),
            'cpu_memory_percent': process.memory_percent(),
        })

        self.memory_history.append((time.time(), memory_info))

        # Trigger cleanup if needed
        if memory_info.get('gpu_utilization', 0) > self.gc_threshold:
            self.cleanup_memory()

        return memory_info

    def cleanup_memory(self) -> None:
        """Perform memory cleanup."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()
        logger.info("Memory cleanup performed")

    def optimize_batch_size(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        max_batch_size: int = 64,
    ) -> int:
        """Find optimal batch size for given memory constraints."""
        model.eval()
        optimal_batch_size = 1

        for batch_size in [1, 2, 4, 8, 16, 32, 64]:
            if batch_size > max_batch_size:
                break

            try:
                # Create batch
                batch_input = sample_input.repeat(batch_size, 1, 1, 1)

                # Test forward pass
                with torch.no_grad():
                    _ = model(batch_input)

                # Check memory usage
                memory_info = self.monitor_memory()
                if memory_info.get('gpu_utilization', 0) < 0.9:  # 90% threshold
                    optimal_batch_size = batch_size
                else:
                    break

            except RuntimeError as e:
                if "out of memory" in str(e):
                    break
                raise

        logger.info(f"Optimal batch size determined: {optimal_batch_size}")
        return optimal_batch_size


class AsyncDataLoader:
    """Asynchronous data loading for improved training efficiency."""

    def __init__(self, dataset: Dataset, batch_size: int, num_workers: int = 4):
        """Initialize async data loader.

        Args:
            dataset: Dataset to load from
            batch_size: Batch size
            num_workers: Number of worker processes
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_buffer = queue.Queue(maxsize=16)
        self.loading_thread = None
        self.stop_loading = threading.Event()

    def start_loading(self) -> None:
        """Start background data loading."""
        self.stop_loading.clear()
        self.loading_thread = threading.Thread(target=self._loading_worker)
        self.loading_thread.start()

    def stop_loading(self) -> None:
        """Stop background data loading."""
        self.stop_loading.set()
        if self.loading_thread:
            self.loading_thread.join(timeout=5.0)

    def _loading_worker(self) -> None:
        """Background worker for data loading."""
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        while not self.stop_loading.is_set():
            try:
                for batch in dataloader:
                    if self.stop_loading.is_set():
                        break

                    # Add to prefetch buffer (blocking if full)
                    self.prefetch_buffer.put(batch, timeout=1.0)

            except queue.Full:
                continue
            except Exception as e:
                logger.error(f"Data loading error: {e}")

    def get_batch(self, timeout: float = 1.0) -> Optional[Any]:
        """Get next batch from prefetch buffer."""
        try:
            return self.prefetch_buffer.get(timeout=timeout)
        except queue.Empty:
            return None

    def __len__(self) -> int:
        """Get number of batches."""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class GradientCompressor:
    """Advanced gradient compression for communication efficiency."""

    def __init__(self, compression_ratio: float = 0.01, error_feedback: bool = True):
        """Initialize gradient compressor.

        Args:
            compression_ratio: Target compression ratio (0.01 = 1% of original)
            error_feedback: Whether to use error feedback
        """
        self.compression_ratio = compression_ratio
        self.error_feedback = error_feedback
        self.error_memory = {}

    def compress_gradients(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Compress gradients using multiple techniques.

        Args:
            gradients: Dictionary of gradient tensors

        Returns:
            Compressed gradient data
        """
        compressed = {}

        for name, grad in gradients.items():
            if grad is None:
                continue

            # Add error feedback
            if self.error_feedback and name in self.error_memory:
                grad = grad + self.error_memory[name]

            # Top-k sparsification
            compressed_grad, indices = self._top_k_sparsification(grad, self.compression_ratio)

            # Quantization
            quantized_grad = self._quantize_gradient(compressed_grad)

            compressed[name] = {
                'values': quantized_grad,
                'indices': indices,
                'shape': grad.shape,
                'dtype': grad.dtype,
            }

            # Update error memory
            if self.error_feedback:
                reconstructed = self._reconstruct_gradient(compressed[name])
                self.error_memory[name] = grad - reconstructed

        return compressed

    def decompress_gradients(self, compressed_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Decompress gradients.

        Args:
            compressed_data: Compressed gradient data

        Returns:
            Reconstructed gradients
        """
        gradients = {}

        for name, data in compressed_data.items():
            gradients[name] = self._reconstruct_gradient(data)

        return gradients

    def _top_k_sparsification(self, tensor: torch.Tensor, ratio: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply top-k sparsification to tensor."""
        flat_tensor = tensor.flatten()
        k = max(1, int(len(flat_tensor) * ratio))

        # Get top-k values by absolute magnitude
        _, indices = torch.topk(torch.abs(flat_tensor), k)
        values = flat_tensor[indices]

        return values, indices

    def _quantize_gradient(self, tensor: torch.Tensor, bits: int = 8) -> torch.Tensor:
        """Quantize gradient tensor."""
        if tensor.numel() == 0:
            return tensor

        # Simple uniform quantization
        min_val, max_val = tensor.min(), tensor.max()

        if min_val == max_val:
            return torch.zeros_like(tensor, dtype=torch.int8)

        scale = (max_val - min_val) / (2**bits - 1)
        quantized = torch.round((tensor - min_val) / scale).clamp(0, 2**bits - 1)

        return quantized.to(torch.int8)

    def _reconstruct_gradient(self, compressed_data: Dict[str, Any]) -> torch.Tensor:
        """Reconstruct gradient from compressed data."""
        values = compressed_data['values'].float()
        indices = compressed_data['indices']
        shape = compressed_data['shape']

        # Reconstruct sparse tensor
        reconstructed = torch.zeros(math.prod(shape), dtype=values.dtype, device=values.device)

        if len(values) > 0:
            reconstructed[indices] = values

        return reconstructed.view(shape)

    def get_compression_stats(self, original_size: int, compressed_size: int) -> Dict[str, float]:
        """Calculate compression statistics."""
        return {
            'compression_ratio': compressed_size / original_size,
            'space_savings': 1.0 - (compressed_size / original_size),
            'compression_factor': original_size / compressed_size,
        }


class AdaptiveScheduler:
    """Adaptive learning rate and resource scheduling."""

    def __init__(self, initial_lr: float = 0.001):
        """Initialize adaptive scheduler.

        Args:
            initial_lr: Initial learning rate
        """
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.performance_history = []
        self.resource_history = []

    def update_learning_rate(
        self,
        optimizer: optim.Optimizer,
        loss: float,
        metrics: Dict[str, float],
    ) -> float:
        """Adaptively update learning rate based on performance.

        Args:
            optimizer: PyTorch optimizer
            loss: Current loss value
            metrics: Current performance metrics

        Returns:
            New learning rate
        """
        self.performance_history.append((time.time(), loss, metrics.copy()))

        # Simple adaptive strategy based on loss trend
        if len(self.performance_history) >= 5:
            recent_losses = [entry[1] for entry in self.performance_history[-5:]]

            # Check if loss is increasing (overfitting or too high LR)
            if all(recent_losses[i] <= recent_losses[i+1] for i in range(len(recent_losses)-1)):
                self.current_lr *= 0.9  # Reduce LR
                logger.info(f"Reducing learning rate to {self.current_lr:.6f}")

            # Check if loss plateaued
            elif max(recent_losses) - min(recent_losses) < 0.001:
                self.current_lr *= 1.05  # Slightly increase LR
                logger.info(f"Increasing learning rate to {self.current_lr:.6f}")

        # Update optimizer
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.current_lr

        return self.current_lr

    def schedule_resources(self, available_resources: Dict[str, Any]) -> Dict[str, Any]:
        """Schedule computational resources adaptively.

        Args:
            available_resources: Available computational resources

        Returns:
            Resource allocation recommendations
        """
        self.resource_history.append((time.time(), available_resources.copy()))

        # Simple resource scheduling logic
        allocation = {
            'batch_size': min(64, available_resources.get('memory_gb', 4) * 8),
            'num_workers': min(8, available_resources.get('cpu_cores', 2)),
            'mixed_precision': available_resources.get('has_tensor_cores', False),
        }

        return allocation
