"""
Performance benchmarks for Fed-ViT-AutoRL models.

These benchmarks measure and track performance metrics including:
- Inference latency
- Training throughput
- Memory usage
- Model accuracy
- Communication overhead
"""

import time
import pytest
import torch
import psutil
import gc
from typing import Dict, List

from fed_vit_autorl.models.vit_perception import ViTPerception


@pytest.mark.benchmark
class TestModelPerformanceBenchmarks:
    """Benchmark tests for model performance."""

    def test_vit_inference_latency_benchmark(self, benchmark, benchmark_config):
        """Benchmark ViT inference latency across different configurations."""
        model = ViTPerception(num_classes=1000, img_size=224)
        model.eval()

        sample_input = torch.randn(1, 3, 224, 224)

        def inference_step():
            with torch.no_grad():
                return model(sample_input)

        # Warm up
        for _ in range(benchmark_config["warmup_iterations"]):
            inference_step()

        # Benchmark
        result = benchmark(inference_step)

        # Assertions for reasonable performance
        assert result is not None
        assert hasattr(result, 'shape')
        assert result.shape == (1, 1000)

    @pytest.mark.parametrize("batch_size", [1, 4, 8, 16])
    def test_vit_batch_inference_scaling(self, benchmark, batch_size):
        """Benchmark how inference scales with batch size."""
        model = ViTPerception(num_classes=1000, img_size=224)
        model.eval()

        sample_input = torch.randn(batch_size, 3, 224, 224)

        def batch_inference():
            with torch.no_grad():
                return model(sample_input)

        # Warm up
        for _ in range(3):
            batch_inference()

        result = benchmark(batch_inference)

        assert result.shape == (batch_size, 1000)

    @pytest.mark.parametrize("img_size", [224, 384])
    def test_vit_input_size_performance(self, benchmark, img_size):
        """Benchmark performance with different input sizes."""
        model = ViTPerception(num_classes=1000, img_size=img_size)
        model.eval()

        sample_input = torch.randn(1, 3, img_size, img_size)

        def inference_with_size():
            with torch.no_grad():
                return model(sample_input)

        # Warm up
        for _ in range(3):
            inference_with_size()

        result = benchmark(inference_with_size)

        assert result.shape == (1, 1000)

    def test_memory_usage_benchmark(self):
        """Benchmark memory usage during model operations."""
        # Get initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create model
        model = ViTPerception(num_classes=1000, img_size=224)
        model_memory = process.memory_info().rss / 1024 / 1024 - initial_memory

        # Forward pass
        sample_input = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            output = model(sample_input)

        forward_memory = process.memory_info().rss / 1024 / 1024 - initial_memory

        # Backward pass (training mode)
        model.train()
        criterion = torch.nn.CrossEntropyLoss()
        targets = torch.randint(0, 1000, (4,))

        output = model(sample_input)
        loss = criterion(output, targets)
        loss.backward()

        backward_memory = process.memory_info().rss / 1024 / 1024 - initial_memory

        # Cleanup
        del model, sample_input, output, loss
        gc.collect()

        # Memory usage report
        memory_report = {
            "initial_memory_mb": initial_memory,
            "model_memory_mb": model_memory,
            "forward_memory_mb": forward_memory,
            "backward_memory_mb": backward_memory
        }

        # Assertions for reasonable memory usage
        assert model_memory > 0  # Model should use some memory
        assert forward_memory >= model_memory  # Forward pass uses at least model memory
        assert backward_memory >= forward_memory  # Backward pass uses more memory

        # Log memory usage for analysis
        print(f"Memory usage report: {memory_report}")

    def test_training_throughput_benchmark(self, benchmark):
        """Benchmark training throughput (samples per second)."""
        model = ViTPerception(num_classes=10, img_size=224)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Create training batch
        batch_size = 8
        sample_input = torch.randn(batch_size, 3, 224, 224)
        sample_targets = torch.randint(0, 10, (batch_size,))

        def training_step():
            optimizer.zero_grad()
            outputs = model(sample_input)
            loss = criterion(outputs, sample_targets)
            loss.backward()
            optimizer.step()
            return loss.item()

        # Warm up
        model.train()
        for _ in range(3):
            training_step()

        # Benchmark training step
        result = benchmark(training_step)

        # Calculate throughput
        # Note: benchmark.stats contains timing information
        # We can derive samples per second from this
        assert isinstance(result, float)  # Loss value
        assert result >= 0  # Loss should be non-negative

    def test_feature_extraction_benchmark(self, benchmark):
        """Benchmark feature extraction performance."""
        model = ViTPerception(num_classes=1000, img_size=224)
        model.eval()

        sample_input = torch.randn(4, 3, 224, 224)

        def extract_features():
            with torch.no_grad():
                return model.extract_features(sample_input)

        # Warm up
        for _ in range(3):
            extract_features()

        result = benchmark(extract_features)

        # ViT features: batch_size x (num_patches + 1) x embed_dim
        expected_shape = (4, 197, 768)  # 196 patches + 1 CLS token, 768 embed_dim
        assert result.shape == expected_shape

    def test_gradient_computation_benchmark(self, benchmark):
        """Benchmark gradient computation performance."""
        model = ViTPerception(num_classes=10, img_size=224)
        criterion = torch.nn.CrossEntropyLoss()

        sample_input = torch.randn(4, 3, 224, 224)
        sample_targets = torch.randint(0, 10, (4,))

        def compute_gradients():
            model.zero_grad()
            outputs = model(sample_input)
            loss = criterion(outputs, sample_targets)
            loss.backward()

            # Return gradient norms for verification
            grad_norms = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norms[name] = param.grad.norm().item()

            return grad_norms

        model.train()
        result = benchmark(compute_gradients)

        # Verify gradients were computed
        assert isinstance(result, dict)
        assert len(result) > 0
        assert all(isinstance(norm, float) for norm in result.values())

    @pytest.mark.slow
    def test_long_training_stability_benchmark(self):
        """Benchmark training stability over extended periods."""
        model = ViTPerception(num_classes=10, img_size=224)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Training configuration
        num_steps = 100
        batch_size = 4

        # Track metrics
        losses = []
        step_times = []
        memory_usage = []

        model.train()
        process = psutil.Process()

        for step in range(num_steps):
            start_time = time.time()

            # Generate batch
            sample_input = torch.randn(batch_size, 3, 224, 224)
            sample_targets = torch.randint(0, 10, (batch_size,))

            # Training step
            optimizer.zero_grad()
            outputs = model(sample_input)
            loss = criterion(outputs, sample_targets)
            loss.backward()
            optimizer.step()

            # Record metrics
            step_time = time.time() - start_time
            losses.append(loss.item())
            step_times.append(step_time)
            memory_usage.append(process.memory_info().rss / 1024 / 1024)  # MB

        # Analyze stability
        avg_loss = sum(losses) / len(losses)
        avg_step_time = sum(step_times) / len(step_times)
        avg_memory = sum(memory_usage) / len(memory_usage)

        # Check for memory leaks (memory should be relatively stable)
        memory_growth = memory_usage[-1] - memory_usage[10]  # Compare end to early stage

        stability_report = {
            "avg_loss": avg_loss,
            "avg_step_time_s": avg_step_time,
            "avg_memory_mb": avg_memory,
            "memory_growth_mb": memory_growth,
            "final_loss": losses[-1],
            "loss_std": torch.tensor(losses).std().item()
        }

        # Stability assertions
        assert avg_loss > 0
        assert avg_step_time < 10.0  # Should complete in reasonable time
        assert memory_growth < 100  # Memory growth should be limited (< 100MB)

        print(f"Training stability report: {stability_report}")

    def test_model_size_benchmark(self):
        """Benchmark model size and parameter count."""
        model = ViTPerception(num_classes=1000, img_size=224)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Estimate model size (rough approximation)
        param_size_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32

        # Get model memory footprint
        model_size_bytes = sum(p.element_size() * p.nelement() for p in model.parameters())
        model_size_mb = model_size_bytes / (1024 * 1024)

        size_report = {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "estimated_size_mb": param_size_mb,
            "actual_size_mb": model_size_mb
        }

        # Size assertions for ViT-Base
        assert 80_000_000 < total_params < 90_000_000  # ~86M parameters
        assert total_params == trainable_params  # All should be trainable
        assert 300 < model_size_mb < 400  # ~350MB for ViT-Base

        print(f"Model size report: {size_report}")

    @pytest.mark.parametrize("precision", ["float32", "float16"])
    def test_mixed_precision_benchmark(self, benchmark, precision):
        """Benchmark performance with different precisions."""
        model = ViTPerception(num_classes=1000, img_size=224)
        model.eval()

        sample_input = torch.randn(4, 3, 224, 224)

        if precision == "float16":
            model = model.half()
            sample_input = sample_input.half()

        def inference_with_precision():
            with torch.no_grad():
                return model(sample_input)

        # Warm up
        for _ in range(3):
            inference_with_precision()

        result = benchmark(inference_with_precision)

        assert result.shape == (4, 1000)

        # Check dtype
        if precision == "float16":
            assert result.dtype == torch.float16
        else:
            assert result.dtype == torch.float32
