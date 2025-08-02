"""
Fed-ViT-AutoRL Test Configuration

This module provides pytest fixtures and configuration for the entire test suite.
It includes fixtures for models, data, federated learning components, and more.
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# Test configuration
@pytest.fixture(scope="session", autouse=True)
def test_config():
    """Configure test environment variables and settings."""
    os.environ["FED_VIT_AUTORL_ENV"] = "test"
    os.environ["LOG_LEVEL"] = "WARNING"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU for tests
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    yield
    
    # Cleanup after all tests
    for key in ["FED_VIT_AUTORL_ENV", "LOG_LEVEL"]:
        os.environ.pop(key, None)


@pytest.fixture
def device():
    """Get the device to use for testing (CPU only)."""
    return torch.device("cpu")


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# Data fixtures
@pytest.fixture
def sample_image_data():
    """Generate sample image data for testing."""
    batch_size = 4
    channels = 3
    height = 224
    width = 224
    return torch.randn(batch_size, channels, height, width)


@pytest.fixture
def sample_vit_data():
    """Generate sample Vision Transformer input data."""
    batch_size = 2
    channels = 3
    height = 384
    width = 384
    return torch.randn(batch_size, channels, height, width)


@pytest.fixture
def sample_detection_targets():
    """Generate sample object detection targets."""
    return [
        {
            "boxes": torch.tensor([[10, 10, 50, 50], [60, 60, 100, 100]]),
            "labels": torch.tensor([1, 2]),
            "scores": torch.tensor([0.9, 0.8])
        },
        {
            "boxes": torch.tensor([[20, 20, 40, 40]]),
            "labels": torch.tensor([1]),
            "scores": torch.tensor([0.95])
        }
    ]


@pytest.fixture
def sample_segmentation_targets():
    """Generate sample segmentation targets."""
    batch_size = 2
    height = 384
    width = 384
    num_classes = 10
    return torch.randint(0, num_classes, (batch_size, height, width))


class MockDataset(Dataset):
    """Mock dataset for testing."""
    
    def __init__(self, size: int = 100, image_size: Tuple[int, int] = (224, 224)):
        self.size = size
        self.image_size = image_size
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        image = torch.randn(3, *self.image_size)
        label = torch.randint(0, 10, (1,)).item()
        return image, label


@pytest.fixture
def mock_dataset():
    """Create a mock dataset for testing."""
    return MockDataset()


@pytest.fixture
def mock_dataloader(mock_dataset):
    """Create a mock dataloader for testing."""
    return DataLoader(mock_dataset, batch_size=4, shuffle=False)


# Model fixtures
@pytest.fixture
def mock_vit_model():
    """Create a mock Vision Transformer model for testing."""
    class MockViTModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_dim = 768
            self.num_heads = 12
            self.depth = 12
            self.patch_size = 16
            self.projection = nn.Linear(768, 1000)
            
        def forward(self, x):
            batch_size = x.shape[0]
            # Mock ViT output
            features = torch.randn(batch_size, 197, 768)  # 197 = 196 patches + 1 cls token
            logits = self.projection(features[:, 0])  # Use CLS token
            return logits
            
        def get_features(self, x):
            batch_size = x.shape[0]
            return torch.randn(batch_size, 197, 768)
    
    return MockViTModel()


@pytest.fixture
def mock_detection_head():
    """Create a mock detection head for testing."""
    class MockDetectionHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = nn.Linear(768, 91)  # COCO classes
            self.bbox_predictor = nn.Linear(768, 4)
            
        def forward(self, features):
            batch_size = features.shape[0]
            return {
                "pred_logits": torch.randn(batch_size, 100, 91),
                "pred_boxes": torch.randn(batch_size, 100, 4)
            }
    
    return MockDetectionHead()


@pytest.fixture
def mock_rl_policy():
    """Create a mock RL policy for testing."""
    class MockRLPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = nn.Linear(768, 4)  # 4 actions: throttle, brake, steer, gear
            self.critic = nn.Linear(768, 1)
            
        def forward(self, state):
            batch_size = state.shape[0] if state.dim() > 1 else 1
            if state.dim() == 1:
                state = state.unsqueeze(0)
            
            action_logits = self.actor(state)
            value = self.critic(state)
            
            return {
                "action_logits": action_logits,
                "value": value
            }
    
    return MockRLPolicy()


# Federated learning fixtures
@pytest.fixture
def mock_federated_client():
    """Create a mock federated learning client."""
    class MockFederatedClient:
        def __init__(self, client_id: int, model: nn.Module):
            self.client_id = client_id
            self.model = model
            self.local_data = MockDataset(size=50)
            
        def local_update(self, num_epochs: int = 1):
            """Simulate local training."""
            return {
                "client_id": self.client_id,
                "num_samples": len(self.local_data),
                "loss": torch.tensor(np.random.uniform(0.1, 2.0)),
                "model_state_dict": self.model.state_dict()
            }
            
        def set_model_parameters(self, state_dict):
            """Set model parameters from global model."""
            self.model.load_state_dict(state_dict)
    
    return MockFederatedClient


@pytest.fixture
def mock_federated_server():
    """Create a mock federated learning server."""
    class MockFederatedServer:
        def __init__(self, global_model: nn.Module):
            self.global_model = global_model
            self.round = 0
            
        def aggregate_updates(self, client_updates: List[Dict]):
            """Simulate FedAvg aggregation."""
            self.round += 1
            # Simple averaging simulation
            return {
                "round": self.round,
                "num_clients": len(client_updates),
                "global_loss": torch.tensor(np.mean([u["loss"].item() for u in client_updates]))
            }
            
        def get_global_model_state(self):
            """Get global model state dict."""
            return self.global_model.state_dict()
    
    return MockFederatedServer


# Privacy and security fixtures
@pytest.fixture
def mock_differential_privacy():
    """Create a mock differential privacy mechanism."""
    class MockDifferentialPrivacy:
        def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
            self.epsilon = epsilon
            self.delta = delta
            
        def add_noise(self, tensor: torch.Tensor, sensitivity: float = 1.0):
            """Add Gaussian noise to tensor."""
            noise_scale = sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
            noise = torch.normal(0, noise_scale, tensor.shape)
            return tensor + noise
            
        def privatize_gradients(self, gradients: Dict[str, torch.Tensor]):
            """Add noise to gradients."""
            private_gradients = {}
            for name, grad in gradients.items():
                private_gradients[name] = self.add_noise(grad)
            return private_gradients
    
    return MockDifferentialPrivacy()


# Simulation fixtures
@pytest.fixture
def mock_carla_env():
    """Create a mock CARLA environment."""
    class MockCarlaEnv:
        def __init__(self):
            self.action_space_size = 4
            self.observation_shape = (3, 224, 224)
            self.num_vehicles = 1
            
        def reset(self):
            """Reset environment."""
            return torch.randn(*self.observation_shape)
            
        def step(self, action):
            """Take environment step."""
            obs = torch.randn(*self.observation_shape)
            reward = torch.tensor(np.random.uniform(-1, 1))
            done = np.random.random() < 0.1
            info = {"collision": False, "lane_violation": False}
            return obs, reward, done, info
    
    return MockCarlaEnv()


# Performance and benchmarking fixtures
@pytest.fixture
def benchmark_config():
    """Configuration for benchmarking tests."""
    return {
        "num_iterations": 10,
        "warmup_iterations": 2,
        "batch_sizes": [1, 4, 8],
        "input_sizes": [(224, 224), (384, 384)],
        "devices": ["cpu"]  # Only CPU for CI
    }


# Edge deployment fixtures
@pytest.fixture
def mock_edge_optimizer():
    """Create a mock edge optimization engine."""
    class MockEdgeOptimizer:
        def __init__(self):
            self.supported_formats = ["onnx", "trt", "quantized"]
            
        def optimize_model(self, model: nn.Module, target_format: str = "onnx"):
            """Simulate model optimization."""
            if target_format not in self.supported_formats:
                raise ValueError(f"Unsupported format: {target_format}")
                
            return {
                "optimized_model": model,
                "compression_ratio": np.random.uniform(2, 10),
                "latency_improvement": np.random.uniform(1.2, 5.0),
                "accuracy_retention": np.random.uniform(0.95, 0.99)
            }
    
    return MockEdgeOptimizer()


# Parameterized test fixtures
@pytest.fixture(params=["fedavg", "fedprox", "scaffold"])
def aggregation_algorithm(request):
    """Parameterized fixture for different aggregation algorithms."""
    return request.param


@pytest.fixture(params=[1, 5, 10])
def num_clients(request):
    """Parameterized fixture for different numbers of clients."""
    return request.param


@pytest.fixture(params=["differential_privacy", "secure_aggregation"])
def privacy_mechanism(request):
    """Parameterized fixture for different privacy mechanisms."""
    return request.param


# Skip markers for different test categories
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "e2e: marks tests as end-to-end tests")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "carla: marks tests that require CARLA simulator")
    config.addinivalue_line("markers", "benchmark: marks tests as benchmarks")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add slow marker to tests that might be slow
        if "benchmark" in item.nodeid or "performance" in item.nodeid:
            item.add_marker(pytest.mark.slow)
            
        # Add integration marker to integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
            
        # Add e2e marker to end-to-end tests
        if "e2e" in item.nodeid:
            item.add_marker(pytest.mark.e2e)


# Cleanup fixture
@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup after each test."""
    yield
    
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Reset random seeds
    torch.manual_seed(42)
    np.random.seed(42)