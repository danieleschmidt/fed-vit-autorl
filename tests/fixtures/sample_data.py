"""
Sample data generators for testing Fed-ViT-AutoRL components.

This module provides utilities to generate realistic test data for:
- Vision Transformer inputs
- Autonomous driving scenarios
- Federated learning simulations
- Edge deployment testing
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class VisionDataGenerator:
    """Generate sample vision data for testing."""
    
    @staticmethod
    def generate_image_batch(
        batch_size: int = 4,
        channels: int = 3,
        height: int = 224,
        width: int = 224,
        normalize: bool = True
    ) -> torch.Tensor:
        """Generate a batch of random images."""
        images = torch.randn(batch_size, channels, height, width)
        
        if normalize:
            # Normalize to [0, 1] range like real images
            images = (images - images.min()) / (images.max() - images.min())
        
        return images
    
    @staticmethod
    def generate_detection_targets(
        batch_size: int,
        max_objects_per_image: int = 5,
        num_classes: int = 91,  # COCO classes
        image_size: Tuple[int, int] = (224, 224)
    ) -> List[Dict[str, torch.Tensor]]:
        """Generate sample object detection targets."""
        targets = []
        
        for _ in range(batch_size):
            num_objects = np.random.randint(1, max_objects_per_image + 1)
            
            # Generate random bounding boxes
            boxes = []
            for _ in range(num_objects):
                # Ensure boxes are within image bounds
                x1 = np.random.uniform(0, image_size[1] * 0.8)
                y1 = np.random.uniform(0, image_size[0] * 0.8)
                x2 = np.random.uniform(x1 + 10, image_size[1])
                y2 = np.random.uniform(y1 + 10, image_size[0])
                boxes.append([x1, y1, x2, y2])
            
            # Generate random labels and scores
            labels = torch.randint(1, num_classes, (num_objects,))
            scores = torch.rand(num_objects) * 0.5 + 0.5  # Scores between 0.5 and 1.0
            
            targets.append({
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": labels,
                "scores": scores
            })
        
        return targets
    
    @staticmethod
    def generate_segmentation_masks(
        batch_size: int,
        num_classes: int = 19,  # Cityscapes classes
        height: int = 224,
        width: int = 224
    ) -> torch.Tensor:
        """Generate sample segmentation masks."""
        masks = torch.randint(0, num_classes, (batch_size, height, width))
        return masks


class FederatedDataGenerator:
    """Generate sample data for federated learning scenarios."""
    
    @staticmethod
    def generate_client_datasets(
        num_clients: int,
        samples_per_client: int = 100,
        num_classes: int = 10,
        iid: bool = True,
        alpha: float = 0.5
    ) -> Dict[int, List[Tuple[torch.Tensor, int]]]:
        """Generate datasets for federated clients."""
        client_datasets = {}
        
        if iid:
            # IID data distribution - each client gets similar data distribution
            for client_id in range(num_clients):
                dataset = []
                for _ in range(samples_per_client):
                    image = torch.randn(3, 224, 224)
                    label = np.random.randint(0, num_classes)
                    dataset.append((image, label))
                client_datasets[client_id] = dataset
        else:
            # Non-IID data distribution using Dirichlet distribution
            # Each client has different class distributions
            class_probs = np.random.dirichlet([alpha] * num_classes, num_clients)
            
            for client_id in range(num_clients):
                dataset = []
                client_class_prob = class_probs[client_id]
                
                for _ in range(samples_per_client):
                    image = torch.randn(3, 224, 224)
                    label = np.random.choice(num_classes, p=client_class_prob)
                    dataset.append((image, label))
                
                client_datasets[client_id] = dataset
        
        return client_datasets
    
    @staticmethod
    def generate_client_capabilities(
        num_clients: int
    ) -> Dict[int, Dict[str, any]]:
        """Generate heterogeneous client capabilities."""
        capabilities = {}
        
        device_types = ["high_end", "mid_range", "edge", "mobile"]
        compute_levels = {"high_end": 1.0, "mid_range": 0.7, "edge": 0.4, "mobile": 0.2}
        memory_levels = {"high_end": 16, "mid_range": 8, "edge": 4, "mobile": 2}
        
        for client_id in range(num_clients):
            device_type = np.random.choice(device_types)
            
            capabilities[client_id] = {
                "device_type": device_type,
                "compute_capability": compute_levels[device_type],
                "memory_gb": memory_levels[device_type],
                "bandwidth_mbps": np.random.uniform(10, 1000),
                "battery_level": np.random.uniform(0.2, 1.0) if device_type == "mobile" else 1.0,
                "availability": np.random.uniform(0.7, 1.0)
            }
        
        return capabilities


class AutonomousDrivingDataGenerator:
    """Generate sample data for autonomous driving scenarios."""
    
    @staticmethod
    def generate_driving_scenario(
        scenario_type: str = "highway",
        duration_seconds: int = 30,
        fps: int = 30
    ) -> Dict[str, torch.Tensor]:
        """Generate a complete driving scenario."""
        num_frames = duration_seconds * fps
        
        # Camera images
        images = torch.randn(num_frames, 3, 224, 224)
        
        # Vehicle state (speed, acceleration, steering angle, etc.)
        vehicle_states = torch.randn(num_frames, 10)  # 10 state variables
        
        # Actions (throttle, brake, steering, gear)
        actions = torch.randn(num_frames, 4)
        
        # Rewards (safety, efficiency, comfort)
        if scenario_type == "highway":
            base_reward = 0.8
        elif scenario_type == "city":
            base_reward = 0.6
        elif scenario_type == "parking":
            base_reward = 0.4
        else:
            base_reward = 0.5
        
        rewards = base_reward + torch.randn(num_frames) * 0.2
        
        # Events (collisions, lane changes, etc.)
        collision_events = torch.rand(num_frames) < 0.01  # 1% collision probability
        lane_change_events = torch.rand(num_frames) < 0.05  # 5% lane change probability
        
        return {
            "images": images,
            "vehicle_states": vehicle_states,
            "actions": actions,
            "rewards": rewards,
            "collisions": collision_events,
            "lane_changes": lane_change_events,
            "scenario_type": scenario_type,
            "duration": duration_seconds,
            "fps": fps
        }
    
    @staticmethod
    def generate_multi_vehicle_scenario(
        num_vehicles: int = 5,
        scenario_type: str = "intersection",
        duration_seconds: int = 20
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """Generate multi-vehicle driving scenario."""
        scenarios = {}
        
        for vehicle_id in range(num_vehicles):
            # Each vehicle has slightly different scenario characteristics
            vehicle_scenario = AutonomousDrivingDataGenerator.generate_driving_scenario(
                scenario_type=scenario_type,
                duration_seconds=duration_seconds
            )
            
            # Add vehicle-specific metadata
            vehicle_scenario["vehicle_id"] = vehicle_id
            vehicle_scenario["vehicle_type"] = np.random.choice(["sedan", "suv", "truck"])
            
            scenarios[vehicle_id] = vehicle_scenario
        
        return scenarios


class EdgeDeploymentDataGenerator:
    """Generate sample data for edge deployment testing."""
    
    @staticmethod
    def generate_performance_profiles(
        num_devices: int = 10
    ) -> Dict[str, Dict[str, float]]:
        """Generate performance profiles for different edge devices."""
        device_profiles = {}
        
        device_types = [
            "jetson_nano", "jetson_xavier_nx", "jetson_agx_xavier",
            "raspberry_pi_4", "intel_nuc", "edge_tpu"
        ]
        
        # Base performance characteristics
        base_specs = {
            "jetson_nano": {"compute": 0.5, "memory": 4, "power": 10},
            "jetson_xavier_nx": {"compute": 1.0, "memory": 8, "power": 15},
            "jetson_agx_xavier": {"compute": 2.0, "memory": 32, "power": 30},
            "raspberry_pi_4": {"compute": 0.3, "memory": 4, "power": 5},
            "intel_nuc": {"compute": 1.5, "memory": 16, "power": 25},
            "edge_tpu": {"compute": 3.0, "memory": 8, "power": 8}
        }
        
        for i in range(num_devices):
            device_type = np.random.choice(device_types)
            base_spec = base_specs[device_type]
            
            # Add some variation to base specs
            device_profiles[f"device_{i}"] = {
                "device_type": device_type,
                "compute_capability": base_spec["compute"] * np.random.uniform(0.8, 1.2),
                "memory_gb": base_spec["memory"],
                "power_watts": base_spec["power"] * np.random.uniform(0.9, 1.1),
                "inference_latency_ms": np.random.uniform(50, 200),
                "throughput_fps": np.random.uniform(10, 60),
                "accuracy_retention": np.random.uniform(0.90, 0.99)
            }
        
        return device_profiles
    
    @staticmethod
    def generate_optimization_results(
        model_name: str,
        optimization_techniques: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Generate mock optimization results for different techniques."""
        results = {}
        
        base_metrics = {
            "original_size_mb": 300,
            "original_latency_ms": 150,
            "original_accuracy": 0.95
        }
        
        # Define optimization impact
        technique_impacts = {
            "quantization_int8": {"size_ratio": 0.25, "latency_ratio": 0.6, "accuracy_loss": 0.02},
            "pruning_50": {"size_ratio": 0.5, "latency_ratio": 0.7, "accuracy_loss": 0.01},
            "pruning_75": {"size_ratio": 0.25, "latency_ratio": 0.5, "accuracy_loss": 0.03},
            "distillation": {"size_ratio": 0.3, "latency_ratio": 0.4, "accuracy_loss": 0.04},
            "tensorrt": {"size_ratio": 0.8, "latency_ratio": 0.3, "accuracy_loss": 0.005},
            "onnx": {"size_ratio": 0.9, "latency_ratio": 0.8, "accuracy_loss": 0.001}
        }
        
        for technique in optimization_techniques:
            if technique in technique_impacts:
                impact = technique_impacts[technique]
                
                # Add some randomness
                size_ratio = impact["size_ratio"] * np.random.uniform(0.9, 1.1)
                latency_ratio = impact["latency_ratio"] * np.random.uniform(0.9, 1.1)
                accuracy_loss = impact["accuracy_loss"] * np.random.uniform(0.8, 1.2)
                
                results[technique] = {
                    "optimized_size_mb": base_metrics["original_size_mb"] * size_ratio,
                    "optimized_latency_ms": base_metrics["original_latency_ms"] * latency_ratio,
                    "optimized_accuracy": base_metrics["original_accuracy"] - accuracy_loss,
                    "compression_ratio": 1.0 / size_ratio,
                    "speedup_ratio": 1.0 / latency_ratio,
                    "accuracy_retention": 1.0 - (accuracy_loss / base_metrics["original_accuracy"])
                }
        
        return results


class TestDataManager:
    """Manage test data creation and cleanup."""
    
    def __init__(self, temp_dir: Path):
        self.temp_dir = temp_dir
        self.created_files = []
    
    def create_sample_dataset(
        self,
        dataset_name: str,
        num_samples: int = 100,
        save_to_disk: bool = False
    ) -> List[Tuple[torch.Tensor, int]]:
        """Create and optionally save a sample dataset."""
        dataset = []
        
        for i in range(num_samples):
            image = VisionDataGenerator.generate_image_batch(
                batch_size=1, height=224, width=224
            ).squeeze(0)
            label = np.random.randint(0, 10)
            dataset.append((image, label))
        
        if save_to_disk:
            dataset_path = self.temp_dir / f"{dataset_name}.pt"
            torch.save(dataset, dataset_path)
            self.created_files.append(dataset_path)
        
        return dataset
    
    def create_model_checkpoint(
        self,
        model_name: str,
        num_classes: int = 10
    ) -> Path:
        """Create a sample model checkpoint."""
        from fed_vit_autorl.models.vit_perception import ViTPerception
        
        model = ViTPerception(num_classes=num_classes)
        
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "num_classes": num_classes,
                "img_size": 224,
                "patch_size": 16
            },
            "training_metadata": {
                "epochs": 10,
                "accuracy": 0.85,
                "loss": 0.5
            }
        }
        
        checkpoint_path = self.temp_dir / f"{model_name}_checkpoint.pth"
        torch.save(checkpoint, checkpoint_path)
        self.created_files.append(checkpoint_path)
        
        return checkpoint_path
    
    def cleanup(self):
        """Clean up created test files."""
        for file_path in self.created_files:
            if file_path.exists():
                file_path.unlink()
        self.created_files.clear()