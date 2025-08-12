"""Test suite for advanced federated learning components."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from fed_vit_autorl.federated.advanced_aggregation import (
    AttentionBasedAggregator,
    HierarchicalFedAggregator,
    AdversarialRobustAggregator,
    MetaFedAggregator,
    ClientProfile,
)
from fed_vit_autorl.models.advanced_vit import (
    AdaptiveViT,
    MultiModalPatchEmbedding,
    TemporalPatchEmbedding,
)
from fed_vit_autorl.autonomous.self_improving_system import (
    AutonomousOptimizer,
    SelfHealingSystem,
    OptimizationGoal,
    SystemState,
)


class TestAdvancedAggregation:
    """Test advanced aggregation algorithms."""
    
    def test_attention_based_aggregator(self):
        """Test attention-based federated aggregation."""
        aggregator = AttentionBasedAggregator()
        
        # Create mock client updates
        client_updates = [
            {"layer1.weight": torch.randn(10, 10), "layer1.bias": torch.randn(10)}
            for _ in range(3)
        ]
        
        # Create client profiles
        client_profiles = [
            ClientProfile(
                client_id=f"client_{i}",
                compute_capability=0.8,
                data_quality_score=0.9,
                network_latency=50.0,
                privacy_sensitivity=0.5,
                geographic_region="us-west",
                vehicle_type="sedan",
                driving_scenario_diversity=0.7,
            )
            for i in range(3)
        ]
        
        # Test aggregation
        result = aggregator.aggregate(client_updates, client_profiles)
        
        assert "layer1.weight" in result
        assert "layer1.bias" in result
        assert result["layer1.weight"].shape == (10, 10)
        assert result["layer1.bias"].shape == (10,)
    
    def test_hierarchical_aggregator(self):
        """Test hierarchical federated aggregation."""
        from fed_vit_autorl.federated.aggregation import FedAvgAggregator
        
        # Create regional aggregators
        region_aggregators = {
            "us-west": FedAvgAggregator(),
            "eu-central": FedAvgAggregator(),
        }
        global_aggregator = FedAvgAggregator()
        
        aggregator = HierarchicalFedAggregator(
            region_aggregators=region_aggregators,
            global_aggregator=global_aggregator,
        )
        
        # Mock client updates
        client_updates = [
            {"layer1.weight": torch.randn(5, 5)}
            for _ in range(4)
        ]
        
        # Client profiles with different regions
        client_profiles = [
            ClientProfile(
                client_id=f"client_{i}",
                compute_capability=0.8,
                data_quality_score=0.9,
                network_latency=50.0,
                privacy_sensitivity=0.5,
                geographic_region="us-west" if i < 2 else "eu-central",
                vehicle_type="sedan",
                driving_scenario_diversity=0.7,
            )
            for i in range(4)
        ]
        
        result = aggregator.aggregate(client_updates, client_profiles)
        
        assert "layer1.weight" in result
        assert result["layer1.weight"].shape == (5, 5)
    
    def test_adversarial_robust_aggregator(self):
        """Test adversarially robust aggregation."""
        aggregator = AdversarialRobustAggregator()
        
        # Create mix of normal and adversarial updates
        normal_updates = [
            {"layer1.weight": torch.randn(5, 5) * 0.1}
            for _ in range(3)
        ]
        
        # Adversarial update (large magnitude)
        adversarial_update = {"layer1.weight": torch.randn(5, 5) * 10.0}
        
        client_updates = normal_updates + [adversarial_update]
        
        client_profiles = [
            ClientProfile(
                client_id=f"client_{i}",
                compute_capability=0.8,
                data_quality_score=0.9 if i < 3 else 0.1,  # Low quality for adversarial
                network_latency=50.0,
                privacy_sensitivity=0.5,
                geographic_region="us-west",
                vehicle_type="sedan",
                driving_scenario_diversity=0.7,
            )
            for i in range(4)
        ]
        
        result = aggregator.aggregate(client_updates, client_profiles)
        
        assert "layer1.weight" in result
        # Result should be closer to normal updates (adversarial filtered out)
        assert torch.norm(result["layer1.weight"]) < 5.0


class TestAdvancedViT:
    """Test advanced Vision Transformer architectures."""
    
    def test_adaptive_vit(self):
        """Test adaptive Vision Transformer."""
        model = AdaptiveViT(
            img_size=224,
            patch_size=16,
            num_classes=10,
            embed_dim=256,
            depth=6,
            num_heads=8,
            enable_early_exit=True,
        )
        
        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        
        assert output.shape == (2, 10)
        
        # Test early exit
        output_with_exits = model(x, return_all_exits=True)
        
        assert isinstance(output_with_exits, dict)
        assert "final" in output_with_exits
        assert "early_exits" in output_with_exits
        assert output_with_exits["final"].shape == (2, 10)
    
    def test_multimodal_patch_embedding(self):
        """Test multi-modal patch embedding."""
        embedding = MultiModalPatchEmbedding(
            img_size=224,
            lidar_size=64,
            patch_size=16,
            embed_dim=256,
            fusion_type="early",
        )
        
        # Test forward pass
        rgb_img = torch.randn(2, 3, 224, 224)
        lidar_bev = torch.randn(2, 4, 64, 64)
        
        output = embedding(rgb_img, lidar_bev)
        
        expected_patches = (224 // 16) ** 2
        assert output.shape == (2, expected_patches, 256)
    
    def test_temporal_patch_embedding(self):
        """Test temporal patch embedding."""
        embedding = TemporalPatchEmbedding(
            img_size=224,
            patch_size=16,
            num_frames=8,
            temporal_patch_size=2,
            embed_dim=256,
            use_3d_patches=True,
        )
        
        # Test forward pass
        x = torch.randn(2, 8, 3, 224, 224)  # batch, time, channels, height, width
        
        output = embedding(x)
        
        # Calculate expected output shape
        spatial_patches = (224 // 16) ** 2
        temporal_patches = 8 // 2
        expected_patches = spatial_patches * temporal_patches
        
        assert output.shape == (2, expected_patches, 256)


class TestAutonomousOptimization:
    """Test autonomous optimization system."""
    
    @pytest.fixture
    def optimization_goals(self):
        """Create optimization goals for testing."""
        return [
            OptimizationGoal(
                name="accuracy",
                metric_name="accuracy",
                target_value=0.95,
                direction="maximize",
                weight=1.0,
            ),
            OptimizationGoal(
                name="latency",
                metric_name="latency_ms",
                target_value=100.0,
                direction="minimize",
                weight=0.5,
            ),
        ]
    
    def test_autonomous_optimizer_initialization(self, optimization_goals):
        """Test autonomous optimizer initialization."""
        optimizer = AutonomousOptimizer(optimization_goals)
        
        assert len(optimizer.goals) == 2
        assert optimizer.performance_predictor is not None
        assert optimizer.auto_nas is not None
        assert optimizer.hyperopt is not None
    
    def test_performance_prediction(self, optimization_goals):
        """Test performance prediction."""
        optimizer = AutonomousOptimizer(optimization_goals)
        
        config = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "model_depth": 12,
            "model_width": 768,
        }
        
        predictions = optimizer.predict_performance(config)
        
        assert isinstance(predictions, dict)
        assert "accuracy" in predictions
        assert "latency" in predictions
    
    def test_self_healing_system(self):
        """Test self-healing system."""
        healing_system = SelfHealingSystem()
        
        # Create a system state with problems
        system_state = SystemState(
            performance_metrics={"improvement_rate": 0.0001},  # Low improvement
            resource_utilization={"memory_usage": 0.95},  # High memory usage
            client_statistics={"participation_rate": 0.3},  # Low participation
            model_complexity={},
            communication_efficiency={"success_rate": 0.7},  # Low success rate
            timestamp=1000.0,
        )
        
        # Test anomaly detection
        anomaly = healing_system.detect_anomaly(system_state)
        assert anomaly is not None
        
        # Test healing action
        healing_action = healing_system.monitor_system_health(system_state)
        assert healing_action is not None
        assert "action" in healing_action


class TestIntegration:
    """Test integration between components."""
    
    def test_advanced_aggregation_with_vit(self):
        """Test using advanced aggregation with ViT models."""
        # Create ViT model
        model = AdaptiveViT(
            img_size=224,
            patch_size=16,
            num_classes=10,
            embed_dim=128,
            depth=4,
            num_heads=4,
        )
        
        # Get model parameters
        model_params = {name: param.clone() for name, param in model.named_parameters()}
        
        # Create multiple client updates (simulate federated learning)
        client_updates = []
        for i in range(3):
            # Simulate local training by adding noise
            update = {}
            for name, param in model_params.items():
                noise = torch.randn_like(param) * 0.01
                update[name] = param + noise
            client_updates.append(update)
        
        # Use attention-based aggregator
        aggregator = AttentionBasedAggregator()
        
        # Create client profiles
        client_profiles = [
            ClientProfile(
                client_id=f"client_{i}",
                compute_capability=0.8,
                data_quality_score=0.9,
                network_latency=50.0,
                privacy_sensitivity=0.5,
                geographic_region="us-west",
                vehicle_type="sedan",
                driving_scenario_diversity=0.7,
            )
            for i in range(3)
        ]
        
        # Aggregate
        aggregated_params = aggregator.aggregate(client_updates, client_profiles)
        
        # Load aggregated parameters back to model
        model.load_state_dict(aggregated_params, strict=False)
        
        # Test that model still works
        x = torch.randn(1, 3, 224, 224)
        output = model(x)
        
        assert output.shape == (1, 10)
    
    @patch('fed_vit_autorl.autonomous.self_improving_system.AutonomousOptimizer.save_checkpoint')
    def test_autonomous_optimization_workflow(self, mock_save, optimization_goals):
        """Test complete autonomous optimization workflow."""
        optimizer = AutonomousOptimizer(optimization_goals)
        
        # Create system state
        system_state = SystemState(
            performance_metrics={"accuracy": 0.8, "latency_ms": 120.0},
            resource_utilization={"cpu": 0.6, "memory": 0.4},
            client_statistics={"active_clients": 100},
            model_complexity={"parameters": 1000000},
            communication_efficiency={"bytes_transferred": 500000},
            timestamp=1000.0,
        )
        
        # Create performance history
        performance_history = [
            {
                "config": {"learning_rate": 0.001, "batch_size": 32},
                "metrics": {"accuracy": 0.75, "latency_ms": 100.0}
            },
            {
                "config": {"learning_rate": 0.002, "batch_size": 64},
                "metrics": {"accuracy": 0.78, "latency_ms": 110.0}
            },
        ]
        
        # Run optimization step
        optimized_config = optimizer.autonomous_optimization_step(
            system_state, performance_history
        )
        
        assert isinstance(optimized_config, dict)
        assert "learning_rate" in optimized_config
        
        # Test checkpoint saving was called
        mock_save.assert_not_called()  # Should not save immediately


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])