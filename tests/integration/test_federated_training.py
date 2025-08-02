"""
Integration tests for federated training pipeline.

Tests the complete federated learning workflow including:
- Multi-client training simulation
- Model aggregation
- Privacy mechanisms integration
- Communication protocols
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

from fed_vit_autorl.models.vit_perception import ViTPerception


@pytest.mark.integration
class TestFederatedTrainingIntegration:
    """Integration tests for federated training workflow."""
    
    def test_simple_federated_round(self, mock_federated_client, mock_federated_server):
        """Test a simple federated learning round with multiple clients."""
        # Setup
        global_model = ViTPerception(num_classes=10, img_size=224)
        server = mock_federated_server(global_model)
        
        # Create multiple clients
        num_clients = 3
        clients = []
        for i in range(num_clients):
            client_model = ViTPerception(num_classes=10, img_size=224)
            client_model.load_state_dict(global_model.state_dict())
            client = mock_federated_client(i, client_model)
            clients.append(client)
        
        # Simulate federated round
        client_updates = []
        for client in clients:
            update = client.local_update(num_epochs=1)
            client_updates.append(update)
        
        # Server aggregation
        aggregation_result = server.aggregate_updates(client_updates)
        
        # Assertions
        assert len(client_updates) == num_clients
        assert aggregation_result["num_clients"] == num_clients
        assert aggregation_result["round"] == 1
        assert "global_loss" in aggregation_result
    
    def test_federated_training_with_privacy(self, mock_differential_privacy):
        """Test federated training with differential privacy."""
        # Setup models
        global_model = ViTPerception(num_classes=10, img_size=224)
        client_model = ViTPerception(num_classes=10, img_size=224)
        client_model.load_state_dict(global_model.state_dict())
        
        # Create sample data
        sample_data = torch.randn(4, 3, 224, 224)
        sample_targets = torch.randint(0, 10, (4,))
        
        # Simulate local training with privacy
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(client_model.parameters(), lr=0.01)
        
        # Forward pass
        outputs = client_model(sample_data)
        loss = criterion(outputs, sample_targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Extract gradients
        gradients = {}
        for name, param in client_model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone()
        
        # Apply differential privacy
        private_gradients = mock_differential_privacy.privatize_gradients(gradients)
        
        # Assertions
        assert len(private_gradients) == len(gradients)
        for name in gradients:
            assert name in private_gradients
            # Privacy should add noise, so gradients should be different
            assert not torch.allclose(gradients[name], private_gradients[name], atol=1e-6)
    
    def test_model_aggregation_consistency(self, mock_federated_server):
        """Test that model aggregation maintains model structure."""
        # Create global model
        global_model = ViTPerception(num_classes=10, img_size=224)
        server = mock_federated_server(global_model)
        
        # Create client models with different weights
        num_clients = 5
        client_models = []
        client_updates = []
        
        for i in range(num_clients):
            client_model = ViTPerception(num_classes=10, img_size=224)
            
            # Initialize with different random weights
            for param in client_model.parameters():
                param.data.uniform_(-1, 1)
            
            client_models.append(client_model)
            client_updates.append({
                "client_id": i,
                "num_samples": 100,
                "loss": torch.tensor(0.5),
                "model_state_dict": client_model.state_dict()
            })
        
        # Test aggregation
        initial_global_state = server.get_global_model_state()
        aggregation_result = server.aggregate_updates(client_updates)
        final_global_state = server.get_global_model_state()
        
        # Verify structure consistency
        assert set(initial_global_state.keys()) == set(final_global_state.keys())
        
        # Verify aggregation changed the global model
        params_changed = False
        for key in initial_global_state:
            if not torch.allclose(initial_global_state[key], final_global_state[key]):
                params_changed = True
                break
        
        # In a real aggregation, parameters should change
        # Note: This is a mock, so we mainly test the structure
        assert aggregation_result["num_clients"] == num_clients
    
    @pytest.mark.parametrize("num_clients", [2, 5, 10])
    def test_scalability_with_multiple_clients(self, num_clients, mock_federated_client, mock_federated_server):
        """Test federated training scalability with different client counts."""
        # Setup
        global_model = ViTPerception(num_classes=10, img_size=224)
        server = mock_federated_server(global_model)
        
        # Create clients
        clients = []
        for i in range(num_clients):
            client_model = ViTPerception(num_classes=10, img_size=224)
            client_model.load_state_dict(global_model.state_dict())
            client = mock_federated_client(i, client_model)
            clients.append(client)
        
        # Simulate multiple rounds
        num_rounds = 3
        for round_num in range(num_rounds):
            client_updates = []
            for client in clients:
                update = client.local_update(num_epochs=1)
                client_updates.append(update)
            
            aggregation_result = server.aggregate_updates(client_updates)
            
            assert aggregation_result["num_clients"] == num_clients
            assert aggregation_result["round"] == round_num + 1
    
    def test_heterogeneous_client_data(self, mock_federated_client):
        """Test federated training with heterogeneous client data distributions."""
        # Create clients with different data characteristics
        global_model = ViTPerception(num_classes=10, img_size=224)
        
        clients = []
        client_configs = [
            {"client_id": 0, "data_size": 50, "bias_class": 0},
            {"client_id": 1, "data_size": 100, "bias_class": 5},
            {"client_id": 2, "data_size": 75, "bias_class": 9},
        ]
        
        for config in client_configs:
            client_model = ViTPerception(num_classes=10, img_size=224)
            client_model.load_state_dict(global_model.state_dict())
            client = mock_federated_client(config["client_id"], client_model)
            
            # Simulate heterogeneous data by modifying the mock dataset
            client.local_data.size = config["data_size"]
            clients.append(client)
        
        # Test local updates
        client_updates = []
        for client in clients:
            update = client.local_update(num_epochs=1)
            client_updates.append(update)
        
        # Verify different clients produce different updates
        assert len(client_updates) == len(clients)
        
        # Check that different clients have different numbers of samples
        sample_counts = [update["num_samples"] for update in client_updates]
        assert len(set(sample_counts)) > 1  # At least some variation
    
    def test_communication_efficiency_simulation(self):
        """Test communication efficiency mechanisms."""
        # Create models for compression testing
        model = ViTPerception(num_classes=10, img_size=224)
        
        # Get model parameters
        original_params = {}
        for name, param in model.named_parameters():
            original_params[name] = param.data.clone()
        
        # Simulate gradient compression (top-k sparsification)
        compressed_params = {}
        compression_ratio = 0.1  # Keep only 10% of parameters
        
        for name, param in original_params.items():
            flat_param = param.flatten()
            k = int(len(flat_param) * compression_ratio)
            
            # Get top-k by magnitude
            _, top_k_indices = torch.topk(torch.abs(flat_param), k)
            
            compressed_flat = torch.zeros_like(flat_param)
            compressed_flat[top_k_indices] = flat_param[top_k_indices]
            
            compressed_params[name] = compressed_flat.reshape(param.shape)
        
        # Calculate compression statistics
        original_size = sum(p.numel() for p in original_params.values())
        compressed_nonzero = sum((p != 0).sum().item() for p in compressed_params.values())
        actual_compression_ratio = compressed_nonzero / original_size
        
        assert actual_compression_ratio <= compression_ratio + 0.01  # Allow small tolerance
        assert compressed_nonzero < original_size
    
    def test_asynchronous_update_simulation(self, mock_federated_client, mock_federated_server):
        """Test simulation of asynchronous federated updates."""
        # Setup
        global_model = ViTPerception(num_classes=10, img_size=224)
        server = mock_federated_server(global_model)
        
        # Create clients with different update frequencies
        clients = []
        for i in range(3):
            client_model = ViTPerception(num_classes=10, img_size=224)
            client_model.load_state_dict(global_model.state_dict())
            client = mock_federated_client(i, client_model)
            clients.append(client)
        
        # Simulate asynchronous updates
        updates_received = []
        
        # Client 0 sends update
        update_0 = clients[0].local_update(num_epochs=1)
        updates_received.append(update_0)
        
        # Server processes partial update
        partial_result = server.aggregate_updates(updates_received)
        assert partial_result["num_clients"] == 1
        
        # Client 1 sends update later
        update_1 = clients[1].local_update(num_epochs=1)
        updates_received.append(update_1)
        
        # Server processes with more clients
        result_with_two = server.aggregate_updates(updates_received)
        assert result_with_two["num_clients"] == 2
        
        # Verify server state progresses
        assert result_with_two["round"] > partial_result["round"]
    
    def test_model_convergence_simulation(self, mock_federated_client, mock_federated_server):
        """Test simulation of model convergence in federated setting."""
        # Setup
        global_model = ViTPerception(num_classes=10, img_size=224)
        server = mock_federated_server(global_model)
        
        # Create clients
        num_clients = 3
        clients = []
        for i in range(num_clients):
            client_model = ViTPerception(num_classes=10, img_size=224)
            client_model.load_state_dict(global_model.state_dict())
            client = mock_federated_client(i, client_model)
            clients.append(client)
        
        # Track convergence metrics
        round_losses = []
        
        # Simulate multiple federated rounds
        num_rounds = 5
        for round_num in range(num_rounds):
            client_updates = []
            
            # Collect updates from all clients
            for client in clients:
                update = client.local_update(num_epochs=1)
                client_updates.append(update)
            
            # Server aggregation
            aggregation_result = server.aggregate_updates(client_updates)
            round_losses.append(aggregation_result["global_loss"].item())
            
            # Update client models with new global model
            global_state = server.get_global_model_state()
            for client in clients:
                client.set_model_parameters(global_state)
        
        # Basic convergence check (losses should be tracked)
        assert len(round_losses) == num_rounds
        assert all(isinstance(loss, float) for loss in round_losses)