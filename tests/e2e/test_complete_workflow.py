"""
End-to-end tests for complete Fed-ViT-AutoRL workflows.

Tests complete user workflows from start to finish, including:
- Training a model from scratch
- Federated learning with multiple clients
- Model optimization and deployment
- Performance validation
"""

import pytest
import torch
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from fed_vit_autorl.models.vit_perception import ViTPerception


@pytest.mark.e2e
class TestCompleteWorkflow:
    """End-to-end tests for complete Fed-ViT-AutoRL workflows."""
    
    @pytest.mark.slow
    def test_single_client_training_workflow(self, temp_dir, mock_dataset):
        """Test complete single-client training workflow."""
        # 1. Initialize model
        model = ViTPerception(num_classes=10, img_size=224)
        
        # 2. Setup training components
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        dataloader = torch.utils.data.DataLoader(mock_dataset, batch_size=4)
        
        # 3. Training loop
        model.train()
        epoch_losses = []
        
        for epoch in range(2):  # Short training for testing
            epoch_loss = 0.0
            batch_count = 0
            
            for batch_idx, (data, targets) in enumerate(dataloader):
                if batch_idx >= 5:  # Limit batches for testing
                    break
                
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
            avg_epoch_loss = epoch_loss / batch_count
            epoch_losses.append(avg_epoch_loss)
        
        # 4. Save model
        model_path = temp_dir / "trained_model.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch_losses': epoch_losses
        }, model_path)
        
        # 5. Load and validate model
        checkpoint = torch.load(model_path)
        new_model = ViTPerception(num_classes=10, img_size=224)
        new_model.load_state_dict(checkpoint['model_state_dict'])
        
        # 6. Test inference
        new_model.eval()
        sample_input = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            output = new_model(sample_input)
        
        # Assertions
        assert len(epoch_losses) == 2
        assert output.shape == (1, 10)
        assert model_path.exists()
        assert checkpoint['epoch_losses'] == epoch_losses
    
    @pytest.mark.slow
    def test_federated_learning_workflow(self, temp_dir, mock_federated_client, mock_federated_server):
        """Test complete federated learning workflow."""
        # 1. Initialize global model
        global_model = ViTPerception(num_classes=10, img_size=224)
        server = mock_federated_server(global_model)
        
        # 2. Create federated clients
        num_clients = 3
        clients = []
        for i in range(num_clients):
            client_model = ViTPerception(num_classes=10, img_size=224)
            client_model.load_state_dict(global_model.state_dict())
            client = mock_federated_client(i, client_model)
            clients.append(client)
        
        # 3. Federated training rounds
        num_rounds = 3
        global_losses = []
        
        for round_num in range(num_rounds):
            # Local training phase
            client_updates = []
            for client in clients:
                update = client.local_update(num_epochs=1)
                client_updates.append(update)
            
            # Aggregation phase
            aggregation_result = server.aggregate_updates(client_updates)
            global_losses.append(aggregation_result["global_loss"].item())
            
            # Distribution phase
            global_state = server.get_global_model_state()
            for client in clients:
                client.set_model_parameters(global_state)
        
        # 4. Save federated model
        federated_model_path = temp_dir / "federated_model.pth"
        torch.save({
            'global_model_state_dict': server.get_global_model_state(),
            'global_losses': global_losses,
            'num_rounds': num_rounds,
            'num_clients': num_clients
        }, federated_model_path)
        
        # 5. Load and test federated model
        checkpoint = torch.load(federated_model_path)
        final_model = ViTPerception(num_classes=10, img_size=224)
        final_model.load_state_dict(checkpoint['global_model_state_dict'])
        
        # 6. Validate final model
        final_model.eval()
        test_input = torch.randn(2, 3, 224, 224)
        
        with torch.no_grad():
            predictions = final_model(test_input)
        
        # Assertions
        assert len(global_losses) == num_rounds
        assert predictions.shape == (2, 10)
        assert federated_model_path.exists()
        assert checkpoint['num_rounds'] == num_rounds
        assert checkpoint['num_clients'] == num_clients
    
    def test_model_optimization_workflow(self, temp_dir, mock_edge_optimizer):
        """Test complete model optimization workflow for edge deployment."""
        # 1. Create pre-trained model
        model = ViTPerception(num_classes=10, img_size=224)
        model.eval()
        
        # 2. Baseline performance measurement
        sample_input = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            baseline_output = model(sample_input)
            baseline_time = 0.1  # Simulated inference time
        
        # 3. Model optimization
        optimization_results = {}
        optimization_formats = ["onnx", "quantized"]
        
        for format_type in optimization_formats:
            result = mock_edge_optimizer.optimize_model(model, format_type)
            optimization_results[format_type] = result
        
        # 4. Performance comparison
        performance_report = {
            "baseline": {
                "format": "pytorch",
                "inference_time": baseline_time,
                "model_size": sum(p.numel() for p in model.parameters()),
                "accuracy": 1.0  # Reference accuracy
            }
        }
        
        for format_type, result in optimization_results.items():
            performance_report[format_type] = {
                "format": format_type,
                "compression_ratio": result["compression_ratio"],
                "latency_improvement": result["latency_improvement"],
                "accuracy_retention": result["accuracy_retention"]
            }
        
        # 5. Save optimization report
        report_path = temp_dir / "optimization_report.json"
        import json
        with open(report_path, 'w') as f:
            json.dump(performance_report, f, indent=2, default=str)
        
        # Assertions
        assert len(optimization_results) == len(optimization_formats)
        assert all("compression_ratio" in result for result in optimization_results.values())
        assert all("latency_improvement" in result for result in optimization_results.values())
        assert report_path.exists()
    
    @pytest.mark.carla
    def test_simulation_integration_workflow(self, mock_carla_env):
        """Test complete CARLA simulation integration workflow."""
        # 1. Initialize perception model
        perception_model = ViTPerception(num_classes=91, img_size=224)  # COCO classes
        perception_model.eval()
        
        # 2. Initialize RL policy (mock)
        class MockRLPolicy:
            def __init__(self):
                self.action_space = 4  # throttle, brake, steer, gear
                
            def act(self, observation):
                # Simple random policy for testing
                return torch.rand(self.action_space)
        
        rl_policy = MockRLPolicy()
        
        # 3. Simulation loop
        env = mock_carla_env
        num_episodes = 2
        episode_rewards = []
        
        for episode in range(num_episodes):
            obs = env.reset()
            episode_reward = 0.0
            done = False
            step_count = 0
            max_steps = 10  # Limit steps for testing
            
            while not done and step_count < max_steps:
                # Perception inference
                with torch.no_grad():
                    if obs.dim() == 3:  # Add batch dimension
                        obs_batch = obs.unsqueeze(0)
                    else:
                        obs_batch = obs
                    
                    perception_output = perception_model(obs_batch)
                
                # RL action selection
                action = rl_policy.act(perception_output)
                
                # Environment step
                next_obs, reward, done, info = env.step(action)
                
                episode_reward += reward.item()
                obs = next_obs
                step_count += 1
            
            episode_rewards.append(episode_reward)
        
        # 4. Performance analysis
        simulation_report = {
            "num_episodes": num_episodes,
            "episode_rewards": episode_rewards,
            "average_reward": sum(episode_rewards) / len(episode_rewards),
            "perception_model_classes": 91,
            "max_steps_per_episode": max_steps
        }
        
        # Assertions
        assert len(episode_rewards) == num_episodes
        assert all(isinstance(reward, float) for reward in episode_rewards)
        assert "average_reward" in simulation_report
    
    def test_privacy_preserving_workflow(self, mock_differential_privacy, temp_dir):
        """Test complete privacy-preserving training workflow."""
        # 1. Initialize model and privacy mechanism
        model = ViTPerception(num_classes=10, img_size=224)
        privacy_mechanism = mock_differential_privacy
        
        # 2. Setup private training
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        # 3. Private training loop
        num_batches = 5
        privacy_spent = []
        
        for batch_idx in range(num_batches):
            # Generate sample data
            sample_data = torch.randn(4, 3, 224, 224)
            sample_targets = torch.randint(0, 10, (4,))
            
            # Forward pass
            outputs = model(sample_data)
            loss = criterion(outputs, sample_targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Apply privacy to gradients
            private_gradients = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    private_grad = privacy_mechanism.add_noise(param.grad, sensitivity=1.0)
                    private_gradients[name] = private_grad
                    param.grad.data = private_grad
            
            # Optimizer step with private gradients
            optimizer.step()
            
            # Track privacy budget (simplified)
            privacy_spent.append(privacy_mechanism.epsilon)
        
        # 4. Privacy accounting
        total_privacy_spent = sum(privacy_spent)
        privacy_report = {
            "epsilon": privacy_mechanism.epsilon,
            "delta": privacy_mechanism.delta,
            "num_batches": num_batches,
            "total_privacy_spent": total_privacy_spent,
            "privacy_per_batch": privacy_mechanism.epsilon
        }
        
        # 5. Save privacy report
        privacy_report_path = temp_dir / "privacy_report.json"
        import json
        with open(privacy_report_path, 'w') as f:
            json.dump(privacy_report, f, indent=2, default=str)
        
        # Assertions
        assert len(privacy_spent) == num_batches
        assert total_privacy_spent > 0
        assert privacy_report_path.exists()
        assert privacy_report["epsilon"] == privacy_mechanism.epsilon
    
    @pytest.mark.slow
    def test_complete_pipeline_workflow(self, temp_dir):
        """Test the complete Fed-ViT-AutoRL pipeline from training to deployment."""
        # This test combines multiple workflows to test the complete pipeline
        
        # 1. Model initialization
        model = ViTPerception(num_classes=10, img_size=224)
        initial_state = model.state_dict().copy()
        
        # 2. Create mock training data
        train_data = []
        for _ in range(20):  # Small dataset for testing
            image = torch.randn(3, 224, 224)
            label = torch.randint(0, 10, (1,)).item()
            train_data.append((image, label))
        
        # 3. Training phase
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        for epoch in range(2):
            for image, label in train_data[:10]:  # Use subset for speed
                optimizer.zero_grad()
                output = model(image.unsqueeze(0))
                loss = criterion(output, torch.tensor([label]))
                loss.backward()
                optimizer.step()
        
        # 4. Validation phase
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for image, label in train_data[10:]:  # Use remaining data for validation
                output = model(image.unsqueeze(0))
                predicted = output.argmax(dim=1).item()
                total += 1
                if predicted == label:
                    correct += 1
        
        accuracy = correct / total if total > 0 else 0.0
        
        # 5. Model saving
        checkpoint_path = temp_dir / "final_checkpoint.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracy': accuracy,
            'training_complete': True
        }, checkpoint_path)
        
        # 6. Deployment preparation (simulation)
        deployment_model = ViTPerception(num_classes=10, img_size=224)
        checkpoint = torch.load(checkpoint_path)
        deployment_model.load_state_dict(checkpoint['model_state_dict'])
        deployment_model.eval()
        
        # 7. Inference test
        test_image = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            inference_output = deployment_model(test_image)
            prediction = inference_output.argmax(dim=1).item()
        
        # 8. Pipeline summary
        pipeline_summary = {
            "training_completed": True,
            "model_saved": checkpoint_path.exists(),
            "final_accuracy": accuracy,
            "inference_shape": list(inference_output.shape),
            "prediction": prediction,
            "model_parameters": sum(p.numel() for p in model.parameters())
        }
        
        # Assertions for complete pipeline
        assert pipeline_summary["training_completed"]
        assert pipeline_summary["model_saved"]
        assert pipeline_summary["inference_shape"] == [1, 10]
        assert 0 <= pipeline_summary["prediction"] <= 9
        assert pipeline_summary["model_parameters"] > 0
        
        # Verify model has been trained (parameters changed)
        final_state = model.state_dict()
        parameters_changed = False
        for key in initial_state:
            if not torch.allclose(initial_state[key], final_state[key], atol=1e-6):
                parameters_changed = True
                break
        
        assert parameters_changed, "Model parameters should change during training"