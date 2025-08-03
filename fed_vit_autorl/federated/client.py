"""Federated learning client implementation."""

import time
import logging
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
import numpy as np

from .privacy import DifferentialPrivacy
from .communication import GradientCompressor


logger = logging.getLogger(__name__)


class FederatedClient:
    """Federated learning client for vehicle-side training.
    
    Handles local training, privacy preservation, and communication
    with the federated server.
    """
    
    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        optimizer: optim.Optimizer,
        device: str = "cpu",
        privacy_budget: float = 1.0,
        compression_ratio: float = 0.1,
        local_epochs: int = 5,
    ):
        """Initialize federated client.
        
        Args:
            client_id: Unique identifier for this client
            model: Local model to train
            optimizer: Optimizer for local training
            device: Device to run computations on
            privacy_budget: Differential privacy epsilon parameter
            compression_ratio: Gradient compression ratio
            local_epochs: Number of local training epochs per round
        """
        self.client_id = client_id
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.local_epochs = local_epochs
        
        # Privacy mechanism
        self.privacy = DifferentialPrivacy(
            epsilon=privacy_budget,
            delta=1e-5,
            sensitivity=1.0,
        )
        
        # Communication efficiency
        self.compressor = GradientCompressor(
            method="top_k",
            compression_ratio=compression_ratio,
        )
        
        # Training state
        self.round_number = 0
        self.local_data_size = 0
        self.training_loss = []
        self.communication_cost = 0.0
        
        logger.info(f"Initialized federated client {client_id}")
    
    def set_global_model(self, global_params: Dict[str, torch.Tensor]) -> None:
        """Update local model with global parameters.
        
        Args:
            global_params: Global model parameters from server
        """
        # Load global parameters into local model
        self.model.load_state_dict(global_params)
        logger.debug(f"Client {self.client_id} updated with global model")
    
    def local_train(
        self,
        train_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        epochs: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Perform local training on client data.
        
        Args:
            train_loader: DataLoader for local training data
            criterion: Loss function
            epochs: Number of local epochs (defaults to self.local_epochs)
            
        Returns:
            Training metrics and statistics
        """
        if epochs is None:
            epochs = self.local_epochs
            
        self.model.train()
        epoch_losses = []
        
        # Store initial parameters
        initial_params = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
        }
        
        # Local training loop
        for epoch in range(epochs):
            batch_losses = []
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                self.optimizer.step()
                
                batch_losses.append(loss.item())
            
            epoch_loss = np.mean(batch_losses)
            epoch_losses.append(epoch_loss)
            
            logger.debug(
                f"Client {self.client_id} Epoch {epoch+1}/{epochs}: "
                f"Loss = {epoch_loss:.6f}"
            )
        
        # Calculate parameter update
        final_params = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
        }
        
        param_update = {
            name: final_params[name] - initial_params[name]
            for name in initial_params.keys()
        }
        
        # Update statistics
        self.local_data_size = len(train_loader.dataset)
        self.training_loss.extend(epoch_losses)
        
        return {
            "param_update": param_update,
            "num_samples": self.local_data_size,
            "loss": np.mean(epoch_losses),
            "epochs": epochs,
        }
    
    def get_model_update(
        self,
        train_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        apply_privacy: bool = True,
        compress_gradients: bool = True,
    ) -> Dict[str, Any]:
        """Get privacy-preserving, compressed model update.
        
        Args:
            train_loader: Local training data
            criterion: Loss function
            apply_privacy: Whether to apply differential privacy
            compress_gradients: Whether to compress gradients
            
        Returns:
            Processed model update for server
        """
        # Perform local training
        training_result = self.local_train(train_loader, criterion)
        param_update = training_result["param_update"]
        
        # Apply differential privacy
        if apply_privacy:
            param_update = self.privacy.privatize_gradients(param_update)
            logger.debug(f"Applied differential privacy to client {self.client_id}")
        
        # Compress gradients
        if compress_gradients:
            compressed_update, compression_info = self.compressor.compress(param_update)
            param_update = compressed_update
            self.communication_cost += compression_info.get("compressed_size", 0)
            logger.debug(
                f"Compressed gradients for client {self.client_id}: "
                f"ratio={compression_info.get('compression_ratio', 0):.3f}"
            )
        
        self.round_number += 1
        
        return {
            "client_id": self.client_id,
            "param_update": param_update,
            "num_samples": training_result["num_samples"],
            "loss": training_result["loss"],
            "round": self.round_number,
            "privacy_applied": apply_privacy,
            "compressed": compress_gradients,
        }
    
    def evaluate(self, test_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Evaluate local model on test data.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                # Calculate loss
                loss = nn.CrossEntropyLoss()(output, target)
                total_loss += loss.item()
                
                # Calculate accuracy
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(test_loader)
        
        return {
            "accuracy": accuracy,
            "loss": avg_loss,
            "num_samples": total,
        }
    
    def get_client_info(self) -> Dict[str, Any]:
        """Get client information and statistics.
        
        Returns:
            Client metadata and performance metrics
        """
        return {
            "client_id": self.client_id,
            "round_number": self.round_number,
            "local_data_size": self.local_data_size,
            "avg_training_loss": np.mean(self.training_loss) if self.training_loss else 0.0,
            "communication_cost": self.communication_cost,
            "device": self.device,
            "privacy_budget": self.privacy.epsilon,
        }
    
    def reset_privacy_budget(self, new_epsilon: float) -> None:
        """Reset differential privacy budget.
        
        Args:
            new_epsilon: New privacy budget
        """
        self.privacy = DifferentialPrivacy(
            epsilon=new_epsilon,
            delta=self.privacy.delta,
            sensitivity=self.privacy.sensitivity,
        )
        logger.info(f"Reset privacy budget for client {self.client_id} to ε={new_epsilon}")


class VehicleClient(FederatedClient):
    """Specialized federated client for autonomous vehicles.
    
    Extends base client with vehicle-specific functionality like
    sensor data handling, driving scenario tracking, and safety constraints.
    """
    
    def __init__(
        self,
        vehicle_id: str,
        model: nn.Module,
        optimizer: optim.Optimizer,
        device: str = "cpu",
        **kwargs
    ):
        """Initialize vehicle client.
        
        Args:
            vehicle_id: Unique vehicle identifier
            model: Perception model
            optimizer: Model optimizer
            device: Compute device
            **kwargs: Additional arguments for base client
        """
        super().__init__(vehicle_id, model, optimizer, device, **kwargs)
        
        # Vehicle-specific state
        self.driving_scenarios = []
        self.sensor_data_stats = {}
        self.safety_violations = 0
        self.miles_driven = 0.0
        
        logger.info(f"Initialized vehicle client {vehicle_id}")
    
    def record_driving_scenario(
        self,
        scenario_type: str,
        performance_metrics: Dict[str, float],
        safety_critical: bool = False,
    ) -> None:
        """Record driving scenario for learning.
        
        Args:
            scenario_type: Type of driving scenario (e.g., "highway", "urban")
            performance_metrics: Performance metrics for this scenario
            safety_critical: Whether this was a safety-critical scenario
        """
        scenario = {
            "type": scenario_type,
            "timestamp": time.time(),
            "metrics": performance_metrics,
            "safety_critical": safety_critical,
            "round": self.round_number,
        }
        
        self.driving_scenarios.append(scenario)
        
        if safety_critical:
            self.safety_violations += 1
            logger.warning(
                f"Safety violation recorded for vehicle {self.client_id}: {scenario_type}"
            )
    
    def get_vehicle_stats(self) -> Dict[str, Any]:
        """Get vehicle-specific statistics.
        
        Returns:
            Vehicle performance and safety statistics
        """
        base_stats = self.get_client_info()
        
        # Calculate scenario statistics
        scenario_types = [s["type"] for s in self.driving_scenarios]
        scenario_counts = {
            scenario_type: scenario_types.count(scenario_type)
            for scenario_type in set(scenario_types)
        }
        
        vehicle_stats = {
            **base_stats,
            "miles_driven": self.miles_driven,
            "safety_violations": self.safety_violations,
            "scenario_counts": scenario_counts,
            "total_scenarios": len(self.driving_scenarios),
            "safety_rate": 1 - (self.safety_violations / max(1, len(self.driving_scenarios))),
        }
        
        return vehicle_stats
    
    def update_mileage(self, miles: float) -> None:
        """Update vehicle mileage.
        
        Args:
            miles: Miles driven since last update
        """
        self.miles_driven += miles