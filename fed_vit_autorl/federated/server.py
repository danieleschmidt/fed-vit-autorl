"""Federated learning server implementation."""

import time
import logging
from typing import Dict, List, Optional, Any, Callable
import torch
import torch.nn as nn
from collections import OrderedDict, defaultdict
import numpy as np
import asyncio
from threading import Lock

from .aggregation import FedAvgAggregator, FedProxAggregator, AdaptiveAggregator
from .privacy import SecureAggregator as CryptoSecureAggregator
from .communication import AsyncCommunicator


logger = logging.getLogger(__name__)


class FederatedServer:
    """Federated learning server for coordinating vehicle training.
    
    Manages global model state, client coordination, and aggregation
    of federated updates while ensuring privacy and efficiency.
    """
    
    def __init__(
        self,
        global_model: nn.Module,
        aggregation_method: str = "fedavg",
        min_clients: int = 10,
        max_clients: int = 1000,
        rounds: int = 1000,
        client_fraction: float = 0.1,
        device: str = "cpu",
    ):
        """Initialize federated server.
        
        Args:
            global_model: Global model to be trained
            aggregation_method: Aggregation algorithm ("fedavg", "fedprox", "adaptive")  
            min_clients: Minimum clients needed per round
            max_clients: Maximum clients to select per round
            rounds: Total number of federated rounds
            client_fraction: Fraction of clients to select each round
            device: Device for server computations
        """
        self.global_model = global_model.to(device)
        self.device = device
        self.rounds = rounds
        self.min_clients = min_clients
        self.max_clients = max_clients
        self.client_fraction = client_fraction
        
        # Initialize aggregator
        if aggregation_method == "fedavg":
            self.aggregator = FedAvgAggregator()
        elif aggregation_method == "fedprox":
            self.aggregator = FedProxAggregator(mu=0.01)
        elif aggregation_method == "adaptive":
            self.aggregator = AdaptiveAggregator()
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
        
        # Server state
        self.current_round = 0
        self.client_registry = {}
        self.round_history = []
        self.performance_metrics = defaultdict(list)
        
        # Async communication
        self.communicator = AsyncCommunicator()
        self.update_lock = Lock()
        
        logger.info(f"Initialized federated server with {aggregation_method} aggregation")
    
    def register_client(self, client_info: Dict[str, Any]) -> bool:
        """Register a new client with the server.
        
        Args:
            client_info: Client metadata and capabilities
            
        Returns:
            True if registration successful, False otherwise
        """
        client_id = client_info.get("client_id")
        if not client_id:
            logger.error("Client registration failed: missing client_id")
            return False
        
        self.client_registry[client_id] = {
            **client_info,
            "registered_at": time.time(),
            "last_seen": time.time(),
            "rounds_participated": 0,
            "avg_loss": float("inf"),
            "reliability_score": 1.0,
        }
        
        logger.info(f"Registered client {client_id}")
        return True
    
    def select_clients(self, available_clients: List[str]) -> List[str]:
        """Select clients for federated round.
        
        Args:
            available_clients: List of available client IDs
            
        Returns:
            Selected client IDs for this round
        """
        # Filter clients based on reliability and recent activity
        eligible_clients = []
        current_time = time.time()
        
        for client_id in available_clients:
            if client_id not in self.client_registry:
                continue
                
            client_info = self.client_registry[client_id]
            
            # Check if client is recently active (within last hour)
            if current_time - client_info["last_seen"] > 3600:
                continue
                
            # Check reliability score (minimum 0.5)
            if client_info["reliability_score"] < 0.5:
                continue
                
            eligible_clients.append(client_id)
        
        # Select subset based on client fraction
        num_select = min(
            max(self.min_clients, int(len(eligible_clients) * self.client_fraction)),
            min(self.max_clients, len(eligible_clients))
        )
        
        if num_select < self.min_clients:
            logger.warning(
                f"Only {num_select} eligible clients, need {self.min_clients}"
            )
            return eligible_clients
        
        # Prioritize clients with better reliability scores
        client_scores = [
            (client_id, self.client_registry[client_id]["reliability_score"])
            for client_id in eligible_clients
        ]
        client_scores.sort(key=lambda x: x[1], reverse=True)
        
        selected = [client_id for client_id, _ in client_scores[:num_select]]
        
        logger.info(f"Selected {len(selected)} clients for round {self.current_round}")
        return selected
    
    def get_global_model(self) -> Dict[str, torch.Tensor]:
        """Get current global model parameters.
        
        Returns:
            Global model state dict
        """
        return {
            name: param.clone().detach()
            for name, param in self.global_model.named_parameters()
        }
    
    def aggregate_updates(
        self,
        client_updates: List[Dict[str, Any]],
        round_number: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Aggregate client updates into global model.
        
        Args:
            client_updates: List of client update dictionaries
            round_number: Current round number (for staleness tracking)
            
        Returns:
            Updated global model parameters
        """
        if not client_updates:
            logger.warning("No client updates to aggregate")
            return self.get_global_model()
        
        with self.update_lock:
            # Extract parameter updates and weights
            param_updates = []
            client_weights = []
            staleness_info = []
            
            for update in client_updates:
                param_updates.append(update["param_update"])
                client_weights.append(update.get("num_samples", 1.0))
                
                # Calculate staleness (how many rounds behind)
                client_round = update.get("round", self.current_round)
                staleness = max(0, self.current_round - client_round)
                staleness_info.append(staleness)
                
                # Update client info
                client_id = update["client_id"]
                if client_id in self.client_registry:
                    self.client_registry[client_id]["last_seen"] = time.time()
                    self.client_registry[client_id]["rounds_participated"] += 1
                    self.client_registry[client_id]["avg_loss"] = update.get("loss", float("inf"))
            
            # Perform aggregation
            if isinstance(self.aggregator, AdaptiveAggregator):
                aggregated_params = self.aggregator.aggregate(
                    param_updates, client_weights, staleness_info
                )
            else:
                aggregated_params = self.aggregator.aggregate(param_updates, client_weights)
            
            # Update global model
            current_params = self.get_global_model()
            updated_params = {}
            
            for name in current_params.keys():
                if name in aggregated_params:
                    updated_params[name] = (
                        current_params[name] + aggregated_params[name]
                    ).to(self.device)
                else:
                    updated_params[name] = current_params[name]
            
            # Load updated parameters
            self.global_model.load_state_dict(updated_params)
            
            # Record round statistics
            self._record_round_stats(client_updates, aggregated_params)
            
            logger.info(
                f"Aggregated {len(client_updates)} client updates for round {self.current_round}"
            )
            
            return updated_params
    
    def _record_round_stats(
        self,
        client_updates: List[Dict[str, Any]],
        aggregated_params: Dict[str, torch.Tensor],
    ) -> None:
        """Record statistics for the current round.
        
        Args:
            client_updates: Client updates for this round
            aggregated_params: Aggregated parameter update
        """
        # Calculate round statistics
        client_losses = [update.get("loss", 0.0) for update in client_updates]
        total_samples = sum(update.get("num_samples", 0) for update in client_updates)
        
        # Calculate parameter update magnitude
        param_magnitude = sum(
            torch.norm(param).item() for param in aggregated_params.values()
        )
        
        round_stats = {
            "round": self.current_round,
            "timestamp": time.time(),
            "num_clients": len(client_updates),
            "total_samples": total_samples,
            "avg_client_loss": np.mean(client_losses),
            "std_client_loss": np.std(client_losses),
            "param_update_magnitude": param_magnitude,
        }
        
        self.round_history.append(round_stats)
        
        # Update performance metrics
        self.performance_metrics["client_losses"].extend(client_losses)
        self.performance_metrics["participation_rate"].append(
            len(client_updates) / len(self.client_registry)
        )
        self.performance_metrics["round_times"].append(time.time())
    
    def evaluate_global_model(
        self,
        test_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
    ) -> Dict[str, float]:
        """Evaluate global model on test data.
        
        Args:
            test_loader: Test data loader
            criterion: Loss function
            
        Returns:
            Evaluation metrics
        """
        self.global_model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                
                # Calculate loss
                loss = criterion(output, target)
                total_loss += loss.item()
                
                # Calculate accuracy
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else float("inf")
        
        metrics = {
            "accuracy": accuracy,
            "loss": avg_loss,
            "num_samples": total,
        }
        
        # Record evaluation metrics
        self.performance_metrics["global_accuracy"].append(accuracy)
        self.performance_metrics["global_loss"].append(avg_loss)
        
        logger.info(f"Global model evaluation: accuracy={accuracy:.4f}, loss={avg_loss:.6f}")
        return metrics
    
    def run_round(
        self,
        available_clients: List[str],
        client_updates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Execute a single federated learning round.
        
        Args:
            available_clients: List of available client IDs
            client_updates: Client updates for this round
            
        Returns:
            Round results and statistics
        """
        round_start_time = time.time()
        
        # Select clients for this round
        selected_clients = self.select_clients(available_clients)
        
        if len(selected_clients) < self.min_clients:
            logger.error(
                f"Insufficient clients for round {self.current_round}: "
                f"{len(selected_clients)} < {self.min_clients}"
            )
            return {"success": False, "reason": "insufficient_clients"}
        
        # Filter updates to selected clients
        selected_updates = [
            update for update in client_updates
            if update["client_id"] in selected_clients
        ]
        
        if not selected_updates:
            logger.error(f"No updates received from selected clients")
            return {"success": False, "reason": "no_updates"}
        
        # Aggregate updates
        try:
            updated_params = self.aggregate_updates(selected_updates, self.current_round)
        except Exception as e:
            logger.error(f"Aggregation failed: {e}")
            return {"success": False, "reason": "aggregation_failed", "error": str(e)}
        
        # Update round counter
        self.current_round += 1
        
        round_time = time.time() - round_start_time
        
        result = {
            "success": True,
            "round": self.current_round - 1,
            "selected_clients": selected_clients,
            "num_updates": len(selected_updates),
            "round_time": round_time,
            "global_params": updated_params,
        }
        
        logger.info(f"Completed round {self.current_round - 1} in {round_time:.2f}s")
        return result
    
    def get_server_stats(self) -> Dict[str, Any]:
        """Get server statistics and performance metrics.
        
        Returns:
            Comprehensive server statistics
        """
        if not self.round_history:
            return {"rounds_completed": 0}
        
        # Calculate aggregate statistics
        recent_rounds = self.round_history[-10:]  # Last 10 rounds
        
        stats = {
            "rounds_completed": self.current_round,
            "total_clients": len(self.client_registry),
            "avg_participation_rate": np.mean(self.performance_metrics["participation_rate"])
            if self.performance_metrics["participation_rate"] else 0.0,
            "avg_round_time": np.mean([
                r["timestamp"] - self.round_history[i-1]["timestamp"]
                for i, r in enumerate(self.round_history[1:], 1)
            ]) if len(self.round_history) > 1 else 0.0,
            "recent_global_accuracy": self.performance_metrics["global_accuracy"][-1]
            if self.performance_metrics["global_accuracy"] else 0.0,
            "recent_global_loss": self.performance_metrics["global_loss"][-1]
            if self.performance_metrics["global_loss"] else float("inf"),
        }
        
        return stats
    
    def save_checkpoint(self, filepath: str) -> None:
        """Save server state checkpoint.
        
        Args:
            filepath: Path to save checkpoint
        """
        checkpoint = {
            "global_model_state": self.global_model.state_dict(),
            "current_round": self.current_round,
            "client_registry": self.client_registry,
            "round_history": self.round_history,
            "performance_metrics": dict(self.performance_metrics),
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"Saved server checkpoint to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> None:
        """Load server state from checkpoint.
        
        Args:
            filepath: Path to checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.global_model.load_state_dict(checkpoint["global_model_state"])
        self.current_round = checkpoint["current_round"]
        self.client_registry = checkpoint["client_registry"]
        self.round_history = checkpoint["round_history"]
        self.performance_metrics = defaultdict(list, checkpoint["performance_metrics"])
        
        logger.info(f"Loaded server checkpoint from {filepath}")


class HierarchicalServer(FederatedServer):
    """Hierarchical federated server for multi-level aggregation.
    
    Supports regional aggregation before global aggregation to reduce
    communication overhead and handle geographic distribution.
    """
    
    def __init__(
        self,
        global_model: nn.Module,
        regions: List[str],
        **kwargs
    ):
        """Initialize hierarchical server.
        
        Args:
            global_model: Global model
            regions: List of region identifiers
            **kwargs: Additional arguments for base server
        """
        super().__init__(global_model, **kwargs)
        
        self.regions = regions
        self.regional_servers = {}
        self.regional_models = {}
        
        # Initialize regional servers
        for region in regions:
            regional_model = type(global_model)().to(self.device)
            regional_model.load_state_dict(global_model.state_dict())
            
            self.regional_servers[region] = FederatedServer(
                regional_model, device=self.device, **kwargs
            )
            self.regional_models[region] = regional_model
        
        logger.info(f"Initialized hierarchical server with {len(regions)} regions")
    
    def assign_client_to_region(self, client_id: str) -> str:
        """Assign client to a region based on location or other criteria.
        
        Args:
            client_id: Client identifier
            
        Returns:
            Assigned region identifier
        """
        # Simple hash-based assignment for now
        # In practice, this would use geographic or network-based assignment
        region_idx = hash(client_id) % len(self.regions)
        return self.regions[region_idx]
    
    def hierarchical_aggregation(
        self,
        client_updates: List[Dict[str, Any]],
    ) -> Dict[str, torch.Tensor]:
        """Perform hierarchical aggregation: regional then global.
        
        Args:
            client_updates: All client updates
            
        Returns:
            Global model parameters after hierarchical aggregation
        """
        # Group updates by region
        regional_updates = defaultdict(list)
        for update in client_updates:
            client_id = update["client_id"]
            region = self.assign_client_to_region(client_id)
            regional_updates[region].append(update)
        
        # First level: Regional aggregation
        regional_models = {}
        for region, updates in regional_updates.items():
            if updates:
                regional_model = self.regional_servers[region].aggregate_updates(updates)
                regional_models[region] = regional_model
        
        # Second level: Global aggregation of regional models
        if regional_models:
            regional_updates_for_global = []
            for region, model_params in regional_models.items():
                # Convert regional model to update format
                regional_update = {
                    "client_id": f"region_{region}",
                    "param_update": model_params,
                    "num_samples": sum(u.get("num_samples", 1) for u in regional_updates[region]),
                    "round": self.current_round,
                }
                regional_updates_for_global.append(regional_update)
            
            # Aggregate regional updates
            global_params = self.aggregate_updates(regional_updates_for_global)
            
            # Update regional models with global model
            for region in self.regions:
                self.regional_models[region].load_state_dict(global_params)
            
            return global_params
        
        return self.get_global_model()