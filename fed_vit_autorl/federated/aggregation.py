"""Federated aggregation algorithms for model updates."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict


class BaseAggregator(ABC):
    """Base class for federated aggregation algorithms."""

    @abstractmethod
    def aggregate(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        client_weights: Optional[List[float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Aggregate client updates into global model update.

        Args:
            client_updates: List of client model parameter updates
            client_weights: Optional weights for each client (e.g., by data size)

        Returns:
            Aggregated global model parameters
        """
        pass


class FedAvgAggregator(BaseAggregator):
    """Federated Averaging (FedAvg) aggregation algorithm.

    The standard FedAvg algorithm that averages client updates weighted
    by the number of local training samples.
    """

    def __init__(self):
        self.round_number = 0

    def aggregate(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        client_weights: Optional[List[float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Aggregate updates using weighted averaging.

        Args:
            client_updates: List of client parameter dictionaries
            client_weights: Weights for each client (defaults to equal weighting)

        Returns:
            Aggregated parameters as state_dict
        """
        if not client_updates:
            raise ValueError("No client updates provided")

        num_clients = len(client_updates)

        # Use equal weights if none provided
        if client_weights is None:
            client_weights = [1.0 / num_clients] * num_clients
        else:
            # Normalize weights
            total_weight = sum(client_weights)
            client_weights = [w / total_weight for w in client_weights]

        # Initialize global update with zeros
        global_update = OrderedDict()
        first_update = client_updates[0]

        for key in first_update.keys():
            global_update[key] = torch.zeros_like(first_update[key])

        # Weighted aggregation
        for client_update, weight in zip(client_updates, client_weights):
            for key in global_update.keys():
                global_update[key] += weight * client_update[key]

        self.round_number += 1
        return global_update


class FedProxAggregator(BaseAggregator):
    """Federated Proximal (FedProx) aggregation algorithm.

    Extends FedAvg with a proximal term to handle heterogeneous data
    and systems heterogeneity.
    """

    def __init__(self, mu: float = 0.01):
        """Initialize FedProx aggregator.

        Args:
            mu: Proximal term coefficient
        """
        self.mu = mu
        self.round_number = 0
        self.global_model_state = None

    def aggregate(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        client_weights: Optional[List[float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Aggregate with proximal regularization.

        Args:
            client_updates: List of client parameter dictionaries
            client_weights: Weights for each client

        Returns:
            Aggregated parameters with proximal regularization
        """
        if not client_updates:
            raise ValueError("No client updates provided")

        # First apply standard FedAvg
        fedavg_aggregator = FedAvgAggregator()
        aggregated_update = fedavg_aggregator.aggregate(client_updates, client_weights)

        # Apply proximal regularization if we have previous global state
        if self.global_model_state is not None:
            for key in aggregated_update.keys():
                # Add proximal term: μ * (w_global - w_local)
                proximal_term = self.mu * (
                    self.global_model_state[key] - aggregated_update[key]
                )
                aggregated_update[key] += proximal_term

        # Update global model state
        self.global_model_state = {
            key: tensor.clone() for key, tensor in aggregated_update.items()
        }

        self.round_number += 1
        return aggregated_update


class AdaptiveAggregator(BaseAggregator):
    """Adaptive aggregation that handles staleness and system heterogeneity."""

    def __init__(self, staleness_factor: float = 0.5, learning_rate: float = 1.0):
        """Initialize adaptive aggregator.

        Args:
            staleness_factor: Factor to discount stale updates
            learning_rate: Global learning rate for aggregation
        """
        self.staleness_factor = staleness_factor
        self.learning_rate = learning_rate
        self.round_number = 0

    def aggregate(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        client_weights: Optional[List[float]] = None,
        staleness_info: Optional[List[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Aggregate with staleness-aware weighting.

        Args:
            client_updates: List of client parameter dictionaries
            client_weights: Base weights for each client
            staleness_info: List of staleness values (rounds behind)

        Returns:
            Aggregated parameters with staleness adjustments
        """
        if not client_updates:
            raise ValueError("No client updates provided")

        num_clients = len(client_updates)

        # Calculate adaptive weights
        if staleness_info is None:
            staleness_info = [0] * num_clients

        adaptive_weights = []
        for i, staleness in enumerate(staleness_info):
            # Discount weight based on staleness
            base_weight = client_weights[i] if client_weights else 1.0
            stale_weight = base_weight * (self.staleness_factor ** staleness)
            adaptive_weights.append(stale_weight)

        # Normalize weights
        total_weight = sum(adaptive_weights)
        if total_weight > 0:
            adaptive_weights = [w / total_weight for w in adaptive_weights]
        else:
            adaptive_weights = [1.0 / num_clients] * num_clients

        # Apply FedAvg with adaptive weights
        fedavg_aggregator = FedAvgAggregator()
        global_update = fedavg_aggregator.aggregate(client_updates, adaptive_weights)

        # Apply global learning rate
        for key in global_update.keys():
            global_update[key] *= self.learning_rate

        self.round_number += 1
        return global_update


class SecureAggregator(BaseAggregator):
    """Secure aggregation using cryptographic protocols."""

    def __init__(self, threshold: int, total_clients: int):
        """Initialize secure aggregator.

        Args:
            threshold: Minimum clients needed for decryption
            total_clients: Total number of participating clients
        """
        self.threshold = threshold
        self.total_clients = total_clients
        self.round_number = 0

    def aggregate(
        self,
        encrypted_updates: List[Dict[str, torch.Tensor]],
        client_weights: Optional[List[float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Aggregate encrypted client updates.

        This is a simplified implementation. In practice, this would use
        actual cryptographic protocols like Shamir's secret sharing.

        Args:
            encrypted_updates: List of encrypted client updates
            client_weights: Weights for each client

        Returns:
            Aggregated (decrypted) parameters if threshold is met
        """
        if len(encrypted_updates) < self.threshold:
            raise ValueError(
                f"Insufficient clients: {len(encrypted_updates)} < {self.threshold}"
            )

        # In practice, this would involve cryptographic operations
        # For now, we'll simulate by using regular FedAvg
        fedavg_aggregator = FedAvgAggregator()
        return fedavg_aggregator.aggregate(encrypted_updates, client_weights)


def compute_model_difference(
    old_params: Dict[str, torch.Tensor],
    new_params: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Compute difference between two model parameter sets.

    Args:
        old_params: Previous model parameters
        new_params: Updated model parameters

    Returns:
        Parameter differences (new - old)
    """
    diff = OrderedDict()
    for key in old_params.keys():
        if key in new_params:
            diff[key] = new_params[key] - old_params[key]
        else:
            diff[key] = torch.zeros_like(old_params[key])
    return diff


def apply_model_update(
    model_params: Dict[str, torch.Tensor],
    update: Dict[str, torch.Tensor],
    learning_rate: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Apply parameter update to model.

    Args:
        model_params: Current model parameters
        update: Parameter update to apply
        learning_rate: Learning rate for update

    Returns:
        Updated model parameters
    """
    updated_params = OrderedDict()
    for key in model_params.keys():
        if key in update:
            updated_params[key] = model_params[key] + learning_rate * update[key]
        else:
            updated_params[key] = model_params[key].clone()
    return updated_params
