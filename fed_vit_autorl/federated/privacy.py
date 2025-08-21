"""Privacy-preserving mechanisms for federated learning."""

import math
import logging
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import hashlib
import secrets


logger = logging.getLogger(__name__)


class DifferentialPrivacy:
    """Differential privacy implementation for federated learning.

    Provides local differential privacy by adding calibrated noise
    to model parameters or gradients before sharing with server.
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        sensitivity: float = 1.0,
        mechanism: str = "gaussian",
        clip_norm: float = 1.0,
    ):
        """Initialize differential privacy mechanism.

        Args:
            epsilon: Privacy budget parameter (lower = more private)
            delta: Probability of privacy violation
            sensitivity: L2 sensitivity of the function
            mechanism: Noise mechanism ("gaussian" or "laplace")
            clip_norm: Gradient clipping norm
        """
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.mechanism = mechanism
        self.clip_norm = clip_norm

        # Calculate noise parameters
        if mechanism == "gaussian":
            self.sigma = self._calculate_gaussian_sigma()
        elif mechanism == "laplace":
            self.scale = sensitivity / epsilon
        else:
            raise ValueError(f"Unknown mechanism: {mechanism}")

        logger.info(
            f"Initialized DP with ε={epsilon}, δ={delta}, mechanism={mechanism}"
        )

    def _calculate_gaussian_sigma(self) -> float:
        """Calculate Gaussian noise sigma for (ε,δ)-differential privacy.

        Returns:
            Noise standard deviation
        """
        if self.delta == 0:
            raise ValueError("Delta must be > 0 for Gaussian mechanism")

        # Using the analytical Gaussian mechanism
        sigma = self.sensitivity * math.sqrt(2 * math.log(1.25 / self.delta)) / self.epsilon
        return sigma

    def clip_gradients(
        self,
        gradients: Dict[str, torch.Tensor],
        max_norm: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """Clip gradients to bound sensitivity.

        Args:
            gradients: Dictionary of gradient tensors
            max_norm: Maximum norm for clipping (defaults to self.clip_norm)

        Returns:
            Clipped gradients
        """
        if max_norm is None:
            max_norm = self.clip_norm

        # Calculate total gradient norm
        total_norm = 0.0
        for grad in gradients.values():
            if grad is not None:
                total_norm += grad.norm().item() ** 2
        total_norm = math.sqrt(total_norm)

        # Clip if necessary
        if total_norm > max_norm:
            clip_coef = max_norm / total_norm
            clipped_gradients = OrderedDict()
            for name, grad in gradients.items():
                if grad is not None:
                    clipped_gradients[name] = grad * clip_coef
                else:
                    clipped_gradients[name] = grad

            logger.debug(f"Clipped gradients: {total_norm:.6f} -> {max_norm}")
            return clipped_gradients

        return gradients

    def add_noise(
        self,
        tensors: Dict[str, torch.Tensor],
        mechanism: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """Add differential privacy noise to tensors.

        Args:
            tensors: Dictionary of tensors to add noise to
            mechanism: Noise mechanism (defaults to self.mechanism)

        Returns:
            Noisy tensors
        """
        if mechanism is None:
            mechanism = self.mechanism

        noisy_tensors = OrderedDict()

        for name, tensor in tensors.items():
            if tensor is None:
                noisy_tensors[name] = tensor
                continue

            if mechanism == "gaussian":
                noise = torch.normal(
                    mean=0.0,
                    std=self.sigma,
                    size=tensor.shape,
                    device=tensor.device,
                    dtype=tensor.dtype,
                )
            elif mechanism == "laplace":
                # Laplace noise using exponential distribution
                uniform = torch.rand(tensor.shape, device=tensor.device, dtype=tensor.dtype)
                # Convert uniform to Laplace
                noise = -self.scale * torch.sign(uniform - 0.5) * torch.log(
                    1 - 2 * torch.abs(uniform - 0.5)
                )
            else:
                raise ValueError(f"Unknown mechanism: {mechanism}")

            noisy_tensors[name] = tensor + noise

        return noisy_tensors

    def privatize_gradients(
        self,
        gradients: Dict[str, torch.Tensor],
        clip_first: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Apply full DP pipeline: clipping + noise addition.

        Args:
            gradients: Raw gradients
            clip_first: Whether to clip gradients first

        Returns:
            Private gradients
        """
        # Step 1: Clip gradients to bound sensitivity
        if clip_first:
            clipped_gradients = self.clip_gradients(gradients)
        else:
            clipped_gradients = gradients

        # Step 2: Add calibrated noise
        private_gradients = self.add_noise(clipped_gradients)

        return private_gradients

    def get_privacy_cost(self) -> Tuple[float, float]:
        """Get current privacy cost.

        Returns:
            Tuple of (epsilon, delta) privacy cost
        """
        return (self.epsilon, self.delta)

    def compose_privacy(self, other_epsilon: float, other_delta: float) -> Tuple[float, float]:
        """Compose privacy costs using basic composition theorem.

        Args:
            other_epsilon: Epsilon from another mechanism
            other_delta: Delta from another mechanism

        Returns:
            Composed (epsilon, delta)
        """
        # Basic composition theorem (can be improved with advanced composition)
        composed_epsilon = self.epsilon + other_epsilon
        composed_delta = self.delta + other_delta

        return (composed_epsilon, composed_delta)


class SecureAggregator:
    """Secure aggregation using cryptographic protocols.

    Implements a simplified version of secure aggregation where
    individual client updates cannot be revealed to the server.
    """

    def __init__(
        self,
        num_clients: int,
        threshold: int,
        key_length: int = 256,
    ):
        """Initialize secure aggregator.

        Args:
            num_clients: Total number of clients
            threshold: Minimum clients needed for decryption
            key_length: Length of cryptographic keys in bits
        """
        self.num_clients = num_clients
        self.threshold = threshold
        self.key_length = key_length

        if threshold > num_clients:
            raise ValueError("Threshold cannot exceed number of clients")

        # Generate server keys (simplified)
        self.server_key = secrets.randbits(key_length)
        self.client_keys = {}

        logger.info(
            f"Initialized secure aggregator: {num_clients} clients, "
            f"threshold={threshold}"
        )

    def generate_client_keys(self) -> Dict[str, int]:
        """Generate cryptographic keys for all clients.

        Returns:
            Dictionary mapping client_id to their secret key
        """
        client_keys = {}
        for i in range(self.num_clients):
            client_id = f"client_{i}"
            client_keys[client_id] = secrets.randbits(self.key_length)

        self.client_keys = client_keys
        return client_keys

    def encrypt_update(
        self,
        client_id: str,
        update: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Encrypt client update (simplified implementation).

        Args:
            client_id: Client identifier
            update: Client model update

        Returns:
            Encrypted update
        """
        if client_id not in self.client_keys:
            raise ValueError(f"Unknown client: {client_id}")

        client_key = self.client_keys[client_id]
        encrypted_update = OrderedDict()

        # Simplified encryption: XOR with key-derived mask
        for name, tensor in update.items():
            # Create deterministic mask from key and tensor name
            mask_seed = hash(f"{client_key}_{name}") % (2**32)
            torch.manual_seed(mask_seed)

            mask = torch.randn_like(tensor) * 0.01  # Small noise for encryption
            encrypted_tensor = tensor + mask
            encrypted_update[name] = encrypted_tensor

        return encrypted_update

    def aggregate_encrypted(
        self,
        encrypted_updates: List[Dict[str, torch.Tensor]],
        client_ids: List[str],
    ) -> Dict[str, torch.Tensor]:
        """Aggregate encrypted updates.

        Args:
            encrypted_updates: List of encrypted client updates
            client_ids: Corresponding client IDs

        Returns:
            Encrypted aggregated result
        """
        if len(encrypted_updates) < self.threshold:
            raise ValueError(
                f"Insufficient clients: {len(encrypted_updates)} < {self.threshold}"
            )

        # Simple aggregation of encrypted values
        if not encrypted_updates:
            return OrderedDict()

        aggregated = OrderedDict()
        first_update = encrypted_updates[0]

        for name in first_update.keys():
            aggregated[name] = torch.zeros_like(first_update[name])

        # Sum encrypted values
        for encrypted_update in encrypted_updates:
            for name in aggregated.keys():
                aggregated[name] += encrypted_update[name]

        # Average
        num_updates = len(encrypted_updates)
        for name in aggregated.keys():
            aggregated[name] /= num_updates

        return aggregated

    def decrypt_aggregate(
        self,
        encrypted_aggregate: Dict[str, torch.Tensor],
        participating_clients: List[str],
    ) -> Dict[str, torch.Tensor]:
        """Decrypt aggregated result (simplified implementation).

        Args:
            encrypted_aggregate: Encrypted aggregated update
            participating_clients: List of clients who participated

        Returns:
            Decrypted aggregated update
        """
        if len(participating_clients) < self.threshold:
            raise ValueError("Insufficient clients for decryption")

        decrypted = OrderedDict()

        # Remove encryption masks
        for name, encrypted_tensor in encrypted_aggregate.items():
            # Calculate combined mask from all participating clients
            combined_mask = torch.zeros_like(encrypted_tensor)

            for client_id in participating_clients:
                if client_id in self.client_keys:
                    client_key = self.client_keys[client_id]
                    mask_seed = hash(f"{client_key}_{name}") % (2**32)
                    torch.manual_seed(mask_seed)

                    mask = torch.randn_like(encrypted_tensor) * 0.01
                    combined_mask += mask

            # Average the mask (since we averaged the encrypted values)
            combined_mask /= len(participating_clients)

            # Decrypt by removing mask
            decrypted_tensor = encrypted_tensor - combined_mask
            decrypted[name] = decrypted_tensor

        return decrypted


class LocalDifferentialPrivacy:
    """Local differential privacy for client-side data protection."""

    def __init__(
        self,
        epsilon: float = 1.0,
        mechanism: str = "randomized_response",
    ):
        """Initialize local DP mechanism.

        Args:
            epsilon: Privacy parameter
            mechanism: Local DP mechanism
        """
        self.epsilon = epsilon
        self.mechanism = mechanism

    def privatize_categorical(self, value: int, num_categories: int) -> int:
        """Apply local DP to categorical data using randomized response.

        Args:
            value: True categorical value
            num_categories: Total number of categories

        Returns:
            Privatized categorical value
        """
        # Randomized response mechanism
        p = math.exp(self.epsilon) / (math.exp(self.epsilon) + num_categories - 1)

        if np.random.random() < p:
            # Report true value
            return value
        else:
            # Report random value from other categories
            other_categories = list(range(num_categories))
            other_categories.remove(value)
            return np.random.choice(other_categories)

    def privatize_numerical(
        self,
        value: float,
        bounds: Tuple[float, float],
        sensitivity: float = 1.0,
    ) -> float:
        """Apply local DP to numerical data using Laplace mechanism.

        Args:
            value: True numerical value
            bounds: (min, max) bounds for the value
            sensitivity: Sensitivity of the query

        Returns:
            Privatized numerical value
        """
        # Clamp value to bounds
        clamped_value = max(bounds[0], min(bounds[1], value))

        # Add Laplace noise
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale)
        private_value = clamped_value + noise

        # Clamp to bounds again
        return max(bounds[0], min(bounds[1], private_value))

    def privatize_location(
        self,
        latitude: float,
        longitude: float,
        radius_km: float = 1.0,
    ) -> Tuple[float, float]:
        """Apply geo-indistinguishability to location data.

        Args:
            latitude: True latitude
            longitude: True longitude
            radius_km: Privacy radius in kilometers

        Returns:
            Privatized (latitude, longitude)
        """
        # Convert radius to degrees (approximate)
        radius_deg = radius_km / 111.0  # 1 degree ≈ 111 km

        # Sample from planar Laplace distribution
        scale = radius_deg / self.epsilon

        # Sample radius and angle
        r = np.random.exponential(scale)
        theta = np.random.uniform(0, 2 * math.pi)

        # Add noise
        noise_lat = r * math.cos(theta)
        noise_lon = r * math.sin(theta)

        private_lat = latitude + noise_lat
        private_lon = longitude + noise_lon

        # Clamp to valid coordinates
        private_lat = max(-90, min(90, private_lat))
        private_lon = max(-180, min(180, private_lon))

        return (private_lat, private_lon)


class PrivacyAccountant:
    """Track and manage privacy budget across federated rounds."""

    def __init__(self, total_epsilon: float = 10.0, total_delta: float = 1e-4):
        """Initialize privacy accountant.

        Args:
            total_epsilon: Total privacy budget
            total_delta: Total delta budget
        """
        self.total_epsilon = total_epsilon
        self.total_delta = total_delta
        self.used_epsilon = 0.0
        self.used_delta = 0.0
        self.privacy_history = []

    def spend_privacy(self, epsilon: float, delta: float, description: str = "") -> bool:
        """Spend privacy budget for an operation.

        Args:
            epsilon: Epsilon cost of operation
            delta: Delta cost of operation
            description: Description of the operation

        Returns:
            True if budget allows, False otherwise
        """
        if self.used_epsilon + epsilon > self.total_epsilon:
            logger.warning(f"Epsilon budget exceeded: {self.used_epsilon + epsilon} > {self.total_epsilon}")
            return False

        if self.used_delta + delta > self.total_delta:
            logger.warning(f"Delta budget exceeded: {self.used_delta + delta} > {self.total_delta}")
            return False

        # Spend budget
        self.used_epsilon += epsilon
        self.used_delta += delta

        # Record transaction
        self.privacy_history.append({
            "epsilon": epsilon,
            "delta": delta,
            "description": description,
            "timestamp": time.time(),
            "remaining_epsilon": self.total_epsilon - self.used_epsilon,
            "remaining_delta": self.total_delta - self.used_delta,
        })

        logger.info(
            f"Privacy spent: ε={epsilon:.6f}, δ={delta:.6f} ({description}). "
            f"Remaining: ε={self.total_epsilon - self.used_epsilon:.6f}, "
            f"δ={self.total_delta - self.used_delta:.6f}"
        )

        return True

    def get_remaining_budget(self) -> Tuple[float, float]:
        """Get remaining privacy budget.

        Returns:
            Tuple of (remaining_epsilon, remaining_delta)
        """
        return (
            self.total_epsilon - self.used_epsilon,
            self.total_delta - self.used_delta,
        )

    def reset_budget(self, new_epsilon: float, new_delta: float) -> None:
        """Reset privacy budget (e.g., for new time period).

        Args:
            new_epsilon: New total epsilon budget
            new_delta: New total delta budget
        """
        self.total_epsilon = new_epsilon
        self.total_delta = new_delta
        self.used_epsilon = 0.0
        self.used_delta = 0.0
        self.privacy_history.clear()

        logger.info(f"Reset privacy budget: ε={new_epsilon}, δ={new_delta}")

    def get_privacy_report(self) -> Dict[str, Any]:
        """Generate privacy usage report.

        Returns:
            Privacy usage statistics
        """
        return {
            "total_epsilon": self.total_epsilon,
            "total_delta": self.total_delta,
            "used_epsilon": self.used_epsilon,
            "used_delta": self.used_delta,
            "remaining_epsilon": self.total_epsilon - self.used_epsilon,
            "remaining_delta": self.total_delta - self.used_delta,
            "privacy_exhausted": (
                self.used_epsilon >= self.total_epsilon or
                self.used_delta >= self.total_delta
            ),
            "num_operations": len(self.privacy_history),
            "history": self.privacy_history,
        }
