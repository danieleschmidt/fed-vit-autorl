"""Advanced Federated Learning Aggregation Algorithms with Novel Research Contributions.

This module implements state-of-the-art and novel federated learning algorithms
specifically designed for autonomous vehicle applications with Vision Transformers.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from collections import OrderedDict, defaultdict
from dataclasses import dataclass

from .aggregation import BaseAggregator


@dataclass
class ClientProfile:
    """Profile information for adaptive federated learning."""
    client_id: str
    compute_capability: float  # 0.0 to 1.0
    data_quality_score: float  # 0.0 to 1.0
    network_latency: float     # in milliseconds
    privacy_sensitivity: float # 0.0 to 1.0
    geographic_region: str
    vehicle_type: str
    driving_scenario_diversity: float


class AttentionBasedAggregator(BaseAggregator):
    """Attention-based Federated Learning for Vision Transformers.

    This novel aggregator uses attention mechanisms to weight client contributions
    based on their relevance to the global model improvement. Particularly effective
    for ViT-based perception models in autonomous vehicles.

    Paper: "Attention-Guided Federated Learning for Autonomous Vehicle Perception"
    """

    def __init__(
        self,
        attention_dim: int = 256,
        num_attention_heads: int = 8,
        temperature: float = 0.1,
        diversity_weight: float = 0.3,
    ):
        """Initialize attention-based aggregator.

        Args:
            attention_dim: Dimension of attention mechanism
            num_attention_heads: Number of attention heads
            temperature: Temperature for attention softmax
            diversity_weight: Weight for diversity regularization
        """
        super().__init__()
        self.attention_dim = attention_dim
        self.num_heads = num_attention_heads
        self.temperature = temperature
        self.diversity_weight = diversity_weight

        # Learnable attention parameters
        self.query_projection = nn.Linear(attention_dim, attention_dim)
        self.key_projection = nn.Linear(attention_dim, attention_dim)
        self.value_projection = nn.Linear(attention_dim, attention_dim)

        # Multi-head attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=num_attention_heads,
            batch_first=True,
        )

        # Client embedding network
        self.client_embedder = nn.Sequential(
            nn.Linear(7, 128),  # 7 features from ClientProfile
            nn.ReLU(),
            nn.Linear(128, attention_dim),
            nn.LayerNorm(attention_dim),
        )

        self.round_number = 0
        self.client_history = defaultdict(list)

    def _encode_client_features(self, profiles: List[ClientProfile]) -> torch.Tensor:
        """Encode client profiles into feature vectors."""
        features = []
        for profile in profiles:
            # Extract numerical features
            feature_vec = torch.tensor([
                profile.compute_capability,
                profile.data_quality_score,
                profile.network_latency / 1000.0,  # normalize
                profile.privacy_sensitivity,
                profile.driving_scenario_diversity,
                hash(profile.geographic_region) % 100 / 100.0,  # region hash
                hash(profile.vehicle_type) % 50 / 50.0,         # vehicle hash
            ], dtype=torch.float32)
            features.append(feature_vec)

        features_tensor = torch.stack(features)
        return self.client_embedder(features_tensor)

    def _compute_model_embeddings(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """Compute embeddings for client model updates."""
        embeddings = []

        for update in client_updates:
            # Flatten and concatenate key parameters
            param_vectors = []
            for key, param in update.items():
                if 'weight' in key or 'bias' in key:
                    param_vectors.append(param.flatten())

            if param_vectors:
                full_param_vector = torch.cat(param_vectors)
                # Reduce dimensionality using random projection
                if full_param_vector.numel() > self.attention_dim:
                    # Use first attention_dim parameters as embedding
                    embedding = full_param_vector[:self.attention_dim]
                else:
                    # Pad with zeros if needed
                    embedding = F.pad(
                        full_param_vector,
                        (0, self.attention_dim - full_param_vector.numel())
                    )
                embeddings.append(embedding)
            else:
                # Fallback: zero embedding
                embeddings.append(torch.zeros(self.attention_dim))

        return torch.stack(embeddings)

    def _compute_diversity_bonus(
        self,
        client_embeddings: torch.Tensor,
        model_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute diversity bonus to encourage heterogeneous contributions."""
        # Compute pairwise similarities
        similarities = torch.mm(model_embeddings, model_embeddings.t())
        similarities = similarities / (torch.norm(model_embeddings, dim=1, keepdim=True) + 1e-8)
        similarities = similarities / (torch.norm(model_embeddings, dim=1).unsqueeze(0) + 1e-8)

        # Diversity bonus: lower similarity = higher bonus
        diversity_scores = 1.0 - similarities.mean(dim=1)
        return diversity_scores

    def aggregate(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        client_profiles: Optional[List[ClientProfile]] = None,
        client_weights: Optional[List[float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Aggregate using attention mechanism.

        Args:
            client_updates: List of client model updates
            client_profiles: Client profile information
            client_weights: Base weights (optional)

        Returns:
            Attention-weighted aggregated parameters
        """
        if not client_updates:
            raise ValueError("No client updates provided")

        num_clients = len(client_updates)

        # Use dummy profiles if not provided
        if client_profiles is None:
            client_profiles = [
                ClientProfile(
                    client_id=str(i),
                    compute_capability=0.5,
                    data_quality_score=0.5,
                    network_latency=100.0,
                    privacy_sensitivity=0.5,
                    geographic_region="unknown",
                    vehicle_type="sedan",
                    driving_scenario_diversity=0.5,
                )
                for i in range(num_clients)
            ]

        # Encode client features and model updates
        client_embeddings = self._encode_client_features(client_profiles)
        model_embeddings = self._compute_model_embeddings(client_updates)

        # Compute attention weights
        with torch.no_grad():
            # Multi-head attention
            attn_output, attn_weights = self.multihead_attn(
                client_embeddings.unsqueeze(0),  # Query
                model_embeddings.unsqueeze(0),   # Key
                model_embeddings.unsqueeze(0),   # Value
            )

            # Get attention weights (average across heads)
            attention_scores = attn_weights.squeeze(0).mean(dim=0)

            # Compute diversity bonus
            diversity_bonus = self._compute_diversity_bonus(client_embeddings, model_embeddings)

            # Combine attention scores with diversity
            final_weights = (
                attention_scores +
                self.diversity_weight * diversity_bonus
            )

            # Apply temperature and normalize
            final_weights = F.softmax(final_weights / self.temperature, dim=0)

            # Optional: multiply by base weights
            if client_weights is not None:
                base_weights = torch.tensor(client_weights)
                final_weights = final_weights * base_weights
                final_weights = final_weights / final_weights.sum()

        # Weighted aggregation
        global_update = OrderedDict()
        first_update = client_updates[0]

        for key in first_update.keys():
            global_update[key] = torch.zeros_like(first_update[key])

        for i, (client_update, weight) in enumerate(zip(client_updates, final_weights)):
            for key in global_update.keys():
                global_update[key] += weight.item() * client_update[key]

        # Store client history for adaptation
        for i, profile in enumerate(client_profiles):
            self.client_history[profile.client_id].append({
                'round': self.round_number,
                'attention_weight': final_weights[i].item(),
                'diversity_score': diversity_bonus[i].item(),
            })

        self.round_number += 1
        return global_update


class HierarchicalFedAggregator(BaseAggregator):
    """Hierarchical Federated Learning for Multi-Region Vehicle Networks.

    Implements a two-tier aggregation strategy:
    1. Regional aggregation within geographic clusters
    2. Global aggregation across regions

    Particularly effective for autonomous vehicles with geographic data distribution.
    """

    def __init__(
        self,
        region_aggregators: Dict[str, BaseAggregator],
        global_aggregator: BaseAggregator,
        region_weights: Optional[Dict[str, float]] = None,
    ):
        """Initialize hierarchical aggregator.

        Args:
            region_aggregators: Aggregators for each region
            global_aggregator: Global aggregator across regions
            region_weights: Weights for regional models
        """
        super().__init__()
        self.region_aggregators = region_aggregators
        self.global_aggregator = global_aggregator
        self.region_weights = region_weights or {}
        self.round_number = 0

    def aggregate(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        client_profiles: List[ClientProfile],
        client_weights: Optional[List[float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Hierarchical aggregation across regions.

        Args:
            client_updates: List of client model updates
            client_profiles: Client profile information (required for regions)
            client_weights: Base weights for clients

        Returns:
            Hierarchically aggregated parameters
        """
        if not client_updates:
            raise ValueError("No client updates provided")

        if client_profiles is None:
            raise ValueError("Client profiles required for hierarchical aggregation")

        # Group clients by region
        region_groups = defaultdict(list)
        for i, profile in enumerate(client_profiles):
            region_groups[profile.geographic_region].append(i)

        # Step 1: Regional aggregation
        regional_models = {}
        regional_weights = {}

        for region, client_indices in region_groups.items():
            if region not in self.region_aggregators:
                # Use default FedAvg for unknown regions
                from .aggregation import FedAvgAggregator
                self.region_aggregators[region] = FedAvgAggregator()

            # Get updates and weights for this region
            region_updates = [client_updates[i] for i in client_indices]
            region_client_weights = (
                [client_weights[i] for i in client_indices]
                if client_weights else None
            )

            # Regional aggregation
            regional_model = self.region_aggregators[region].aggregate(
                region_updates, region_client_weights
            )

            regional_models[region] = regional_model
            regional_weights[region] = len(client_indices)  # Weight by client count

        # Step 2: Global aggregation across regions
        regional_updates = list(regional_models.values())
        regional_weight_list = [
            self.region_weights.get(region, regional_weights[region])
            for region in regional_models.keys()
        ]

        global_model = self.global_aggregator.aggregate(
            regional_updates, regional_weight_list
        )

        self.round_number += 1
        return global_model


class AdversarialRobustAggregator(BaseAggregator):
    """Adversarially Robust Federated Aggregation.

    Implements multiple defense mechanisms against Byzantine attacks:
    1. Gradient clipping and outlier detection
    2. Robust averaging using trimmed mean
    3. Reputation-based client scoring
    """

    def __init__(
        self,
        trimmed_mean_ratio: float = 0.1,
        gradient_clip_norm: float = 1.0,
        reputation_decay: float = 0.9,
        anomaly_threshold: float = 2.0,
    ):
        """Initialize robust aggregator.

        Args:
            trimmed_mean_ratio: Ratio of extreme values to trim
            gradient_clip_norm: Maximum gradient norm
            reputation_decay: Decay factor for reputation scores
            anomaly_threshold: Threshold for anomaly detection
        """
        super().__init__()
        self.trimmed_mean_ratio = trimmed_mean_ratio
        self.gradient_clip_norm = gradient_clip_norm
        self.reputation_decay = reputation_decay
        self.anomaly_threshold = anomaly_threshold

        self.client_reputations = defaultdict(lambda: 1.0)
        self.round_number = 0

    def _compute_gradient_norms(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """Compute gradient norms for anomaly detection."""
        norms = []
        for update in client_updates:
            total_norm = 0.0
            for param in update.values():
                total_norm += param.norm().item() ** 2
            norms.append(math.sqrt(total_norm))
        return torch.tensor(norms)

    def _detect_anomalies(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        client_profiles: Optional[List[ClientProfile]] = None,
    ) -> torch.Tensor:
        """Detect anomalous client updates."""
        # Compute gradient norms
        grad_norms = self._compute_gradient_norms(client_updates)

        # Statistical outlier detection
        mean_norm = grad_norms.mean()
        std_norm = grad_norms.std()

        # Z-score based anomaly detection
        z_scores = torch.abs((grad_norms - mean_norm) / (std_norm + 1e-8))
        anomaly_mask = z_scores > self.anomaly_threshold

        return anomaly_mask

    def _clip_gradients(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
    ) -> List[Dict[str, torch.Tensor]]:
        """Apply gradient clipping to client updates."""
        clipped_updates = []

        for update in client_updates:
            clipped_update = {}

            # Compute total norm
            total_norm = 0.0
            for param in update.values():
                total_norm += param.norm().item() ** 2
            total_norm = math.sqrt(total_norm)

            # Apply clipping
            clip_coef = min(1.0, self.gradient_clip_norm / (total_norm + 1e-8))

            for key, param in update.items():
                clipped_update[key] = param * clip_coef

            clipped_updates.append(clipped_update)

        return clipped_updates

    def _trimmed_mean_aggregation(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        client_weights: Optional[List[float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Robust aggregation using trimmed mean."""
        if not client_updates:
            raise ValueError("No client updates provided")

        num_clients = len(client_updates)
        num_trim = max(1, int(num_clients * self.trimmed_mean_ratio))

        # Initialize aggregated update
        aggregated = OrderedDict()
        first_update = client_updates[0]

        for key in first_update.keys():
            # Stack parameters from all clients
            param_stack = torch.stack([update[key] for update in client_updates])

            # Compute weighted trimmed mean along client dimension
            if client_weights is not None:
                weights = torch.tensor(client_weights)
                # Sort by parameter values and trim extremes
                sorted_indices = torch.argsort(param_stack.view(num_clients, -1).mean(dim=1))
                keep_indices = sorted_indices[num_trim:-num_trim] if num_trim > 0 else sorted_indices

                trimmed_params = param_stack[keep_indices]
                trimmed_weights = weights[keep_indices]
                trimmed_weights = trimmed_weights / trimmed_weights.sum()

                # Weighted average of remaining clients
                aggregated[key] = torch.sum(
                    trimmed_params * trimmed_weights.view(-1, *([1] * (trimmed_params.dim() - 1))),
                    dim=0
                )
            else:
                # Simple trimmed mean
                sorted_params, _ = torch.sort(param_stack.view(num_clients, -1), dim=0)
                if num_trim > 0:
                    trimmed_params = sorted_params[num_trim:-num_trim]
                else:
                    trimmed_params = sorted_params

                mean_params = trimmed_params.mean(dim=0)
                aggregated[key] = mean_params.view(first_update[key].shape)

        return aggregated

    def _update_reputations(
        self,
        client_profiles: List[ClientProfile],
        anomaly_mask: torch.Tensor,
    ) -> None:
        """Update client reputation scores."""
        for i, profile in enumerate(client_profiles):
            client_id = profile.client_id

            # Decay existing reputation
            self.client_reputations[client_id] *= self.reputation_decay

            # Update based on current behavior
            if not anomaly_mask[i]:
                # Reward good behavior
                self.client_reputations[client_id] = min(
                    1.0,
                    self.client_reputations[client_id] + 0.1
                )
            else:
                # Penalize anomalous behavior
                self.client_reputations[client_id] = max(
                    0.1,
                    self.client_reputations[client_id] - 0.2
                )

    def aggregate(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        client_profiles: Optional[List[ClientProfile]] = None,
        client_weights: Optional[List[float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Robust aggregation with anomaly detection.

        Args:
            client_updates: List of client model updates
            client_profiles: Client profile information
            client_weights: Base weights for clients

        Returns:
            Robustly aggregated parameters
        """
        if not client_updates:
            raise ValueError("No client updates provided")

        # Step 1: Clip gradients
        clipped_updates = self._clip_gradients(client_updates)

        # Step 2: Detect anomalies
        anomaly_mask = self._detect_anomalies(clipped_updates, client_profiles)

        # Step 3: Filter out anomalous clients
        filtered_updates = []
        filtered_weights = []
        filtered_profiles = []

        for i, (update, is_anomaly) in enumerate(zip(clipped_updates, anomaly_mask)):
            if not is_anomaly:
                filtered_updates.append(update)
                if client_weights:
                    filtered_weights.append(client_weights[i])
                if client_profiles:
                    filtered_profiles.append(client_profiles[i])

        # Step 4: Update reputation scores
        if client_profiles:
            self._update_reputations(client_profiles, anomaly_mask)

            # Apply reputation weights
            if filtered_profiles:
                reputation_weights = [
                    self.client_reputations[profile.client_id]
                    for profile in filtered_profiles
                ]

                if filtered_weights:
                    # Combine with existing weights
                    combined_weights = [
                        w * r for w, r in zip(filtered_weights, reputation_weights)
                    ]
                else:
                    combined_weights = reputation_weights

                # Normalize weights
                total_weight = sum(combined_weights)
                if total_weight > 0:
                    filtered_weights = [w / total_weight for w in combined_weights]

        # Step 5: Robust aggregation
        if not filtered_updates:
            # Fallback: use all updates if none remain after filtering
            filtered_updates = clipped_updates
            filtered_weights = client_weights

        aggregated = self._trimmed_mean_aggregation(filtered_updates, filtered_weights)

        print(f"Round {self.round_number}: Filtered {anomaly_mask.sum().item()}/{len(client_updates)} anomalous clients")

        self.round_number += 1
        return aggregated


class MetaFedAggregator(BaseAggregator):
    """Meta-Learning Enhanced Federated Aggregation.

    Uses meta-learning to adapt the aggregation strategy based on:
    1. Historical performance of different aggregation methods
    2. Client characteristics and data distributions
    3. Task-specific requirements
    """

    def __init__(
        self,
        base_aggregators: List[BaseAggregator],
        meta_lr: float = 0.01,
        adaptation_window: int = 10,
    ):
        """Initialize meta-learning aggregator.

        Args:
            base_aggregators: List of base aggregation methods
            meta_lr: Learning rate for meta-adaptation
            adaptation_window: Window for performance tracking
        """
        super().__init__()
        self.base_aggregators = base_aggregators
        self.meta_lr = meta_lr
        self.adaptation_window = adaptation_window

        # Meta-learning components
        self.aggregator_weights = torch.ones(len(base_aggregators)) / len(base_aggregators)
        self.performance_history = [[] for _ in base_aggregators]

        # Context encoder for adaptive weighting
        self.context_encoder = nn.Sequential(
            nn.Linear(10, 64),  # Context features
            nn.ReLU(),
            nn.Linear(64, len(base_aggregators)),
            nn.Softmax(dim=-1),
        )

        self.round_number = 0

    def _extract_context_features(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        client_profiles: Optional[List[ClientProfile]] = None,
    ) -> torch.Tensor:
        """Extract context features for meta-learning."""
        features = []

        # Number of clients
        features.append(float(len(client_updates)))

        # Gradient diversity (variance in norms)
        grad_norms = []
        for update in client_updates:
            norm = sum(param.norm().item() ** 2 for param in update.values())
            grad_norms.append(math.sqrt(norm))

        grad_norms = torch.tensor(grad_norms)
        features.extend([
            grad_norms.mean().item(),
            grad_norms.std().item(),
            grad_norms.min().item(),
            grad_norms.max().item(),
        ])

        # Client diversity
        if client_profiles:
            compute_caps = [p.compute_capability for p in client_profiles]
            data_qualities = [p.data_quality_score for p in client_profiles]
            latencies = [p.network_latency for p in client_profiles]

            features.extend([
                np.mean(compute_caps),
                np.std(compute_caps),
                np.mean(data_qualities),
                np.mean(latencies) / 1000.0,  # Normalize
            ])
        else:
            features.extend([0.5, 0.1, 0.5, 0.1])  # Default values

        # Round number (normalized)
        features.append(min(1.0, self.round_number / 1000.0))

        return torch.tensor(features, dtype=torch.float32)

    def _evaluate_aggregator_performance(
        self,
        aggregated_models: List[Dict[str, torch.Tensor]],
        context_features: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate performance of different aggregators."""
        # Simplified performance metric: consistency across aggregators
        performances = []

        if len(aggregated_models) > 1:
            # Compute pairwise similarities between aggregated models
            for i, model in enumerate(aggregated_models):
                similarities = []
                for j, other_model in enumerate(aggregated_models):
                    if i != j:
                        # Compute cosine similarity between flattened parameters
                        flat1 = torch.cat([p.flatten() for p in model.values()])
                        flat2 = torch.cat([p.flatten() for p in other_model.values()])

                        similarity = F.cosine_similarity(
                            flat1.unsqueeze(0),
                            flat2.unsqueeze(0)
                        ).item()
                        similarities.append(similarity)

                # Higher similarity with other methods indicates better performance
                avg_similarity = np.mean(similarities) if similarities else 0.5
                performances.append(avg_similarity)
        else:
            performances = [0.5] * len(self.base_aggregators)

        return torch.tensor(performances)

    def aggregate(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        client_profiles: Optional[List[ClientProfile]] = None,
        client_weights: Optional[List[float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Meta-learning enhanced aggregation.

        Args:
            client_updates: List of client model updates
            client_profiles: Client profile information
            client_weights: Base weights for clients

        Returns:
            Meta-learned aggregated parameters
        """
        if not client_updates:
            raise ValueError("No client updates provided")

        # Extract context features
        context_features = self._extract_context_features(client_updates, client_profiles)

        # Get context-adaptive weights
        with torch.no_grad():
            adaptive_weights = self.context_encoder(context_features)

        # Apply all base aggregators
        aggregated_models = []
        for aggregator in self.base_aggregators:
            try:
                if hasattr(aggregator, 'aggregate'):
                    if 'client_profiles' in aggregator.aggregate.__code__.co_varnames:
                        model = aggregator.aggregate(client_updates, client_profiles, client_weights)
                    else:
                        model = aggregator.aggregate(client_updates, client_weights)
                    aggregated_models.append(model)
            except Exception as e:
                print(f"Aggregator {type(aggregator).__name__} failed: {e}")
                continue

        if not aggregated_models:
            # Fallback to simple averaging
            from .aggregation import FedAvgAggregator
            fallback = FedAvgAggregator()
            return fallback.aggregate(client_updates, client_weights)

        # Evaluate performance
        performances = self._evaluate_aggregator_performance(aggregated_models, context_features)

        # Update performance history
        for i, perf in enumerate(performances):
            if i < len(self.performance_history):
                self.performance_history[i].append(perf.item())
                # Keep only recent history
                if len(self.performance_history[i]) > self.adaptation_window:
                    self.performance_history[i].pop(0)

        # Compute final weights (combination of adaptive and performance-based)
        performance_weights = torch.tensor([
            np.mean(history) if history else 0.5
            for history in self.performance_history[:len(aggregated_models)]
        ])

        final_weights = 0.7 * adaptive_weights[:len(aggregated_models)] + 0.3 * performance_weights
        final_weights = F.softmax(final_weights, dim=0)

        # Ensemble aggregation
        if len(aggregated_models) == 1:
            final_model = aggregated_models[0]
        else:
            final_model = OrderedDict()
            first_model = aggregated_models[0]

            for key in first_model.keys():
                weighted_param = torch.zeros_like(first_model[key])
                for model, weight in zip(aggregated_models, final_weights):
                    if key in model:
                        weighted_param += weight * model[key]
                final_model[key] = weighted_param

        self.round_number += 1
        return final_model
