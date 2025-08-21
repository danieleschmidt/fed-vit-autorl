"""Federated Proximal Policy Optimization (PPO) for autonomous driving."""

import logging
from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
import numpy as np
from collections import namedtuple

from ..federated.privacy import DifferentialPrivacy


logger = logging.getLogger(__name__)

# Experience tuple for PPO
PPOExperience = namedtuple(
    "PPOExperience",
    ["state", "action", "reward", "next_state", "done", "log_prob", "value"]
)


class PPOPolicy(nn.Module):
    """PPO policy network for continuous control tasks."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        activation: str = "relu",
        action_std: float = 0.1,
    ):
        """Initialize PPO policy network.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            activation: Activation function
            action_std: Standard deviation for action noise
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_std = action_std

        # Build network layers
        layers = []
        input_dim = state_dim

        activation_fn = getattr(F, activation)

        for _ in range(num_layers):
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                activation_fn,
                nn.Dropout(0.1),
            ])
            input_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)

        # Policy head (mean of action distribution)
        self.action_mean = nn.Linear(hidden_dim, action_dim)

        # Action log standard deviation (learnable)
        self.action_log_std = nn.Parameter(
            torch.full((action_dim,), np.log(action_std))
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=0.01)
            torch.nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through policy network.

        Args:
            state: State tensor (batch_size, state_dim)

        Returns:
            Tuple of (action_mean, action_std)
        """
        features = self.feature_extractor(state)
        action_mean = self.action_mean(features)
        action_std = torch.exp(self.action_log_std)

        return action_mean, action_std

    def get_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy.

        Args:
            state: State tensor
            deterministic: Whether to use deterministic policy

        Returns:
            Tuple of (action, log_prob)
        """
        action_mean, action_std = self.forward(state)

        if deterministic:
            action = action_mean
            # Log prob of deterministic action (delta function)
            log_prob = torch.zeros(action_mean.shape[0], device=action_mean.device)
        else:
            # Sample from normal distribution
            dist = Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)

        # Apply tanh to bound actions to [-1, 1]
        action = torch.tanh(action)

        # Adjust log probability for tanh transformation
        if not deterministic:
            log_prob -= torch.log(1 - action**2 + 1e-6).sum(dim=-1)

        return action, log_prob

    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log probabilities and entropy of given actions.

        Args:
            states: State tensor
            actions: Action tensor

        Returns:
            Tuple of (log_probs, entropy)
        """
        action_mean, action_std = self.forward(states)

        # Inverse tanh transformation
        actions_pretanh = torch.atanh(torch.clamp(actions, -0.999, 0.999))

        dist = Normal(action_mean, action_std)
        log_probs = dist.log_prob(actions_pretanh).sum(dim=-1)

        # Adjust for tanh transformation
        log_probs -= torch.log(1 - actions**2 + 1e-6).sum(dim=-1)

        # Calculate entropy
        entropy = dist.entropy().sum(dim=-1)

        return log_probs, entropy


class PPOValueFunction(nn.Module):
    """PPO value function network."""

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        activation: str = "relu",
    ):
        """Initialize value function network.

        Args:
            state_dim: Dimension of state space
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            activation: Activation function
        """
        super().__init__()

        # Build network layers
        layers = []
        input_dim = state_dim

        activation_fn = getattr(F, activation)

        for _ in range(num_layers):
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                activation_fn,
                nn.Dropout(0.1),
            ])
            input_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, 1))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=1.0)
            torch.nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through value network.

        Args:
            state: State tensor (batch_size, state_dim)

        Returns:
            Value estimates (batch_size, 1)
        """
        return self.network(state).squeeze(-1)


class FederatedPPO:
    """Federated Proximal Policy Optimization for autonomous driving."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr_policy: float = 3e-4,
        lr_value: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        clip_value: float = 0.5,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 10,
        batch_size: int = 64,
        device: str = "cpu",
        privacy_budget: float = 1.0,
    ):
        """Initialize Federated PPO agent.

        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            lr_policy: Policy learning rate
            lr_value: Value function learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            clip_value: Value function clipping parameter
            entropy_coef: Entropy regularization coefficient
            value_coef: Value function loss coefficient
            max_grad_norm: Maximum gradient norm for clipping
            ppo_epochs: Number of PPO epochs per update
            batch_size: Batch size for training
            device: Device for computation
            privacy_budget: Differential privacy budget
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.clip_value = clip_value
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.device = device

        # Initialize networks
        self.policy = PPOPolicy(state_dim, action_dim).to(device)
        self.value_function = PPOValueFunction(state_dim).to(device)

        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr_policy)
        self.value_optimizer = optim.Adam(self.value_function.parameters(), lr=lr_value)

        # Privacy mechanism
        self.privacy = DifferentialPrivacy(
            epsilon=privacy_budget,
            delta=1e-5,
            sensitivity=1.0,
        )

        # Training statistics
        self.training_stats = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "kl_divergence": [],
            "clip_fraction": [],
        }

        logger.info(f"Initialized Federated PPO with state_dim={state_dim}, action_dim={action_dim}")

    def get_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action from policy.

        Args:
            state: Current state
            deterministic: Whether to use deterministic policy

        Returns:
            Tuple of (action, log_prob, value)
        """
        self.policy.eval()
        self.value_function.eval()

        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)

            action, log_prob = self.policy.get_action(state, deterministic)
            value = self.value_function(state)

        return action.squeeze(0), log_prob.squeeze(0), value.squeeze(0)

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards: Reward tensor
            values: Value estimates
            dones: Done flags
            next_value: Value of next state

        Returns:
            Tuple of (advantages, returns)
        """
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        # Bootstrap from next value
        gae = 0
        next_val = next_value

        # Compute advantages backward
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value_t = next_val
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value_t = values[t + 1]

            delta = rewards[t] + self.gamma * next_value_t * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae

        # Returns = advantages + values
        returns = advantages + values

        return advantages, returns

    def update(
        self,
        experiences: List[PPOExperience],
    ) -> Dict[str, float]:
        """Update policy and value function using PPO.

        Args:
            experiences: List of PPO experiences

        Returns:
            Training statistics
        """
        if not experiences:
            return {}

        # Convert experiences to tensors
        states = torch.stack([exp.state for exp in experiences]).to(self.device)
        actions = torch.stack([exp.action for exp in experiences]).to(self.device)
        rewards = torch.tensor([exp.reward for exp in experiences], dtype=torch.float32).to(self.device)
        dones = torch.tensor([exp.done for exp in experiences], dtype=torch.float32).to(self.device)
        old_log_probs = torch.stack([exp.log_prob for exp in experiences]).to(self.device)
        old_values = torch.stack([exp.value for exp in experiences]).to(self.device)

        # Compute advantages and returns
        with torch.no_grad():
            # Get value of last state for bootstrapping
            if not experiences[-1].done:
                next_value = self.value_function(experiences[-1].next_state.unsqueeze(0).to(self.device))
            else:
                next_value = torch.tensor(0.0).to(self.device)

            advantages, returns = self.compute_gae(rewards, old_values, dones, next_value)

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update loop
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        total_clip_fraction = 0.0

        for epoch in range(self.ppo_epochs):
            # Create random batches
            indices = torch.randperm(len(experiences))

            for start in range(0, len(experiences), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_values = old_values[batch_indices]

                # Evaluate current policy
                log_probs, entropy = self.policy.evaluate_actions(batch_states, batch_actions)
                values = self.value_function(batch_states)

                # Policy loss (PPO clip)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped)
                if self.clip_value > 0:
                    value_pred_clipped = batch_old_values + torch.clamp(
                        values - batch_old_values, -self.clip_value, self.clip_value
                    )
                    value_loss1 = F.mse_loss(values, batch_returns)
                    value_loss2 = F.mse_loss(value_pred_clipped, batch_returns)
                    value_loss = torch.max(value_loss1, value_loss2)
                else:
                    value_loss = F.mse_loss(values, batch_returns)

                # Entropy loss
                entropy_loss = -entropy.mean()

                # Total loss
                total_loss = (
                    policy_loss +
                    self.value_coef * value_loss +
                    self.entropy_coef * entropy_loss
                )

                # Update policy
                self.policy_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy_optimizer.step()

                # Update value function
                self.value_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value_function.parameters(), self.max_grad_norm)
                self.value_optimizer.step()

                # Record statistics
                with torch.no_grad():
                    kl = (batch_old_log_probs - log_probs).mean()
                    clip_fraction = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                total_kl += kl.item()
                total_clip_fraction += clip_fraction.item()

        # Average statistics
        num_updates = self.ppo_epochs * ((len(experiences) + self.batch_size - 1) // self.batch_size)

        stats = {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates,
            "kl_divergence": total_kl / num_updates,
            "clip_fraction": total_clip_fraction / num_updates,
        }

        # Update training statistics
        for key, value in stats.items():
            self.training_stats[key].append(value)

        logger.info(f"PPO update: policy_loss={stats['policy_loss']:.6f}, value_loss={stats['value_loss']:.6f}")

        return stats

    def get_model_parameters(self) -> Dict[str, torch.Tensor]:
        """Get model parameters for federated learning.

        Returns:
            Dictionary of model parameters
        """
        params = {}

        # Policy parameters
        for name, param in self.policy.named_parameters():
            params[f"policy.{name}"] = param.clone().detach()

        # Value function parameters
        for name, param in self.value_function.named_parameters():
            params[f"value_function.{name}"] = param.clone().detach()

        return params

    def set_model_parameters(self, params: Dict[str, torch.Tensor]) -> None:
        """Set model parameters from federated update.

        Args:
            params: Dictionary of model parameters
        """
        # Update policy parameters
        policy_state_dict = {}
        value_state_dict = {}

        for name, param in params.items():
            if name.startswith("policy."):
                policy_name = name[7:]  # Remove "policy." prefix
                policy_state_dict[policy_name] = param
            elif name.startswith("value_function."):
                value_name = name[15:]  # Remove "value_function." prefix
                value_state_dict[value_name] = param

        self.policy.load_state_dict(policy_state_dict, strict=False)
        self.value_function.load_state_dict(value_state_dict, strict=False)

    def get_private_update(
        self,
        old_params: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Get privacy-preserving parameter update.

        Args:
            old_params: Previous model parameters

        Returns:
            Private parameter update
        """
        current_params = self.get_model_parameters()

        # Compute parameter difference
        param_update = {}
        for name in current_params.keys():
            if name in old_params:
                param_update[name] = current_params[name] - old_params[name]
            else:
                param_update[name] = current_params[name]

        # Apply differential privacy
        private_update = self.privacy.privatize_gradients(param_update)

        return private_update

    def save_checkpoint(self, filepath: str) -> None:
        """Save model checkpoint.

        Args:
            filepath: Path to save checkpoint
        """
        checkpoint = {
            "policy_state_dict": self.policy.state_dict(),
            "value_function_state_dict": self.value_function.state_dict(),
            "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
            "value_optimizer_state_dict": self.value_optimizer.state_dict(),
            "training_stats": self.training_stats,
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Saved PPO checkpoint to {filepath}")

    def load_checkpoint(self, filepath: str) -> None:
        """Load model checkpoint.

        Args:
            filepath: Path to checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.value_function.load_state_dict(checkpoint["value_function_state_dict"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])
        self.value_optimizer.load_state_dict(checkpoint["value_optimizer_state_dict"])
        self.training_stats = checkpoint["training_stats"]

        logger.info(f"Loaded PPO checkpoint from {filepath}")
