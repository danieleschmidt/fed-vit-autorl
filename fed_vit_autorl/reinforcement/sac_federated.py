"""Federated Soft Actor-Critic (SAC) implementation for distributed RL training.

Generation 1: Simple SAC with basic federated aggregation capability.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
import numpy as np


class FederatedSAC:
    """Simple Federated Soft Actor-Critic implementation.
    
    Generation 1: Basic functionality with simple aggregation.
    """
    
    def __init__(
        self,
        state_dim: int = 768,
        action_dim: int = 4,
        hidden_dim: int = 256,
        learning_rate: float = 3e-4,
        tau: float = 0.005,
        alpha: float = 0.2,
        device: str = "cpu"
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.tau = tau
        self.alpha = alpha
        self.device = torch.device(device)
        
        # Actor network
        self.actor = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        
        # Critic networks
        self.critic1 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic2 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        
        # Target critic networks
        self.target_critic1 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_critic2 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        
        # Copy weights to target networks
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate)
        
    def act(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Select action for given state."""
        with torch.no_grad():
            state = state.to(self.device)
            if len(state.shape) == 1:
                state = state.unsqueeze(0)
            
            if deterministic:
                action, _ = self.actor.sample(state, deterministic=True)
            else:
                action, _ = self.actor.sample(state)
            
            return action.cpu()
    
    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        gamma: float = 0.99
    ) -> Dict[str, float]:
        """Update SAC networks with batch of experiences."""
        
        batch_size = states.size(0)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Update critic networks
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            target_q1 = self.target_critic1(next_states, next_actions)
            target_q2 = self.target_critic2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * gamma * target_q
        
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Update actor network
        new_actions, log_probs = self.actor.sample(states)
        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.alpha * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update target networks
        self._soft_update(self.target_critic1, self.critic1)
        self._soft_update(self.target_critic2, self.critic2)
        
        return {
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'actor_loss': actor_loss.item()
        }
    
    def _soft_update(self, target_network, source_network):
        """Soft update target network weights."""
        for target_param, source_param in zip(target_network.parameters(), source_network.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def get_model_state(self) -> Dict[str, Any]:
        """Get current model state for federated aggregation."""
        return {
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict()
        }
    
    def set_model_state(self, state_dict: Dict[str, Any]):
        """Set model state from federated aggregation."""
        if 'actor' in state_dict:
            self.actor.load_state_dict(state_dict['actor'])
        if 'critic1' in state_dict:
            self.critic1.load_state_dict(state_dict['critic1'])
        if 'critic2' in state_dict:
            self.critic2.load_state_dict(state_dict['critic2'])
            
        # Update target networks
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())


class PolicyNetwork(nn.Module):
    """Actor network for SAC."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through policy network."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        
        return mean, log_std
    
    def sample(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy distribution."""
        mean, log_std = self.forward(state)
        
        if deterministic:
            action = torch.tanh(mean)
            log_prob = torch.zeros_like(action)
        else:
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()
            action = torch.tanh(x_t)
            
            # Compute log probability with change of variables formula
            log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob


class QNetwork(nn.Module):
    """Critic network for SAC."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass through Q-network."""
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        
        return q_value