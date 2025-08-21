"""Experience replay buffers for reinforcement learning."""

import random
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import torch
import numpy as np
from collections import deque, namedtuple
import pickle


logger = logging.getLogger(__name__)

# Experience tuple for replay buffer
Experience = namedtuple(
    "Experience",
    ["state", "action", "reward", "next_state", "done", "info"]
)


class ExperienceReplayBuffer:
    """Experience replay buffer for storing and sampling transitions."""

    def __init__(
        self,
        capacity: int = 100000,
        device: str = "cpu",
        save_info: bool = False,
    ):
        """Initialize experience replay buffer.

        Args:
            capacity: Maximum number of experiences to store
            device: Device to store tensors on
            save_info: Whether to save info dict from environment
        """
        self.capacity = capacity
        self.device = device
        self.save_info = save_info

        self.buffer = deque(maxlen=capacity)
        self.position = 0

        logger.info(f"Initialized replay buffer with capacity {capacity}")

    def add(
        self,
        state: Union[torch.Tensor, np.ndarray],
        action: Union[torch.Tensor, np.ndarray],
        reward: float,
        next_state: Union[torch.Tensor, np.ndarray],
        done: bool,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add experience to buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            info: Additional information
        """
        # Convert to tensors if necessary
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float()
        if isinstance(next_state, np.ndarray):
            next_state = torch.from_numpy(next_state).float()

        # Move to device
        state = state.to(self.device)
        action = action.to(self.device)
        next_state = next_state.to(self.device)

        # Create experience
        experience = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            info=info if self.save_info else None,
        )

        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample batch of experiences.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        if len(self.buffer) < batch_size:
            raise ValueError(f"Not enough experiences: {len(self.buffer)} < {batch_size}")

        # Random sampling
        experiences = random.sample(self.buffer, batch_size)

        # Unpack experiences
        states = torch.stack([exp.state for exp in experiences])
        actions = torch.stack([exp.action for exp in experiences])
        rewards = torch.tensor([exp.reward for exp in experiences], dtype=torch.float32, device=self.device)
        next_states = torch.stack([exp.next_state for exp in experiences])
        dones = torch.tensor([exp.done for exp in experiences], dtype=torch.bool, device=self.device)

        return states, actions, rewards, next_states, dones

    def sample_sequential(
        self,
        sequence_length: int,
        batch_size: int,
    ) -> Tuple[torch.Tensor, ...]:
        """Sample sequential experiences for RNN training.

        Args:
            sequence_length: Length of sequences to sample
            batch_size: Number of sequences to sample

        Returns:
            Tuple of batched sequences
        """
        if len(self.buffer) < sequence_length:
            raise ValueError(f"Not enough experiences for sequences: {len(self.buffer)} < {sequence_length}")

        sequences = []

        for _ in range(batch_size):
            # Sample random start position
            start_idx = random.randint(0, len(self.buffer) - sequence_length)
            sequence = list(self.buffer)[start_idx:start_idx + sequence_length]
            sequences.append(sequence)

        # Stack sequences
        states = torch.stack([
            torch.stack([exp.state for exp in seq]) for seq in sequences
        ])  # (batch_size, seq_len, state_dim)

        actions = torch.stack([
            torch.stack([exp.action for exp in seq]) for seq in sequences
        ])  # (batch_size, seq_len, action_dim)

        rewards = torch.tensor([
            [exp.reward for exp in seq] for seq in sequences
        ], dtype=torch.float32, device=self.device)  # (batch_size, seq_len)

        next_states = torch.stack([
            torch.stack([exp.next_state for exp in seq]) for seq in sequences
        ])  # (batch_size, seq_len, state_dim)

        dones = torch.tensor([
            [exp.done for exp in seq] for seq in sequences
        ], dtype=torch.bool, device=self.device)  # (batch_size, seq_len)

        return states, actions, rewards, next_states, dones

    def get_all_experiences(self) -> List[Experience]:
        """Get all experiences in buffer.

        Returns:
            List of all experiences
        """
        return list(self.buffer)

    def clear(self) -> None:
        """Clear all experiences from buffer."""
        self.buffer.clear()
        self.position = 0

    def __len__(self) -> int:
        """Get number of experiences in buffer."""
        return len(self.buffer)

    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics.

        Returns:
            Dictionary of buffer statistics
        """
        if not self.buffer:
            return {"size": 0, "capacity": self.capacity}

        rewards = [exp.reward for exp in self.buffer]

        return {
            "size": len(self.buffer),
            "capacity": self.capacity,
            "utilization": len(self.buffer) / self.capacity,
            "avg_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards),
        }

    def save(self, filepath: str) -> None:
        """Save buffer to file.

        Args:
            filepath: Path to save buffer
        """
        buffer_data = {
            "buffer": list(self.buffer),
            "capacity": self.capacity,
            "device": self.device,
            "save_info": self.save_info,
        }

        with open(filepath, "wb") as f:
            pickle.dump(buffer_data, f)

        logger.info(f"Saved replay buffer to {filepath}")

    def load(self, filepath: str) -> None:
        """Load buffer from file.

        Args:
            filepath: Path to buffer file
        """
        with open(filepath, "rb") as f:
            buffer_data = pickle.load(f)

        self.buffer = deque(buffer_data["buffer"], maxlen=buffer_data["capacity"])
        self.capacity = buffer_data["capacity"]
        self.device = buffer_data["device"]
        self.save_info = buffer_data["save_info"]

        logger.info(f"Loaded replay buffer from {filepath}")


class PrioritizedReplayBuffer(ExperienceReplayBuffer):
    """Prioritized experience replay buffer using TD-error for prioritization."""

    def __init__(
        self,
        capacity: int = 100000,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 1e-6,
        device: str = "cpu",
        save_info: bool = False,
    ):
        """Initialize prioritized replay buffer.

        Args:
            capacity: Maximum number of experiences
            alpha: Prioritization exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling exponent (0 = no correction, 1 = full correction)
            beta_increment: Beta increment per sampling step
            epsilon: Small constant to avoid zero priorities
            device: Device to store tensors on
            save_info: Whether to save info dict
        """
        super().__init__(capacity, device, save_info)

        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon

        # Priority storage
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0

        logger.info(f"Initialized prioritized replay buffer with α={alpha}, β={beta}")

    def add(
        self,
        state: Union[torch.Tensor, np.ndarray],
        action: Union[torch.Tensor, np.ndarray],
        reward: float,
        next_state: Union[torch.Tensor, np.ndarray],
        done: bool,
        info: Optional[Dict[str, Any]] = None,
        priority: Optional[float] = None,
    ) -> None:
        """Add experience with priority to buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            info: Additional information
            priority: Priority for this experience (uses max priority if None)
        """
        # Add experience to base buffer
        super().add(state, action, reward, next_state, done, info)

        # Add priority
        if priority is None:
            priority = self.max_priority

        self.priorities.append(priority)
        self.max_priority = max(self.max_priority, priority)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample batch of experiences based on priorities.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones, weights, indices)
        """
        if len(self.buffer) < batch_size:
            raise ValueError(f"Not enough experiences: {len(self.buffer)} < {batch_size}")

        # Calculate sampling probabilities
        priorities = np.array(self.priorities, dtype=np.float32)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        # Sample indices based on probabilities
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)

        # Get experiences
        experiences = [self.buffer[idx] for idx in indices]

        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize weights
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

        # Unpack experiences
        states = torch.stack([exp.state for exp in experiences])
        actions = torch.stack([exp.action for exp in experiences])
        rewards = torch.tensor([exp.reward for exp in experiences], dtype=torch.float32, device=self.device)
        next_states = torch.stack([exp.next_state for exp in experiences])
        dones = torch.tensor([exp.done for exp in experiences], dtype=torch.bool, device=self.device)

        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        return states, actions, rewards, next_states, dones, weights, indices

    def update_priorities(self, indices: List[int], priorities: torch.Tensor) -> None:
        """Update priorities for sampled experiences.

        Args:
            indices: Indices of experiences to update
            priorities: New priorities (typically TD-errors)
        """
        priorities = priorities.detach().cpu().numpy()

        for idx, priority in zip(indices, priorities):
            if 0 <= idx < len(self.priorities):
                # Clip priority to avoid extreme values
                priority = np.clip(priority, self.epsilon, None)
                self.priorities[idx] = priority
                self.max_priority = max(self.max_priority, priority)

    def clear(self) -> None:
        """Clear all experiences and priorities from buffer."""
        super().clear()
        self.priorities.clear()
        self.max_priority = 1.0


class VehicleExperienceBuffer(ExperienceReplayBuffer):
    """Specialized experience buffer for autonomous driving scenarios."""

    def __init__(
        self,
        capacity: int = 100000,
        device: str = "cpu",
        scenario_weighting: bool = True,
        safety_prioritization: bool = True,
    ):
        """Initialize vehicle experience buffer.

        Args:
            capacity: Maximum number of experiences
            device: Device to store tensors on
            scenario_weighting: Whether to weight different driving scenarios
            safety_prioritization: Whether to prioritize safety-critical experiences
        """
        super().__init__(capacity, device, save_info=True)

        self.scenario_weighting = scenario_weighting
        self.safety_prioritization = safety_prioritization

        # Scenario tracking
        self.scenario_counts = {}
        self.safety_critical_indices = set()

        logger.info(
            f"Initialized vehicle experience buffer with scenario weighting: "
            f"{scenario_weighting}, safety prioritization: {safety_prioritization}"
        )

    def add(
        self,
        state: Union[torch.Tensor, np.ndarray],
        action: Union[torch.Tensor, np.ndarray],
        reward: float,
        next_state: Union[torch.Tensor, np.ndarray],
        done: bool,
        scenario_type: Optional[str] = None,
        safety_critical: bool = False,
        collision_risk: float = 0.0,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add driving experience to buffer.

        Args:
            state: Current state (sensor observations)
            action: Action taken (steering, throttle, brake)
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            scenario_type: Type of driving scenario
            safety_critical: Whether this is a safety-critical situation
            collision_risk: Risk of collision (0-1)
            info: Additional information
        """
        # Enhance info with driving-specific data
        if info is None:
            info = {}

        info.update({
            "scenario_type": scenario_type,
            "safety_critical": safety_critical,
            "collision_risk": collision_risk,
            "timestamp": len(self.buffer),
        })

        # Add to base buffer
        super().add(state, action, reward, next_state, done, info)

        # Track scenarios
        if scenario_type:
            self.scenario_counts[scenario_type] = self.scenario_counts.get(scenario_type, 0) + 1

        # Mark safety-critical experiences
        if safety_critical or collision_risk > 0.7:
            self.safety_critical_indices.add(len(self.buffer) - 1)

    def sample_balanced(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample experiences with balanced scenario representation.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            Tuple of batched experiences
        """
        if not self.scenario_weighting:
            return self.sample(batch_size)

        if len(self.buffer) < batch_size:
            raise ValueError(f"Not enough experiences: {len(self.buffer)} < {batch_size}")

        # Group experiences by scenario
        scenario_experiences = {}
        for i, exp in enumerate(self.buffer):
            if exp.info and exp.info.get("scenario_type"):
                scenario_type = exp.info["scenario_type"]
                if scenario_type not in scenario_experiences:
                    scenario_experiences[scenario_type] = []
                scenario_experiences[scenario_type].append((i, exp))

        # Sample from each scenario proportionally
        selected_experiences = []

        if scenario_experiences:
            scenarios = list(scenario_experiences.keys())
            samples_per_scenario = max(1, batch_size // len(scenarios))

            for scenario in scenarios:
                scenario_exps = scenario_experiences[scenario]
                num_samples = min(samples_per_scenario, len(scenario_exps))

                # Add safety-critical bias
                if self.safety_prioritization:
                    safety_critical = [
                        (idx, exp) for idx, exp in scenario_exps
                        if idx in self.safety_critical_indices
                    ]
                    regular = [
                        (idx, exp) for idx, exp in scenario_exps
                        if idx not in self.safety_critical_indices
                    ]

                    # Sample 30% from safety-critical, 70% from regular
                    safety_samples = min(int(num_samples * 0.3), len(safety_critical))
                    regular_samples = num_samples - safety_samples

                    if safety_critical:
                        selected_experiences.extend(
                            random.sample(safety_critical, safety_samples)
                        )
                    if regular:
                        selected_experiences.extend(
                            random.sample(regular, min(regular_samples, len(regular)))
                        )
                else:
                    selected_experiences.extend(
                        random.sample(scenario_exps, num_samples)
                    )

        # Fill remaining slots with random samples
        remaining = batch_size - len(selected_experiences)
        if remaining > 0:
            all_indices = set(range(len(self.buffer)))
            used_indices = {idx for idx, _ in selected_experiences}
            available_indices = list(all_indices - used_indices)

            if available_indices:
                additional_indices = random.sample(
                    available_indices, min(remaining, len(available_indices))
                )
                for idx in additional_indices:
                    selected_experiences.append((idx, self.buffer[idx]))

        # Extract experiences
        experiences = [exp for _, exp in selected_experiences[:batch_size]]

        # Unpack experiences
        states = torch.stack([exp.state for exp in experiences])
        actions = torch.stack([exp.action for exp in experiences])
        rewards = torch.tensor([exp.reward for exp in experiences], dtype=torch.float32, device=self.device)
        next_states = torch.stack([exp.next_state for exp in experiences])
        dones = torch.tensor([exp.done for exp in experiences], dtype=torch.bool, device=self.device)

        return states, actions, rewards, next_states, dones

    def get_scenario_stats(self) -> Dict[str, Any]:
        """Get driving scenario statistics.

        Returns:
            Dictionary of scenario statistics
        """
        total_experiences = len(self.buffer)
        safety_critical_count = len(self.safety_critical_indices)

        stats = {
            "total_experiences": total_experiences,
            "safety_critical_count": safety_critical_count,
            "safety_critical_ratio": safety_critical_count / max(1, total_experiences),
            "scenario_counts": self.scenario_counts.copy(),
            "scenario_distribution": {
                scenario: count / max(1, total_experiences)
                for scenario, count in self.scenario_counts.items()
            },
        }

        return stats
