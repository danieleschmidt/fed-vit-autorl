"""
RLHyperparamTuner: Multi-armed bandit (UCB1) for federated hyperparameter optimization.

Uses Upper Confidence Bound (UCB1) to balance exploration/exploitation across
discrete hyperparameter arms (learning rate × batch size combinations).

Each federation round, the tuner:
1. Selects the arm with highest UCB score
2. Provides the chosen (lr, batch_size) to clients
3. Observes the federated accuracy as reward
4. Updates arm statistics
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class Arm:
    """A single hyperparameter configuration arm."""
    lr: float
    batch_size: int
    n_pulls: int = 0
    total_reward: float = 0.0
    rewards: List[float] = field(default_factory=list)

    @property
    def mean_reward(self) -> float:
        return self.total_reward / self.n_pulls if self.n_pulls > 0 else 0.0

    def ucb_score(self, total_pulls: int, c: float = 1.4) -> float:
        """
        UCB1 score: mean + c * sqrt(ln(t) / n)

        Args:
            total_pulls: Total number of pulls across all arms
            c: Exploration constant (higher = more exploration)
        """
        if self.n_pulls == 0:
            return float("inf")  # Force exploration of unpulled arms
        exploration = c * math.sqrt(math.log(total_pulls) / self.n_pulls)
        return self.mean_reward + exploration

    def update(self, reward: float) -> None:
        self.n_pulls += 1
        self.total_reward += reward
        self.rewards.append(reward)

    def __repr__(self) -> str:
        return (
            f"Arm(lr={self.lr:.0e}, bs={self.batch_size}, "
            f"n={self.n_pulls}, mean={self.mean_reward:.4f})"
        )


class RLHyperparamTuner:
    """
    Multi-armed bandit (UCB1) for joint learning rate + batch size selection.

    Args:
        lr_options: List of candidate learning rates
        batch_size_options: List of candidate batch sizes
        exploration_c: UCB1 exploration constant
        seed: Random seed (unused here since UCB is deterministic)
    """

    def __init__(
        self,
        lr_options: Optional[List[float]] = None,
        batch_size_options: Optional[List[int]] = None,
        exploration_c: float = 1.4,
    ):
        if lr_options is None:
            lr_options = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
        if batch_size_options is None:
            batch_size_options = [16, 32, 64]

        self.exploration_c = exploration_c
        self.arms: List[Arm] = [
            Arm(lr=lr, batch_size=bs)
            for lr in lr_options
            for bs in batch_size_options
        ]
        self.total_pulls = 0
        self.history: List[dict] = []

    def select(self) -> Tuple[float, int, int]:
        """
        Select the arm with highest UCB score.

        Returns:
            (lr, batch_size, arm_index)
        """
        scores = [arm.ucb_score(max(1, self.total_pulls), self.exploration_c) for arm in self.arms]
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        arm = self.arms[best_idx]
        return arm.lr, arm.batch_size, best_idx

    def update(self, arm_idx: int, reward: float) -> None:
        """
        Update arm statistics with observed reward.

        Args:
            arm_idx: Index of the arm that was pulled
            reward: Observed reward (e.g., federated accuracy in [0, 1])
        """
        self.arms[arm_idx].update(reward)
        self.total_pulls += 1
        self.history.append({
            "round": self.total_pulls,
            "arm_idx": arm_idx,
            "lr": self.arms[arm_idx].lr,
            "batch_size": self.arms[arm_idx].batch_size,
            "reward": reward,
        })

    def best_arm(self) -> Arm:
        """Return the arm with highest mean reward (among pulled arms)."""
        pulled = [arm for arm in self.arms if arm.n_pulls > 0]
        if not pulled:
            return self.arms[0]
        return max(pulled, key=lambda a: a.mean_reward)

    def summary(self) -> str:
        """Human-readable summary of arm statistics."""
        lines = ["RLHyperparamTuner Summary (UCB1)", "=" * 40]
        for i, arm in enumerate(self.arms):
            marker = " ← best" if arm is self.best_arm() else ""
            lines.append(f"  [{i:2d}] {arm}{marker}")
        lines.append(f"\n  Total pulls: {self.total_pulls}")
        best = self.best_arm()
        lines.append(f"  Best config: lr={best.lr:.0e}, batch_size={best.batch_size}")
        lines.append(f"  Best mean reward: {best.mean_reward:.4f}")
        return "\n".join(lines)
