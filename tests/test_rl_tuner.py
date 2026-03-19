"""Tests for RLHyperparamTuner (UCB1 bandit)."""

import math
import pytest
from fed_vit_autorl.rl_tuner import RLHyperparamTuner, Arm


class TestArm:
    def test_initial_state(self):
        arm = Arm(lr=1e-3, batch_size=32)
        assert arm.n_pulls == 0
        assert arm.total_reward == 0.0
        assert arm.mean_reward == 0.0

    def test_update(self):
        arm = Arm(lr=1e-3, batch_size=32)
        arm.update(0.8)
        assert arm.n_pulls == 1
        assert arm.mean_reward == pytest.approx(0.8)

    def test_mean_reward_avg(self):
        arm = Arm(lr=1e-3, batch_size=32)
        arm.update(0.4)
        arm.update(0.6)
        assert arm.mean_reward == pytest.approx(0.5)

    def test_ucb_infinite_for_unpulled(self):
        arm = Arm(lr=1e-3, batch_size=32)
        assert arm.ucb_score(total_pulls=10) == float("inf")

    def test_ucb_finite_for_pulled(self):
        arm = Arm(lr=1e-3, batch_size=32)
        arm.update(0.5)
        score = arm.ucb_score(total_pulls=1)
        assert math.isfinite(score)
        assert score >= 0.5  # at least mean_reward


class TestRLHyperparamTuner:
    def setup_method(self):
        self.tuner = RLHyperparamTuner(
            lr_options=[1e-3, 1e-2],
            batch_size_options=[16, 32],
        )

    def test_num_arms(self):
        assert len(self.tuner.arms) == 4  # 2 lrs × 2 bss

    def test_select_returns_valid_config(self):
        lr, bs, idx = self.tuner.select()
        assert lr in [1e-3, 1e-2]
        assert bs in [16, 32]
        assert 0 <= idx < len(self.tuner.arms)

    def test_unpulled_arms_selected_first(self):
        """UCB1 must pull all arms before repeating."""
        seen_idxs = set()
        for _ in range(len(self.tuner.arms)):
            lr, bs, idx = self.tuner.select()
            # Simulate reward
            self.tuner.update(idx, reward=0.5)
            seen_idxs.add(idx)
        assert len(seen_idxs) == len(self.tuner.arms)

    def test_update_increments_total_pulls(self):
        _, _, idx = self.tuner.select()
        self.tuner.update(idx, reward=0.7)
        assert self.tuner.total_pulls == 1

    def test_best_arm_after_updates(self):
        """Arm with highest reward should be selected as best."""
        # Pull all arms once
        for i in range(len(self.tuner.arms)):
            _, _, idx = self.tuner.select()
            # Give arm 0 highest reward
            reward = 0.9 if idx == 0 else 0.3
            self.tuner.update(idx, reward=reward)

        best = self.tuner.best_arm()
        assert best.lr == self.tuner.arms[0].lr or best.mean_reward >= 0.9

    def test_history_recorded(self):
        for _ in range(3):
            _, _, idx = self.tuner.select()
            self.tuner.update(idx, reward=0.5)
        assert len(self.tuner.history) == 3

    def test_summary_string(self):
        _, _, idx = self.tuner.select()
        self.tuner.update(idx, reward=0.6)
        summary = self.tuner.summary()
        assert "UCB1" in summary
        assert "Total pulls" in summary

    def test_exploration_favors_high_reward(self):
        """After many pulls, UCB should favor arm with higher mean reward."""
        tuner = RLHyperparamTuner(
            lr_options=[1e-3, 1e-2],
            batch_size_options=[32],
            exploration_c=0.1,  # low exploration
        )
        # Warm up all arms
        for i in range(len(tuner.arms)):
            _, _, idx = tuner.select()
            reward = 0.9 if tuner.arms[idx].lr == 1e-3 else 0.1
            tuner.update(idx, reward=reward)

        # Now pull multiple times — should mostly pick lr=1e-3
        picks = []
        for _ in range(10):
            lr, _, idx = tuner.select()
            tuner.update(idx, reward=0.9 if lr == 1e-3 else 0.1)
            picks.append(lr)

        assert picks.count(1e-3) > picks.count(1e-2)
