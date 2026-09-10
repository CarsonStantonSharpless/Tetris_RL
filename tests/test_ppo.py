"""Focused tests for PPO's model boundary and standalone GAE calculation."""

import unittest

import numpy as np

from agents.policies.dqn import require_torch
from agents.policies.ppo import PPO
from training.reinforcement.ppo.gae import generalized_advantage_estimate
from training.reinforcement.ppo.trainer import (
    PPOConfig,
    _bottom_up_potential,
    _bottom_up_reward,
)


class GAETests(unittest.TestCase):
    def test_terminal_transition_cuts_off_bootstrap(self) -> None:
        result = generalized_advantage_estimate(
            rewards=np.array([1.0, 1.0]),
            values=np.array([0.5, 0.25]),
            dones=np.array([False, True]),
            next_value=99.0,
            discount=0.9,
            gae_lambda=0.8,
        )

        np.testing.assert_allclose(result.td_residuals, [0.725, 0.75])
        np.testing.assert_allclose(result.advantages, [1.265, 0.75])
        np.testing.assert_allclose(result.returns, [1.765, 1.0])

    def test_truncated_rollout_uses_bootstrap_value(self) -> None:
        result = generalized_advantage_estimate(
            rewards=np.array([1.0]),
            values=np.array([0.5]),
            dones=np.array([False]),
            next_value=2.0,
            discount=0.9,
        )

        np.testing.assert_allclose(result.td_residuals, [2.3])
        np.testing.assert_allclose(result.advantages, [2.3])
        np.testing.assert_allclose(result.returns, [2.8])


class PPOModelTests(unittest.TestCase):
    def test_default_collects_a_fixed_number_of_transitions(self) -> None:
        self.assertEqual(PPOConfig().rollout_steps, 1_024)
        self.assertEqual(PPOConfig().bottom_up_bias, 0.0)

    def test_bottom_up_hint_is_small_and_potential_based(self) -> None:
        empty = np.zeros((22, 10), dtype=np.uint8)
        low_piece = empty.copy()
        low_piece[-1, :4] = 1
        covered_hole = empty.copy()
        covered_hole[-2, 0] = 1

        self.assertEqual(_bottom_up_potential(empty), 22.0)
        self.assertEqual(_bottom_up_potential(low_piece), 21.0)
        self.assertEqual(_bottom_up_potential(covered_hole), 18.0)
        self.assertEqual(
            _bottom_up_reward(empty, low_piece, done=False, discount=1.0),
            -1.0,
        )
        self.assertEqual(
            _bottom_up_reward(low_piece, low_piece, done=True, discount=1.0),
            -21.0,
        )

    def test_actor_scores_actions_and_critic_scores_states(self) -> None:
        torch = require_torch()
        model = PPO()
        action_boards = torch.zeros((3, 22, 10))
        state_boards = torch.zeros((2, 22, 10))

        logits = model.policy_logits(
            action_boards,
            torch.tensor([1, 1, 1]),
            torch.tensor([2, 2, 2]),
            torch.tensor([0.0, 0.0, 0.0]),
        )
        values = model.state_values(
            state_boards,
            torch.tensor([1, 3]),
            torch.tensor([2, 4]),
            torch.tensor([0.0, 1.0]),
        )

        self.assertEqual(tuple(logits.shape), (3,))
        self.assertEqual(tuple(values.shape), (2,))


if __name__ == "__main__":
    unittest.main()
