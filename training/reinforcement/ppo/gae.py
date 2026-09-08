"""Generalized Advantage Estimation (GAE), kept separate from PPO.

The critic emits values ``V(s_t)``.  The environment emits rewards and done
flags.  GAE combines those two streams into an advantage for the actor and a
return target for the critic; GAE itself is not another neural network.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GAEOutput:
    """The three teaching signals produced by the GAE calculation."""

    td_residuals: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray


def generalized_advantage_estimate(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    next_value: float,
    discount: float = 0.99,
    gae_lambda: float = 0.95,
) -> GAEOutput:
    """Compute TD residuals, GAE advantages, and critic return targets.

    For each step, the one-step surprise is::

        delta_t = reward_t + gamma * V(s_(t+1)) - V(s_t)

    GAE then carries future surprises backward with ``gamma * lambda``.
    A terminal flag cuts both calculations so a new episode cannot leak into
    the previous one.  ``next_value`` is zero after game-over and is the
    critic's bootstrap value when a rollout was merely truncated.
    """
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.bool_)
    _validate_inputs(rewards, values, dones, discount, gae_lambda)

    td_residuals = np.zeros_like(rewards)
    advantages = np.zeros_like(rewards)
    following_value = float(next_value)
    following_advantage = 0.0

    # Work backward because today's advantage contains tomorrow's advantage.
    for index in range(len(rewards) - 1, -1, -1):
        # A terminal state has no future reward and therefore no bootstrap.
        continues = 0.0 if dones[index] else 1.0
        # This is the critic's one-step error: observed reward plus its next
        # estimate, minus the estimate it made before the action.
        td_residuals[index] = (
            rewards[index]
            + discount * following_value * continues
            - values[index]
        )
        # Lambda controls how many future TD errors receive meaningful weight.
        advantages[index] = (
            td_residuals[index]
            + discount * gae_lambda * following_advantage * continues
        )
        following_value = float(values[index])
        following_advantage = float(advantages[index])

    # V(s) + advantage is the regression target that trains the critic.
    returns = values + advantages
    return GAEOutput(td_residuals, advantages, returns)


def _validate_inputs(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    discount: float,
    gae_lambda: float,
) -> None:
    if rewards.ndim != 1 or values.ndim != 1 or dones.ndim != 1:
        raise ValueError("rewards, values, and dones must be one-dimensional")
    if not len(rewards):
        raise ValueError("GAE requires at least one transition")
    if not (len(rewards) == len(values) == len(dones)):
        raise ValueError("rewards, values, and dones must have equal lengths")
    if not 0 <= discount <= 1:
        raise ValueError("discount must be between zero and one")
    if not 0 <= gae_lambda <= 1:
        raise ValueError("gae_lambda must be between zero and one")
