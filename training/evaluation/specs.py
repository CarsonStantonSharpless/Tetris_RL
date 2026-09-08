from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from agents.player import Player, Policy
from training.genetic.fighter import Genome


class PolicySpec(Protocol):
    """Serializable instructions for constructing an evaluation player."""

    def build_player(self) -> Player:
        """Create a fresh player for one episode."""


@dataclass(frozen=True)
class HeuristicSpec:
    genome: Genome

    def build_player(self) -> Player:
        return Player(
            policy=Policy.HEURISTIC,
            display=False,
            instant_placement=True,
            policy_params={
                "alpha": self.genome.alpha,
                "beta": self.genome.beta,
                "gamma": self.genome.gamma,
                "delta": self.genome.delta,
                "epsilon": self.genome.epsilon,
            },
        )


@dataclass(frozen=True)
class RandomSpec:
    def build_player(self) -> Player:
        return Player(
            policy=Policy.RANDOM,
            display=False,
            instant_placement=True,
        )


@dataclass(frozen=True)
class DQNPolicySpec:
    """Serializable instructions for evaluating a saved DQN model."""

    filepath: str
    device: str = "cpu"
    heuristic_top_k: int | None = None

    def build_player(self) -> Player:
        return Player(
            policy=Policy.DQN,
            display=False,
            instant_placement=True,
            policy_params={
                "filepath": self.filepath,
                "device": self.device,
                "heuristic_top_k": self.heuristic_top_k,
            },
        )


@dataclass(frozen=True)
class PPOPolicySpec:
    """Serializable instructions for evaluating a saved PPO actor."""

    filepath: str
    device: str = "cpu"
    heuristic_top_k: int | None = None

    def build_player(self) -> Player:
        return Player(
            policy=Policy.PPO,
            display=False,
            instant_placement=True,
            policy_params={
                "filepath": self.filepath,
                "device": self.device,
                "heuristic_top_k": self.heuristic_top_k,
            },
        )
