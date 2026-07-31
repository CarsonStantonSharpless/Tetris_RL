from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import random

import numpy as np

from training.evaluation.specs import PolicySpec


@dataclass(frozen=True)
class EpisodeResult:
    seed: int
    score: int
    ticks: int
    game_over: bool


def run_episode(
    policy: PolicySpec,
    seed: int,
    max_ticks: int = 1000,
) -> EpisodeResult:
    """Run one reproducible, headless episode."""
    if max_ticks < 1:
        raise ValueError("max_ticks must be at least one")

    random_state = random.getstate()
    numpy_state = np.random.get_state()

    try:
        random.seed(seed)
        np.random.seed(seed)

        player = policy.build_player()
        state = player.start(max_ticks=max_ticks)
        return EpisodeResult(
            seed=seed,
            score=state.score,
            ticks=player.engine.ticks,
            game_over=player.engine.is_game_over,
        )
    finally:
        random.setstate(random_state)
        np.random.set_state(numpy_state)


def evaluate(
    policy: PolicySpec,
    seeds: Sequence[int],
    max_ticks: int = 1000,
) -> list[EpisodeResult]:
    """Evaluate a policy serially on each seed."""
    return [run_episode(policy, seed, max_ticks) for seed in seeds]
