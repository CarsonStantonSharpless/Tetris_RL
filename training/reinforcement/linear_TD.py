import random
from functools import partial
import multiprocessing
from datetime import datetime
from pathlib import Path

import numpy as np

from agents.player import Player
from agents.policies.heuristic import evaluate_features
from storage.parameters import write_checkpoint, write_weights
from visualization.training import TrainingPlot


def linear_TD(
    num_games: int,
    batch_size: int = 4,
    workers: int | None = None,
    learning_rate: float = 1e-4,
    discount: float = 0.99,
    epsilon: float = 0.1,
    max_placements: int = 1000,
    seed: int = 0,
    run_directory: str | Path | None = None,
    display: bool = True,
) -> np.ndarray:
    """Train games in parallel and keep each batch's best weights."""
    if num_games < 1:
        raise ValueError("num_games must be at least one")
    if batch_size < 1:
        raise ValueError("batch_size must be at least one")
    if workers is not None and workers < 1:
        raise ValueError("workers must be at least one")

    weights = np.random.default_rng(seed).dirichlet(np.ones(5))
    bias = 0.0
    run_directory = Path(run_directory or _run_directory())
    plot = TrainingPlot(run_directory / "training.png", display)
    next_checkpoint = 5

    with multiprocessing.Pool(workers) as pool:
        for start in range(0, num_games, batch_size):
            train_game = partial(
                _train_game,
                weights=weights,
                bias=bias,
                learning_rate=learning_rate,
                discount=discount,
                epsilon=epsilon,
                max_placements=max_placements,
            )
            games = pool.map(
                train_game,
                range(seed + start, seed + min(start + batch_size, num_games)),
            )
            weights, bias, _, _ = max(games, key=lambda game: game[2])
            plot.update(
                [loss for _, _, _, loss in games],
                [score for _, _, score, _ in games],
            )

            progress = 100 * min(start + batch_size, num_games) / num_games
            while progress >= next_checkpoint:
                write_checkpoint(
                    normalized(weights),
                    next_checkpoint,
                    run_directory / "checkpoints.json",
                )
                next_checkpoint += 5

    weights = normalized(weights)
    write_weights(weights, run_directory / "winner.json")
    return weights


def _train_game(
    seed: int,
    weights: np.ndarray,
    bias: float,
    learning_rate: float,
    discount: float,
    epsilon: float,
    max_placements: int,
) -> tuple[np.ndarray, float, int, float]:
    """Run one independent TD-learning game."""
    random.seed(seed)
    weights = weights.copy()
    player = Player(display=False, instant_placement=True)
    losses = []

    while (
        not player.engine.is_game_over
        and player.engine.ticks < max_placements
    ):
        placements, features = placement_options(player)

        values = features @ weights + bias
        if random.random() < epsilon:
            choice = random.randrange(len(placements))
        else:
            choice = int(np.argmax(values))

        transition = player.place(placements[choice])

        next_value = 0.0
        if not transition.done:
            _, next_features = placement_options(player)
            next_value = np.max(next_features @ weights + bias)

        td_error = transition.reward + discount * next_value - values[choice]
        weights += learning_rate * td_error * features[choice]
        weights = np.maximum(weights, 0)
        bias += learning_rate * td_error
        losses.append(td_error ** 2)

    return weights, bias, player.state.score, float(np.mean(losses))


def placement_options(player: Player):
    placements = player.possible_placements()
    features = np.array([
        evaluate_features(player.simulate_placement(move).locked_grid)
        for move in placements
    ])
    return placements, features


def normalized(weights: np.ndarray) -> np.ndarray:
    total = weights.sum()
    return weights / total if total else weights


def _run_directory() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("runs") / f"linear_td_{timestamp}"
