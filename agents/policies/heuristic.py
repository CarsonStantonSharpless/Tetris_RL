import numpy as np

from core.board import BoardState
from agents.placements import find_possible_states
from storage.parameters import read_heuristic_parameters


"""
Credit to: https://codemyroad.wordpress.com/2013/04/14/tetris-ai-the-near-perfect-player/
"""

DEFAULT_ALPHA = .3
DEFAULT_BETA = .3
DEFAULT_GAMMA = .1
DEFAULT_DELTA = .2
DEFAULT_EPSILON = .1


def heuristic_policy(
    start_state: BoardState,
    harddrop: bool = True,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    gamma: float = DEFAULT_GAMMA,
    delta: float = DEFAULT_DELTA,
    epsilon: float = DEFAULT_EPSILON,
    filepath: str | None = None,
    next_piece: str | None = None,
    level: int = 0,
) -> BoardState:
    del next_piece, level
    if filepath is not None:
        alpha, beta, gamma, delta, epsilon = read_heuristic_parameters(
            filepath
        ).values()
    poss_states: list[BoardState] = find_possible_states(start_state, harddrop)

    #simple in concept, evaluate each possible state, and give it a score, highest wins
    max_score: float = -float('inf')
    best_state: BoardState | None = None
    for state in poss_states:
        score: float = evaluate(
            state.locked,
            alpha,
            beta,
            gamma,
            delta,
            epsilon,
        )
        if score > max_score:
            max_score = score
            best_state = state
    
    assert(best_state is not None)
    return best_state


def genetic_heuristic_policy(
    start_state: BoardState,
    filepath: str | None = None,
    harddrop: bool = True,
    next_piece: str | None = None,
    level: int = 0,
) -> BoardState:
    return heuristic_policy(
        start_state,
        harddrop,
        filepath=filepath,
        next_piece=next_piece,
        level=level,
    )


def evaluate(
        grid: np.ndarray,
        alpha: float = DEFAULT_ALPHA,
        beta: float = DEFAULT_BETA,
        gamma: float = DEFAULT_GAMMA,
        delta: float = DEFAULT_DELTA,
        epsilon: float = DEFAULT_EPSILON,
        ) -> float:

    return (
        alpha*aggregate_height(grid) +
        beta*complete_lines(grid) +
        gamma*bumpiness(grid) +
        delta*holes(grid) +
        epsilon*tetris_setup(grid)
    )

def evaluate_features(grid: np.ndarray) -> np.ndarray:
    return np.array([
        aggregate_height(grid),
        complete_lines(grid),
        bumpiness(grid),
        holes(grid),
        tetris_setup(grid)
    ])


def top_heuristic_placements(
    placements: list[BoardState],
    top_k: int | None,
) -> list[BoardState]:
    """Return the highest-scoring placements under the existing heuristic."""
    if top_k is not None and top_k < 1:
        raise ValueError("heuristic top-k must be at least one")
    if top_k is None or top_k >= len(placements):
        return placements
    return sorted(
        placements,
        key=lambda placement: evaluate(placement.locked),
        reverse=True,
    )[:top_k]


def aggregate_height(grid: np.ndarray) -> float:
    heights = column_heights(grid)
    return max(0.0, 1 - (float(heights.sum()) / (20*10)))

def complete_lines(grid: np.ndarray) -> float:
    return float(np.all(grid != 0, axis=1).sum()) / 4

def bumpiness(grid: np.ndarray) -> float:
    heights = column_heights(grid)
    maximum = (grid.shape[1] - 1) * grid.shape[0]
    return 1 - float(np.abs(np.diff(heights)).sum()) / maximum

def holes(grid: np.ndarray) -> float:
    filled = grid != 0
    covered = np.maximum.accumulate(filled, axis=0)
    return 1 - float(np.count_nonzero(covered & ~filled)) / grid.size

def tetris_setup(grid: np.ndarray, well_col: int = -1) -> float:
    other_cols = np.delete(grid, well_col, axis=1)
    ready_rows = np.all(other_cols != 0, axis=1) & (grid[:, well_col] == 0)
    return min(float(np.count_nonzero(ready_rows)), 4.0) / 4.0


def column_heights(grid: np.ndarray) -> np.ndarray:
    filled = grid != 0
    first = np.argmax(filled, axis=0)
    return np.where(filled.any(axis=0), grid.shape[0] - first, 0)
