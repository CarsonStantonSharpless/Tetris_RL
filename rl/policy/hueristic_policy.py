import numpy as np

from core.board import BoardState
from core.pieces import Piece
from rl.policy.possible_moves import find_possible_states


"""
Credit to: https://codemyroad.wordpress.com/2013/04/14/tetris-ai-the-near-perfect-player/
"""

def hueristic_policy(start_state: BoardState, harddrop: bool = True) -> BoardState:
    poss_states: list[BoardState] = find_possible_states(start_state, harddrop)

    #simple in concept, evaluate each possible state, and give it a score, highest wins
    max_score: float = -float('inf')
    best_state: BoardState | None = None
    for state in poss_states:
        score: float = evaluate(state.locked)
        if score > max_score:
            max_score = score
            best_state = state
    
    assert(best_state is not None)
    return best_state

def evaluate(
        grid: np.ndarray,
        alpha: float = .33,
        beta: float = .5,
        gamma: float = .1,
        delta: float = .25,
        epsilon: float = 0
        ) -> float:

    return (
        alpha*aggregate_height(grid) +
        beta*complete_lines(grid) +
        gamma*bumpiness(grid) +
        delta*holes(grid) +
        epsilon*tetris(grid)
    )


def aggregate_height(grid: np.ndarray) -> float:
    heights = column_heights(grid)
    return 1 - (float(heights.sum()) / (20*10))

def complete_lines(grid: np.ndarray) -> float:
    return float(np.all(grid != 0, axis=1).sum()) / 4

def bumpiness(grid: np.ndarray) -> float:
    heights = column_heights(grid)
    maximum = (grid.shape[1] - 1) * grid.shape[0]
    return -np.abs(np.diff(heights)).sum() / maximum

def holes(grid: np.ndarray) -> float:
    filled = grid != 0
    covered = np.maximum.accumulate(filled, axis=0)
    return -float(np.count_nonzero(covered & ~filled)) / grid.size

def tetris(grid: np.ndarray) -> float:
    return float(np.all(grid[:, -1] == 0))


def column_heights(grid: np.ndarray) -> np.ndarray:
    filled = grid != 0
    first = np.argmax(filled, axis=0)
    return np.where(filled.any(axis=0), grid.shape[0] - first, 0)
