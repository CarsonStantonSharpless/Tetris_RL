import random

from agents.placements import find_possible_states
from core.board import BoardState


def random_policy(
    state: BoardState,
    harddrop: bool = True,
    next_piece: str | None = None,
    level: int = 0,
) -> BoardState:
    del next_piece, level
    return random.choice(find_possible_states(state, harddrop))
