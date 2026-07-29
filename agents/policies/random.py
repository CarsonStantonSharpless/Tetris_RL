import random

from agents.placements import find_possible_states
from core.board import BoardState


def random_policy(state: BoardState, harddrop: bool = True) -> BoardState:
    return random.choice(find_possible_states(state, harddrop))
