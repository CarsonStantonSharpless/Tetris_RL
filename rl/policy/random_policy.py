import random

from core.board import BoardState
from rl.policy.possible_moves import find_possible_states


def random_policy(state: BoardState, harddrop: bool = True) -> BoardState:
    return random.choice(find_possible_states(state, harddrop))
