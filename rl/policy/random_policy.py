from core.board import BoardState
from possible_moves import find_possible_states
import random

def random_policy(state: BoardState, harddrop: bool = True) -> BoardState:
    states = find_possible_states(state, harddrop)
    n: int = len(states)
    return states[random.randint(0,n-1)]