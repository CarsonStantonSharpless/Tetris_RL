from __future__ import annotations

import time

from core.board import BoardState
from tui.renderer import TUI


def render_board_state(tui: TUI, state: BoardState) -> None:
    """Render a board state with the active piece shown as next."""
    tui.render(state, state.curr_piece.kind, 0, 0, 0)


def cycle_board_states(
        tui: TUI,
        states: list[BoardState],
        tick_speed: float,
) -> None:
    """Loop through board states, rendering each at the given tick speed."""
    if tick_speed <= 0:
        raise ValueError("tick_speed must be greater than zero")
    if not states:
        raise ValueError("states must not be empty")

    while True:
        for state in states:
            render_board_state(tui, state)
            time.sleep(tick_speed)
