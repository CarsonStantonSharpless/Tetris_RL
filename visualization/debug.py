from __future__ import annotations

import curses

from agents.placements import find_possible_states
from core.board import Board, BoardState
from tui.renderer import TUI
from visualization.boards import cycle_board_states


def visualize_hard_drops(
        state: BoardState,
        renderer: TUI,
) -> None:
    
    poss_states: list[BoardState] =  find_possible_states(
        state, True
    )

    cycle_board_states(
        renderer, poss_states, .25
    )



def main(stdscr):
    renderer = TUI(stdscr)

    board = Board()
    board.spawn_piece("I")

    visualize_hard_drops(board.board, renderer)

if __name__ == "__main__":
    curses.wrapper(main)
