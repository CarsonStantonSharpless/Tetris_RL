from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from core.pieces import Piece
from core.board import Board, BoardState


def find_possible_states(
        state: BoardState,
        harddrop: bool = True
) -> list[BoardState]:
    """
    Inputs:
        state: grid, curremt_piece, and position of current piece
        hotdrop: if true asume that these are positions only reached
            by moving left and right on the x axis, then straight down
    """

    if harddrop is True: return find_hard_drop_placements(state)
    else: return find_soft_drop_placements(state)

def find_hard_drop_placements(state: BoardState) -> list[BoardState]:
    """
    Brute Forcey but there aren't too many things to consider
    """
    grid, piece, curr_pos = state.grid, state.curr_piece, state.piece_pos
    
    poss_placements: list[BoardState] = []
    seen_shapes = set()

    for rotation in range(4):
        rotated_piece = Piece(piece.definition, piece.orientation)
        for _ in range(rotation):
            rotated_piece.rotate_right()

        shape = rotated_piece.shape
        rows, cols = np.nonzero(shape)
        trimmed_shape = shape[
            rows.min():rows.max() + 1,
            cols.min():cols.max() + 1,
        ]
        shape_key = (trimmed_shape.shape, trimmed_shape.tobytes())

        if shape_key in seen_shapes:
            continue
        seen_shapes.add(shape_key)

        for col in range(-int(cols.min()), grid.shape[1] - int(cols.max())):
            if (Board.is_legal_position(
                BoardState(grid, rotated_piece, (curr_pos[0],col))) is False):
                    continue

            poss_placements.append(
                simulate_hard_drop(grid, rotated_piece, (curr_pos[0],col))
            )
    
    return poss_placements



def find_soft_drop_placements(state: BoardState) -> list[BoardState]:
    raise NotImplementedError("Have not implemented soft drops yet")


def simulate_hard_drop(
        grid: np.ndarray,
        piece: Piece,
        pos: tuple[int,int]
) -> BoardState:
    """Return the resting state without simulating intermediate rows."""
    start_row, start_col = pos
    landing_row = grid.shape[0]
    occupied_rows, occupied_cols = np.nonzero(piece.shape)

    for local_row, local_col in zip(
        occupied_rows,
        occupied_cols,
        strict=True,
    ):
        grid_col = start_col + int(local_col)
        first_row_below = start_row + int(local_row) + 1
        filled_below = np.flatnonzero(
            grid[first_row_below:, grid_col]
        )

        obstacle_row = (
            first_row_below + int(filled_below[0])
            if filled_below.size
            else grid.shape[0]
        )
        landing_row = min(
            landing_row,
            obstacle_row - int(local_row) - 1,
        )

    return BoardState(grid, piece, (landing_row, start_col))
