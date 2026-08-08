from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from core.pieces import PIECE_DEFS, PIECE_TO_ID, Piece


@dataclass
class BoardState:
    # holds a snapshot for rendering so the renderer cannot alter the game board
    grid: np.ndarray
    curr_piece: Piece
    piece_pos: tuple[int, int]

    @property
    def locked(self) -> np.ndarray:
        """Return a copy of the grid with the active piece drawn into it."""
        combined = self.grid.copy()

        row, col = self.piece_pos
        piece_id = PIECE_TO_ID[self.curr_piece.kind]

        for local_row, shape_row in enumerate(self.curr_piece.shape):
            for local_col, value in enumerate(shape_row):
                if value == 0:
                    continue

                grid_row = row + local_row
                grid_col = col + local_col

                if (
                    0 <= grid_row < combined.shape[0]
                    and 0 <= grid_col < combined.shape[1]
                ):
                    combined[grid_row, grid_col] = piece_id

        return combined


@dataclass(frozen=True)
class PlacementOutcome:
    locked_grid: np.ndarray
    grid: np.ndarray
    lines_cleared: int
    game_over: bool

class Board:
    def __init__(self) -> None:
        # stores locked pieces; zero is empty and each other value is a piece id
        self.grid: np.ndarray = np.zeros((22, 10), dtype=np.int8)
        self.curr_piece: Piece | None = None
        self.piece_pos: tuple[int, int] | None = None

    @property
    def board(self) -> BoardState:
        piece, pos = self.require_active_piece()
        return BoardState(
            self.grid.copy(),
            Piece(piece.definition, piece.orientation),
            pos,
        )

    def require_active_piece(self) -> tuple[Piece, tuple[int, int]]:
        assert self.curr_piece is not None and self.piece_pos is not None
        return self.curr_piece, self.piece_pos

    def spawn_piece(self, piece_name: str) -> None:
        piece: Piece = Piece(PIECE_DEFS[piece_name])
        # starts each piece in the hidden rows, centered over the playfield
        self.piece_pos = (0, 3)
        self.curr_piece = piece

    def move_piece_left(self) -> bool:
        piece, pos = self.require_active_piece()
        proposed: tuple[int, int] = (pos[0], pos[1] - 1)

        if not self.is_legal_position(
                BoardState(
                    self.grid,
                    piece,
                    proposed
                )
            ): return False

        self.piece_pos = proposed
        return True

    def move_piece_right(self) -> bool:
        piece, pos = self.require_active_piece()
        proposed: tuple[int, int] = (pos[0], pos[1] + 1)

        if not self.is_legal_position(
                BoardState(
                    self.grid,
                    piece,
                    proposed
                )
            ): return False

        self.piece_pos = proposed
        return True

    def lower_piece(self) -> bool:
        piece, pos = self.require_active_piece()
        proposed: tuple[int, int] = (pos[0] + 1, pos[1])

        if self.is_legal_position(
                BoardState(
                    self.grid,
                    piece,
                    proposed
                )
            ):
            self.piece_pos = proposed
            return True

        return False

    def rotate_piece_right(self) -> bool:
        piece, pos = self.require_active_piece()

        piece.rotate_right()

        if not self.is_legal_position(
                BoardState(
                    self.grid,
                    piece,
                    pos
                )
            ):
            piece.rotate_left()
            return False

        return True

    def rotate_piece_left(self) -> bool:
        piece, pos = self.require_active_piece()

        piece.rotate_left()

        if not self.is_legal_position(
                BoardState(
                    self.grid,
                    piece,
                    pos
                )
            ):
            piece.rotate_right()
            return False

        return True
    
    @staticmethod
    def is_legal_position(board: BoardState) -> bool:
        grid, piece, pos = board.grid, board.curr_piece, board.piece_pos
        row, col = pos
        shape = piece.shape
        occupied_rows, occupied_cols = np.nonzero(shape)

        top = int(occupied_rows.min())
        bottom = int(occupied_rows.max()) + 1
        left = int(occupied_cols.min())
        right = int(occupied_cols.max()) + 1

        grid_top = row + top
        grid_bottom = row + bottom
        grid_left = col + left
        grid_right = col + right

        if (
            grid_top < 0
            or grid_bottom > grid.shape[0]
            or grid_left < 0
            or grid_right > grid.shape[1]
        ):
            return False

        shape_window = shape[top:bottom, left:right] != 0
        grid_window = grid[
            grid_top:grid_bottom,
            grid_left:grid_right,
        ]
        return not np.any(grid_window[shape_window] != 0)

    def lock_board(self) -> None:
        piece, pos = self.require_active_piece()
        row, col = pos

        piece_id = PIECE_TO_ID[piece.kind]

        # copies the active piece into the permanent board grid
        for local_row, shape_row in enumerate(piece.shape):
            for local_col, value in enumerate(shape_row):
                if value == 0:
                    continue

                grid_row = row + local_row
                grid_col = col + local_col

                if 0 <= grid_row < self.grid.shape[0] and 0 <= grid_col < self.grid.shape[1]:
                    self.grid[grid_row, grid_col] = piece_id

        self.curr_piece = None
        self.piece_pos = None

    def full_rows(self) -> np.ndarray:
        return np.all(self.grid != 0, axis=1)

    def clear_rows(self) -> int:
        self.grid, n_cleared = self._clear_grid(self.grid)
        return n_cleared

    @staticmethod
    def simulate_placement(state: BoardState) -> PlacementOutcome:
        """Lock and clear a candidate placement without changing the board."""
        if not Board.is_legal_position(state):
            raise ValueError("cannot simulate an illegal placement")

        locked_grid = state.locked
        grid, lines_cleared = Board._clear_grid(locked_grid)
        return PlacementOutcome(
            locked_grid=locked_grid,
            grid=grid,
            lines_cleared=lines_cleared,
            game_over=bool(np.any(grid[0:2, :] != 0)),
        )

    @staticmethod
    def _clear_grid(grid: np.ndarray) -> tuple[np.ndarray, int]:
        full = np.all(grid != 0, axis=1)
        n_cleared = int(np.count_nonzero(full))
        if n_cleared == 0:
            return grid, 0

        remaining = grid[~full]
        empty_rows = np.zeros(
            (n_cleared, grid.shape[1]),
            dtype=grid.dtype,
        )
        return np.vstack((empty_rows, remaining)), n_cleared

    def game_over(self) -> bool:
        return np.any(self.grid[0:2, :] != 0)
