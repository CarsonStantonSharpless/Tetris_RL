from __future__ import annotations

import numpy as np

from core.board import BoardState
from core.pieces import Orientation, PIECE_DEFS, PIECE_TO_ID, Piece


MAGIC = b"TRS1"
GRID_SHAPE = (22, 10)
GRID_CELLS = 220
GRID_BYTES = 83

KEYFRAME = 0
SAME_GRID = 1
GRID_DELTA = 2

ID_TO_PIECE = {piece_id: kind for kind, piece_id in PIECE_TO_ID.items()}


def write_board_states(states: list[BoardState], path: str) -> None:
    """Write board states to a compact binary file."""
    if len(states) >= 2 ** 32:
        raise ValueError("too many board states")

    with open(path, "wb") as file:
        file.write(MAGIC)
        file.write(len(states).to_bytes(4, "big"))

        previous_grid: np.ndarray | None = None

        for state in states:
            _validate_state(state)

            if previous_grid is None:
                record_type = KEYFRAME
                changes = np.array([], dtype=np.int64)
            else:
                changes = np.flatnonzero(state.grid.ravel() != previous_grid.ravel())

                if len(changes) == 0:
                    record_type = SAME_GRID
                elif 1 + len(changes) * 2 < GRID_BYTES:
                    record_type = GRID_DELTA
                else:
                    record_type = KEYFRAME

            file.write(_pack_piece(state, record_type))

            if record_type == KEYFRAME:
                file.write(_pack_grid(state.grid))
            elif record_type == GRID_DELTA:
                file.write(len(changes).to_bytes(1, "big"))

                flat_grid = state.grid.ravel()
                for index in changes:
                    change = (int(index) << 3) | int(flat_grid[index])
                    file.write(change.to_bytes(2, "big"))

            previous_grid = state.grid


def read_board_states(path: str) -> list[BoardState]:
    """Read board states from a compact binary file."""
    states: list[BoardState] = []

    with open(path, "rb") as file:
        if file.read(4) != MAGIC:
            raise ValueError("invalid board state file")

        state_count = int.from_bytes(_read_exact(file, 4), "big")
        previous_grid: np.ndarray | None = None

        for _ in range(state_count):
            record_type, piece, position = _unpack_piece(_read_exact(file, 2))

            if record_type == KEYFRAME:
                grid = _unpack_grid(_read_exact(file, GRID_BYTES))
            elif record_type == SAME_GRID:
                if previous_grid is None:
                    raise ValueError("same-grid record has no previous grid")
                grid = previous_grid.copy()
            elif record_type == GRID_DELTA:
                if previous_grid is None:
                    raise ValueError("grid-delta record has no previous grid")

                grid = previous_grid.copy()
                flat_grid = grid.ravel()
                change_count = int.from_bytes(_read_exact(file, 1), "big")

                for _ in range(change_count):
                    change = int.from_bytes(_read_exact(file, 2), "big")
                    index, value = change >> 3, change & 0b111

                    if index >= GRID_CELLS:
                        raise ValueError("invalid grid cell index")
                    flat_grid[index] = value
            else:
                raise ValueError("invalid record type")

            states.append(BoardState(grid, piece, position))
            previous_grid = grid

        if file.read(1):
            raise ValueError("unexpected data after board states")

    return states


def _pack_piece(state: BoardState, record_type: int) -> bytes:
    row, col = state.piece_pos
    row += 3
    col += 3

    if not 0 <= row < 2 ** 5 or not 0 <= col < 2 ** 4:
        raise ValueError("piece position cannot be stored")

    piece_id = PIECE_TO_ID[state.curr_piece.kind]
    value = (
        record_type << 14
        | piece_id << 11
        | int(state.curr_piece.orientation) << 9
        | row << 4
        | col
    )
    return value.to_bytes(2, "big")


def _unpack_piece(data: bytes) -> tuple[int, Piece, tuple[int, int]]:
    value = int.from_bytes(data, "big")

    record_type = value >> 14
    piece_id = value >> 11 & 0b111
    orientation = value >> 9 & 0b11
    row = (value >> 4 & 0b11111) - 3
    col = (value & 0b1111) - 3

    if piece_id not in ID_TO_PIECE:
        raise ValueError("invalid piece id")

    kind = ID_TO_PIECE[piece_id]
    piece = Piece(PIECE_DEFS[kind], Orientation(orientation))
    return record_type, piece, (row, col)


def _pack_grid(grid: np.ndarray) -> bytes:
    value = 0

    for cell in grid.ravel():
        value = value << 3 | int(cell)

    return (value << 4).to_bytes(GRID_BYTES, "big")


def _unpack_grid(data: bytes) -> np.ndarray:
    value = int.from_bytes(data, "big") >> 4
    grid = np.zeros(GRID_CELLS, dtype=np.int8)

    for index in range(GRID_CELLS - 1, -1, -1):
        grid[index] = value & 0b111
        value >>= 3

    return grid.reshape(GRID_SHAPE)


def _validate_state(state: BoardState) -> None:
    if state.grid.shape != GRID_SHAPE:
        raise ValueError(f"grid must have shape {GRID_SHAPE}")
    if np.any(state.grid < 0) or np.any(state.grid > 7):
        raise ValueError("grid cells must be between zero and seven")


def _read_exact(file, size: int) -> bytes:
    data = file.read(size)
    if len(data) != size:
        raise ValueError("incomplete board state file")
    return data