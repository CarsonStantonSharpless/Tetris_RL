from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from core.board import Board, BoardState
from core.pieces import Orientation, Piece


@dataclass(frozen=True)
class SearchState:
    row: int
    col: int
    orientation: Orientation
    timer: int


def bfs_placer(
    board: BoardState,
    target: BoardState,
    timer: int,
    gravity_interval: int,
) -> list[str | None]:
    """Find a tick-accurate path to a policy-selected placement."""
    if target.curr_piece.kind != board.curr_piece.kind:
        raise ValueError("target must use the active piece")

    start = SearchState(
        row=board.piece_pos[0],
        col=board.piece_pos[1],
        orientation=board.curr_piece.orientation,
        timer=timer,
    )
    destination = (
        target.piece_pos[0],
        target.piece_pos[1],
        target.curr_piece.orientation,
    )

    queue = deque([(start, [])])
    visited = {start}

    while queue:
        state, path = queue.popleft()
        if _position(state) == destination:
            return path

        # Trying movement before descent makes equally short paths position the
        # piece near the spawn row instead of dragging it down prematurely.
        for command in ("a", "d", "j", "k", "s", None):
            next_state, locked = _transition(
                board,
                state,
                command,
                gravity_interval,
            )
            next_path = path + [command]

            if locked:
                if _position(next_state) == destination:
                    return next_path
                continue
            if next_state in visited:
                continue

            visited.add(next_state)
            queue.append((next_state, next_path))

    raise ValueError(f"destination {target.piece_pos} is not reachable")


def _transition(
    board: BoardState,
    state: SearchState,
    command: str | None,
    gravity_interval: int,
) -> tuple[SearchState, bool]:
    row = state.row
    col = state.col
    orientation = state.orientation
    soft_dropped = False

    if command == "a":
        if _is_legal(board, row, col - 1, orientation):
            col -= 1
    elif command == "d":
        if _is_legal(board, row, col + 1, orientation):
            col += 1
    elif command == "j":
        proposed = _rotated_orientation(board, orientation, left=True)
        if _is_legal(board, row, col, proposed):
            orientation = proposed
    elif command == "k":
        proposed = _rotated_orientation(board, orientation, left=False)
        if _is_legal(board, row, col, proposed):
            orientation = proposed
    elif command == "s":
        if _is_legal(board, row + 1, col, orientation):
            row += 1
            soft_dropped = True
            if _is_legal(board, row + 1, col, orientation):
                row += 1

    timer = state.timer
    locked = False
    if timer <= 0:
        if not soft_dropped:
            if _is_legal(board, row + 1, col, orientation):
                row += 1
            else:
                locked = True
        timer += gravity_interval
    else:
        timer -= 1

    return SearchState(row, col, orientation, timer), locked


def _is_legal(
    board: BoardState,
    row: int,
    col: int,
    orientation: Orientation,
) -> bool:
    piece = Piece(board.curr_piece.definition, orientation)
    return Board.is_legal_position(BoardState(board.grid, piece, (row, col)))


def _rotated_orientation(
    board: BoardState,
    orientation: Orientation,
    *,
    left: bool,
) -> Orientation:
    piece = Piece(board.curr_piece.definition, orientation)
    if left:
        piece.rotate_left()
    else:
        piece.rotate_right()
    return piece.orientation


def _position(state: SearchState) -> tuple[int, int, Orientation]:
    return state.row, state.col, state.orientation
