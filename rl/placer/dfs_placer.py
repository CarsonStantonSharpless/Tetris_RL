from __future__ import annotations

from core.board import Board, BoardState
from core.pieces import Piece


def DFS_placer(
    board: BoardState,
    dest: tuple[int, int],
    rotation: int = 0,
) -> list[str]:
    piece = Piece(board.curr_piece.definition, board.curr_piece.orientation)
    path: list[str] = []

    command = "k" if rotation >= 0 else "j"
    for _ in range(abs(rotation)):
        if rotation >= 0:
            piece.rotate_right()
        else:
            piece.rotate_left()

        if not Board.is_legal_position(BoardState(board.grid, piece, board.piece_pos)):
            raise ValueError("rotation is not legal")
        path.append(command)

    stack: list[tuple[tuple[int, int], list[str]]] = [(board.piece_pos, path)]
    visited = {board.piece_pos}

    while stack:
        pos, path = stack.pop()

        if pos == dest:
            return path

        for command, next_pos in (
            ("a", (pos[0], pos[1] - 1)),
            ("d", (pos[0], pos[1] + 1)),
            ("s", (pos[0] + 1, pos[1])),
        ):
            if next_pos in visited:
                continue
            if not Board.is_legal_position(BoardState(board.grid, piece, next_pos)):
                continue

            visited.add(next_pos)
            stack.append((next_pos, path + [command]))

    raise ValueError(f"destination {dest} is not reachable")
