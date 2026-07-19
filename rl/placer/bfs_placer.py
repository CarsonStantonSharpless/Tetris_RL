from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from core.pieces import Piece
from core.board import Board, BoardState

"""
Given a destination and rotation, move the piece to that location
"""

def BFS_place(board: BoardState, dest: tuple[int, int]) -> list[str]:
    