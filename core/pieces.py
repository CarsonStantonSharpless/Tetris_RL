from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class Orientation(IntEnum):
    SOUTH = 0
    WEST = 1
    NORTH = 2
    EAST = 3


@dataclass(frozen=True)
class PieceDef:
    kind: str
    shape: np.ndarray
    color_pair: int


@dataclass
class Piece:
    definition: PieceDef
    orientation: Orientation = Orientation.SOUTH

    @property
    def kind(self) -> str:
        return self.definition.kind

    @property
    def shape(self) -> np.ndarray:
        return np.rot90(self.definition.shape, k=self.orientation)

    def rotate_left(self) -> None:
        if self.kind == "O":
            return
        self.orientation = Orientation((self.orientation + 1) % 4)

    def rotate_right(self) -> None:
        if self.kind == "O":
            return
        self.orientation = Orientation((self.orientation - 1) % 4)

# maps piece names to the curses color pairs used throughout the game
PIECE_TO_ID = {
    "I": 1, "O": 2, "T": 3, "J": 4, "L": 5, "S": 6, "Z": 7,
}

# stores each piece in its spawn orientation
PIECE_DEFS = {
    "O": PieceDef(
        kind="O",
        shape=np.array([
            [0, 1, 1],
            [0, 1, 1],
            [0, 0, 0],
        ], dtype=np.int8),
        color_pair=2,
    ),

    "I": PieceDef(
        kind="I",
        shape=np.array([
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int8),
        color_pair=1,
    ),

    "T": PieceDef(
        kind="T",
        shape=np.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 0, 0],
        ], dtype=np.int8),
        color_pair=3,
    ),

    "L": PieceDef(
        kind="L",
        shape=np.array([
            [0, 0, 1],
            [1, 1, 1],
            [0, 0, 0],
        ], dtype=np.int8),
        color_pair=5,
    ),

    "J": PieceDef(
        kind="J",
        shape=np.array([
            [1, 0, 0],
            [1, 1, 1],
            [0, 0, 0],
        ], dtype=np.int8),
        color_pair=4,
    ),

    "S": PieceDef(
        kind="S",
        shape=np.array([
            [0, 1, 1],
            [1, 1, 0],
            [0, 0, 0],
        ], dtype=np.int8),
        color_pair=6,
    ),

    "Z": PieceDef(
        kind="Z",
        shape=np.array([
            [1, 1, 0],
            [0, 1, 1],
            [0, 0, 0],
        ], dtype=np.int8),
        color_pair=7,
    ),
}
