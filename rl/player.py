from __future__ import annotations
from dataclasses import dataclass

from enum import IntEnum
from collections import deque
import numpy as np

from core.pieces import Piece
from core.board import Board, BoardState
from core.engine import Engine
from core.store import *

from rl.placer import *
from rl.policy import *

class Policy(IntEnum):
    RANDOM = 1

class Placer(IntEnum):
    BFS = 1

class Player():
    policy: Policy
    placer: Placer

    engine: Engine

    display: bool

    store: bool
    filepath: str | None

    path: deque[list[str]] | None
    target: BoardState | None


