from enum import IntEnum
from collections import deque
import numpy as np
import curses

from core.pieces import Piece
from core.board import Board, BoardState
from core.engine import Engine
from core.store import *

from rl.placer.dfs_placer import DFS_placer
from rl.policy.random_policy import random_policy

class Policy(IntEnum):
    RANDOM = 1

POLICIES = {
    1 : random_policy
}

class Placer(IntEnum):
    DFS = 1

PLACERS = {
    1 : DFS_placer
}

TICK: float = 1.0 / 60.0

class Player():
    policy: "function"
    placer: Placer

    engine: Engine

    display: bool
    interactive: bool
    stdscr: curses.window | None

    filepath: str | None

    path: deque[list[str]] | None
    target: BoardState | None

    tick_speed: float

    def __init__(
        self,
        policy: Policy,
        placer: Placer,
        display: bool = True,
        interactive: bool = False,
        stdscr: curses.window | None = None,
        filepath: str | None = None,
        tick_speed: float = TICK
    ) -> None:
        
        self.policy = POLICIES[policy]
        self.placer = PLACERS[placer]

        self.engine = Engine(
            window=stdscr,
            render=display,
            tick_speed=tick_speed,
            interactive=interactive,
            auto_run=False
        )

        self.filepath = filepath

        self.start()

    def start(self):
        self.engine.run()

#Yeah I need to do threading if I want ts to work how I want but I think its worth