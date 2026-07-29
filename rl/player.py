from collections import deque
import curses
from enum import IntEnum

from core.board import BoardState
from core.engine import Engine, EngineState, TICK
from core.store import write_board_states
from rl.placer.dfs_placer import DFS_placer
from rl.policy.possible_moves import simulate_hard_drop
from rl.policy.random_policy import random_policy
from rl.policy.hueristic_policy import hueristic_policy


class Policy(IntEnum):
    RANDOM = 1
    HUERISTIC = 2


POLICIES = {
    Policy.RANDOM: random_policy,
    Policy.HUERISTIC: hueristic_policy,
}


class Placer(IntEnum):
    DFS = 1


PLACERS = {
    Placer.DFS: DFS_placer,
}


class Player:
    def __init__(
        self,
        policy: Policy = Policy.RANDOM,
        placer: Placer = Placer.DFS,
        display: bool = True,
        interactive: bool = False,
        stdscr: curses.window | None = None,
        filepath: str | None = None,
        tick_speed: float = TICK,
    ) -> None:
        self.policy = POLICIES[policy]
        self.placer = PLACERS[placer]
        self.interactive = interactive
        self.filepath = filepath

        self.engine = Engine(
            stdscr=stdscr,
            render=display,
            tick_speed=tick_speed,
            auto_run=False,
        )

        self.path: deque[str] = deque()
        self.target: BoardState | None = None
        self.active_piece = None
        self.states = [self.engine.state.board_state] if filepath else []

    def _plan(self, state: BoardState) -> None:
        self.target = self.policy(state)

        try:
            self.path = deque(self._place(state))
        except ValueError:
            self.target = simulate_hard_drop(
                state.grid,
                state.curr_piece,
                state.piece_pos,
            )
            self.path = deque(self._place(state))

    def _place(self, state: BoardState) -> list[str]:
        assert self.target is not None
        rotation = (
            int(state.curr_piece.orientation)
            - int(self.target.curr_piece.orientation)
        ) % 4
        return self.placer(state, self.target.piece_pos, rotation)

    def step(self) -> EngineState:
        board_state = self.engine.state.board_state
        active_piece = self.engine.board.curr_piece

        if active_piece is not self.active_piece:
            self.active_piece = active_piece
            self._plan(board_state)

        manual = self.engine.get_input()
        if not self.interactive:
            manual = [command for command in manual if command in "qrp"]

        commands = manual
        if self.path:
            commands.append(self.path.popleft())

        state = self.engine.step(commands)
        if self.filepath:
            self.states.append(state.board_state)
        if self.engine.render_enabled:
            self.engine.wait()
        return state

    def save(self) -> None:
        if self.filepath:
            write_board_states(self.states, self.filepath)

    def start(self, max_ticks: int | None = None) -> EngineState:
        steps = 0
        try:
            while not self.engine.is_game_over and (
                max_ticks is None or steps < max_ticks
            ):
                state = self.step()
                steps += 1
            return state if steps else self.engine.state
        finally:
            self.save()
