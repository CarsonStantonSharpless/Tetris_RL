from collections import deque
import curses
from enum import IntEnum

from core.board import BoardState, PlacementOutcome
from core.engine import Engine, EngineState, PlacementStep, TICK
from agents.placements import simulate_hard_drop
from agents.placers.bfs import bfs_placer
from agents.placers.dfs import dfs_placer
from agents.policies.heuristic import genetic_heuristic_policy, heuristic_policy
from agents.policies.random import random_policy
from agents.policies.dqn import dqn_policy
from agents.policies.ppo import ppo_policy
from storage.games import write_board_states


class Policy(IntEnum):
    RANDOM = 1
    HEURISTIC = 2
    GENETIC_HEURISTIC = 3
    DQN = 4
    PPO = 5


POLICIES = {
    Policy.RANDOM: random_policy,
    Policy.HEURISTIC: heuristic_policy,
    Policy.GENETIC_HEURISTIC: genetic_heuristic_policy,
    Policy.DQN: dqn_policy,
    Policy.PPO: ppo_policy,
}


class Placer(IntEnum):
    DFS = 1
    BFS = 2


PLACERS = {
    Placer.DFS: dfs_placer,
    Placer.BFS: bfs_placer,
}


class Player:
    def __init__(
        self,
        policy: Policy = Policy.RANDOM,
        placer: Placer = Placer.BFS,
        display: bool = True,
        interactive: bool = False,
        stdscr: curses.window | None = None,
        filepath: str | None = None,
        tick_speed: float = TICK,
        policy_params: dict[str, float | int | str | None] | None = None,
        instant_placement: bool = False,
        seed: int | None = None,
    ) -> None:
        if policy_params and policy not in (
            Policy.HEURISTIC,
            Policy.GENETIC_HEURISTIC,
            Policy.DQN,
            Policy.PPO,
        ):
            raise ValueError("policy_params require a parameterized policy")

        self.policy = POLICIES[policy]
        self.policy_params = dict(policy_params or {})
        self.placer_kind = placer
        self.placer = PLACERS[placer]
        self.interactive = interactive
        self.filepath = filepath
        self.instant_placement = instant_placement

        self.engine = Engine(
            stdscr=stdscr,
            render=display,
            tick_speed=tick_speed,
            auto_run=False,
            instant_placement=instant_placement,
            seed=seed,
        )

        self.path: deque[str | None] = deque()
        self.target: BoardState | None = None
        self.active_piece = None
        self.states = [self.engine.state.board_state] if filepath else []

    @property
    def state(self) -> EngineState:
        return self.engine.state

    def possible_placements(self) -> list[BoardState]:
        return self.engine.possible_placements()

    def simulate_placement(self, placement: BoardState) -> PlacementOutcome:
        return self.engine.simulate_placement(placement)

    def place(self, placement: BoardState) -> PlacementStep:
        return self.engine.place(placement)

    def restart(self, seed: int | None = None) -> EngineState:
        self.engine.restart(seed=seed)
        return self.engine.state

    def _plan(self, state: BoardState) -> None:
        self.target = self.policy(
            state,
            next_piece=self.engine.next_piece(),
            level=self.engine.level,
            **self.policy_params,
        )
        if self.instant_placement:
            self.path.clear()
            return

        try:
            self.path = deque(self._place(state))
        except ValueError:
            self.target = simulate_hard_drop(
                state.grid,
                state.curr_piece,
                state.piece_pos,
            )
            self.path = deque(self._place(state))

    def _place(self, state: BoardState) -> list[str | None]:
        assert self.target is not None
        if self.placer_kind is Placer.BFS:
            return self.placer(
                state,
                self.target,
                self.engine.timer,
                self.engine.gravity_interval,
            )

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

        if self.instant_placement:
            assert self.target is not None
            state = self.engine.step(placement=self.target)
            if self.filepath:
                self.states.append(state.board_state)
            return state

        manual = self.engine.get_input()
        if not self.interactive:
            manual = [command for command in manual if command in "qrp"]

        commands = manual
        if self.path:
            command = self.path.popleft()
            if command is not None:
                commands.append(command)

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
