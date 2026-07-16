from __future__ import annotations

import time
import random
import curses
from dataclasses import dataclass

from core.board import Board, BoardState
from tui.renderer import TUI

# sets the target update rate for both rendering and gravity
TICK: float = 1.0 / 60.0
# follows the standard points for clearing one through four rows
LINE_SCORES = {1: 40, 2: 100, 3: 300, 4: 1200}
# limits the input queue to commands the game understands
VALID_KEYS = frozenset("adsjkqrp")


@dataclass(frozen=True)
class EngineState:
    board_state: BoardState
    next_piece: str
    score: int


class Engine:
    score: int
    board: Board
    ticks: int
    timer: int
    level: int
    lines_cleared: int
    piece_bag: list[str]
    next_tick_time: float
    renderer: TUI | None
    stdscr: curses.window | None

    def __init__(
        self,
        stdscr: curses.window | None = None,
        render: bool = True,
        tick_speed: float = TICK,
        auto_run: bool | None = None,
    ) -> None:
        """Create a Tetris engine.

        Args:
            stdscr: The curses screen used when rendering is enabled.
            render: Enable the terminal UI. Set to ``False`` for RL training.
            tick_speed: Seconds per engine tick.
            auto_run: Start the interactive loop immediately. By default this
                follows ``render``; headless engines are ready for ``step()``.
        """
        if tick_speed <= 0:
            raise ValueError("tick_speed must be greater than zero")
        if render and stdscr is None:
            raise ValueError("stdscr is required when render=True")

        self.stdscr = stdscr
        self.render_enabled = render
        self.tick_speed = tick_speed
        self.renderer = None

        if render:
            assert stdscr is not None
            curses.curs_set(0)
            stdscr.nodelay(True)
            stdscr.keypad(True)
            self.renderer = TUI(stdscr)

        if auto_run is None:
            auto_run = render

        self.restart()
        if auto_run:
            self.run()

    @property
    def time(self) -> float:
        return self.ticks * self.tick_speed

    @property
    def state(self) -> EngineState:
        return EngineState(
            board_state=self.board.board,
            next_piece=self.next_piece(),
            score=self.score,
        )

    def restart(self) -> None:
        self.score = 0
        self.board = Board()
        self.ticks = 0
        self.level = 0
        self.timer = 0
        self.lines_cleared = 0
        self.next_tick_time = time.perf_counter() + self.tick_speed
        self.is_game_over = False

        # shuffling a full bag prevents long runs without a particular piece
        self.piece_bag = []
        self.fill_bag()

        self.start_timer()
        self.board.spawn_piece(self.grab_piece())

    def add_score(self, lines: int) -> None:
        self.score += LINE_SCORES.get(lines, 0) * (self.level + 1)

    def get_input(self) -> list[str]:
        if self.stdscr is None:
            return []

        inputs: list[str] = []

        # drains every queued keystroke so quick inputs are not lost between frames
        while (key := self.stdscr.getch()) != -1:
            command = chr(key)
            if command in VALID_KEYS:
                inputs.append(command)

        return inputs

    def wait(self) -> None:
        now = time.perf_counter()

        if not hasattr(self, "next_tick_time"):
            self.next_tick_time = now + self.tick_speed

        sleep_time = self.next_tick_time - now

        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            # drops missed frames instead of trying to replay a long delay
            self.next_tick_time = now

        self.next_tick_time += self.tick_speed

    def render(self) -> None:
        if self.renderer is None:
            return
        self.renderer.render(
            self.board.board,
            self.next_piece(),
            self.score,
            self.level,
            self.time,
        )

    def game_over(self) -> None:
        self.is_game_over = True
        if self.renderer is None or self.stdscr is None:
            return

        self.renderer.game_over(
            self.board.board,
            self.next_piece(),
            self.score,
            self.level,
            self.time,
        )

        while True:
            key = self.stdscr.getch()

            if key == ord("q"):
                raise SystemExit

            if key == ord("r"):
                self.restart()
                return

            time.sleep(self.tick_speed)

    def fill_bag(self) -> None:
        self.piece_bag = ["O", "I", "T", "L", "J", "S", "Z"]
        random.shuffle(self.piece_bag)

    def next_piece(self) -> str:
        if not self.piece_bag:
            self.fill_bag()
        return self.piece_bag[-1]

    def pause(self) -> None:
        if self.renderer is None or self.stdscr is None:
            return

        self.renderer.pause(
            self.board.board,
            self.next_piece(),
            self.score,
            self.level,
            self.time,
        )

        while True:
            key = self.stdscr.getch()

            if key == ord("q"):
                raise SystemExit

            if key == ord("r"):
                self.restart()
                return

            if key == ord("p"):
                # restarts the frame schedule so resuming feels immediate
                self.next_tick_time = time.perf_counter() + self.tick_speed
                return

            time.sleep(self.tick_speed)

    def grab_piece(self) -> str:
        if not self.piece_bag:
            self.fill_bag()

        piece = self.piece_bag.pop()

        if not self.piece_bag:
            self.fill_bag()

        return piece

    def handle_level(self, lines: int) -> None:
        # increases a level each time the running total crosses ten lines
        if self.lines_cleared // 10 < (self.lines_cleared + lines) // 10:
            if self.level <= 20:
                self.level += 1
        self.lines_cleared += lines

    def handle_input(self, inputs: list[str]) -> bool:
        soft_dropped = False

        for cmd in inputs:
            if cmd == "r":
                self.restart()
                return False
            elif cmd == "p":
                self.pause()
                return False

            elif cmd == "q":
                raise SystemExit

            elif cmd == "s":
                # moves down twice to make a soft drop feel responsive
                if self.board.lower_piece():
                    self.score += 1
                    soft_dropped = True

                    if self.board.lower_piece():
                        self.score += 1

            elif cmd == "a":
                self.board.move_piece_left()

            elif cmd == "d":
                self.board.move_piece_right()

            elif cmd == "j":
                self.board.rotate_piece_left()

            elif cmd == "k":
                self.board.rotate_piece_right()

        return soft_dropped

    def start_timer(self) -> None:
        self.timer += max(3, 48 - self.level * 7)

    def step(self, inputs: list[str] | str = ()) -> EngineState:
        if self.is_game_over:
            return self.state

        self.ticks += 1

        commands = list(inputs)
        invalid = set(commands) - VALID_KEYS
        if invalid:
            raise ValueError(f"invalid command(s): {sorted(invalid)!r}")

        soft_dropped = False
        if commands:
            soft_dropped = self.handle_input(commands)

        if self.timer <= 0:
            if not soft_dropped:
                if not self.board.lower_piece():
                    self.board.lock_board()

                    lines = self.board.clear_rows()
                    if lines:
                        self.handle_level(lines)
                        self.add_score(lines)

                    self.board.spawn_piece(self.grab_piece())
                    if self.board.game_over():
                        self.game_over()

            self.start_timer()
        else:
            self.timer -= 1

        self.render()
        return self.state

    def run(self) -> None:
        while True:
            self.step(self.get_input())
            self.wait()
