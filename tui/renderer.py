from __future__ import annotations

import curses
import sys
import math
from dataclasses import dataclass

import numpy as np

from core.board import BoardState
from core.pieces import PIECE_DEFS, PIECE_TO_ID


@dataclass
class TUI:
    stdscr: curses.window

    # keeps the playfield clear of the controls panel
    top: int = 0
    left: int = 23

    visible_height: int = 20
    visible_width: int = 10
    hidden_rows: int = 2

    sidebar_gap: int = 3
    sidebar_width: int = 14
    left_sidebar_width: int = 18

    auto_resize_terminal: bool = True
    _has_resized_terminal: bool = False

    def __post_init__(self) -> None:
        self.init_colors()
        self._fit_terminal_to_game()

    def init_colors(self) -> None:
        curses.start_color()
        curses.use_default_colors()

        # keeps the color-pair ids aligned with the piece ids in core.pieces
        curses.init_pair(1, curses.COLOR_CYAN, -1)
        curses.init_pair(2, curses.COLOR_YELLOW, -1)
        curses.init_pair(3, curses.COLOR_MAGENTA, -1)
        curses.init_pair(4, curses.COLOR_BLUE, -1)
        curses.init_pair(5, curses.COLOR_WHITE, -1)
        curses.init_pair(6, curses.COLOR_GREEN, -1)
        curses.init_pair(7, curses.COLOR_RED, -1)

        curses.init_pair(8, curses.COLOR_BLUE, -1)
        curses.init_pair(9, curses.COLOR_WHITE, -1)

    def _desired_terminal_size(self) -> tuple[int, int]:
        left_sidebar_total_width = self.left_sidebar_width + 2
        grid_total_width = self.visible_width * 2 + 2
        right_sidebar_total_width = self.sidebar_width + 2

        cols = (
            left_sidebar_total_width
            + self.sidebar_gap
            + grid_total_width
            + self.sidebar_gap
            + right_sidebar_total_width
        )

        # leaves room for all board rows, its border, and curses' bottom edge
        rows = self.hidden_rows + self.visible_height + 3

        return rows, cols

    def _fit_terminal_to_game(self) -> None:
        if not self.auto_resize_terminal or self._has_resized_terminal:
            return

        rows, cols = self._desired_terminal_size()

        # asks compatible terminals to make room for the complete layout
        sys.stdout.write(f"\x1b[8;{rows};{cols}t")
        sys.stdout.flush()

        try:
            curses.resize_term(rows, cols)
        except curses.error:
            pass

        self.stdscr.clear()
        self._has_resized_terminal = True

    def __call__(
        self,
        board: BoardState,
        next_piece: str,
        score: int,
        level: int,
        game_time: float,
    ) -> None:
        self.render(board, next_piece, score, level, game_time)

    def render(
        self,
        board: BoardState,
        next_piece: str,
        score: int,
        level: int,
        game_time: float,
    ) -> None:
        self._fit_terminal_to_game()
        self.stdscr.erase()

        render_grid = self._build_render_grid(board)

        self._draw_left_sidebar(game_time)
        self._draw_grid(render_grid)
        self._draw_right_sidebar(next_piece, score, level, game_time)

        self.stdscr.refresh()

    def _format_time(self, game_time: float) -> str:
        total_centiseconds = int(game_time * 100)

        minutes = total_centiseconds // 6000
        seconds = (total_centiseconds % 6000) // 100
        centiseconds = total_centiseconds % 100

        return f"{minutes:02}:{seconds:02}:{centiseconds:02}"

    def _build_render_grid(self, board: BoardState) -> np.ndarray:
        render_grid = board.grid.copy()

        piece = board.curr_piece
        pos = board.piece_pos

        if piece is None or pos is None:
            return render_grid

        piece_y, piece_x = pos

        piece_id = PIECE_TO_ID[piece.kind]
        shape = piece.shape

        # layers the active piece over a copy of the locked board cells
        for local_y, row in enumerate(shape):
            for local_x, value in enumerate(row):
                if value == 0:
                    continue

                grid_y = piece_y + local_y
                grid_x = piece_x + local_x

                if 0 <= grid_y < render_grid.shape[0] and 0 <= grid_x < render_grid.shape[1]:
                    render_grid[grid_y, grid_x] = piece_id

        return render_grid

    def game_over(
        self,
        board: BoardState,
        next_piece: str,
        score: int,
        level: int,
        game_time: float,
    ) -> None:
        self.render(board, next_piece, score, level, game_time)

        self._draw_overlay("GAME OVER", 7, "R", "TO RESTART", 2)

    def pause(
        self,
        board: BoardState,
        next_piece: str,
        score: int,
        level: int,
        game_time: float,
    ) -> None:
        # draws the board first so the prompt sits over the current game state
        self.render(board, next_piece, score, level, game_time)
        self._draw_overlay("PAUSE", 6, "P", "TO RESUME", 3)

    def _center_x(self, text: str, left: int, width: int) -> int:
        return left + (width - len(text)) // 2

    def _draw_overlay(
        self,
        title: str,
        title_color: int,
        key: str,
        action: str,
        key_color: int,
    ) -> None:
        # centers the modal prompt within the visible part of the board
        center_x = self.left + 1 + self.visible_width
        center_y = self.top + 1 + self.hidden_rows + self.visible_height // 2
        prompt = f"PRESS {key} {action}"
        prompt_x = center_x - len(prompt) // 2

        self._safe_addstr(
            center_y,
            center_x - len(title) // 2,
            title,
            curses.color_pair(title_color),
        )
        self._safe_addstr(center_y + 2, prompt_x, "PRESS ", curses.color_pair(9))
        self._safe_addstr(
            center_y + 2,
            prompt_x + len("PRESS "),
            key,
            curses.color_pair(key_color),
        )
        self._safe_addstr(
            center_y + 2,
            prompt_x + len("PRESS ") + len(key),
            f" {action}",
            curses.color_pair(9),
        )
        self.stdscr.refresh()

    def _safe_addstr(self, y: int, x: int, text: str, attr: int = 0) -> None:
        max_y, max_x = self.stdscr.getmaxyx()

        if y < 0 or y >= max_y or x < 0 or x >= max_x:
            return

        available_width = max_x - x
        if available_width <= 0:
            return

        # avoids curses errors when text reaches the bottom-right cell
        if y == max_y - 1:
            available_width -= 1

        if available_width <= 0:
            return

        try:
            self.stdscr.addstr(y, x, text[:available_width], attr)
        except curses.error:
            pass

    def _draw_grid(self, grid: np.ndarray) -> None:
        height, width = grid.shape
        border_attr = curses.color_pair(8)
        dot_attr = curses.color_pair(8) | curses.A_DIM

        for y in range(height):
            screen_y = self.top + 1 + y

            if y >= self.hidden_rows:
                self._safe_addstr(screen_y, self.left, "|", border_attr)

            for x in range(width):
                screen_x = self.left + 1 + x * 2
                cell = int(grid[y, x])

                if cell == 0:
                    if y >= self.hidden_rows:
                        if x % 2 == 0:
                            self._safe_addstr(screen_y, screen_x, "  ")
                        else:
                            self._safe_addstr(screen_y, screen_x, "''", dot_attr)
                else:
                    self._safe_addstr(
                        screen_y,
                        screen_x,
                        "[]",
                        curses.color_pair(cell),
                    )

            if y >= self.hidden_rows:
                self._safe_addstr(
                    screen_y,
                    self.left + 1 + width * 2,
                    "|",
                    border_attr,
                )

        self._safe_addstr(
            self.top + 1 + height,
            self.left,
            "+" + "--" * width + "+",
            border_attr,
        )

    def _draw_box(self, top: int, left: int, width: int, height: int) -> None:
        # draws the shared frame used by both side panels
        border_attr = curses.color_pair(8)
        self._safe_addstr(top, left, "+" + "-" * width + "+", border_attr)

        for y in range(top + 1, top + height + 1):
            self._safe_addstr(y, left, "|", border_attr)
            self._safe_addstr(y, left + width + 1, "|", border_attr)

        self._safe_addstr(top + height + 1, left, "+" + "-" * width + "+", border_attr)

    def _draw_left_sidebar(self, game_time: float) -> None:
        value_attr = curses.color_pair(9)

        sidebar_left = 0
        sidebar_top = self.top + 1 + self.hidden_rows
        sidebar_height = self.visible_height - 1

        self._draw_box(sidebar_top, sidebar_left, self.left_sidebar_width, sidebar_height)

        title = "TUI TETRIS"
        title_x = self._center_x(title, sidebar_left + 1, self.left_sidebar_width)

        title_top_y = sidebar_top + 1

        title_colors = [
            curses.color_pair(1),
            curses.color_pair(2),
            curses.color_pair(3),
            curses.color_pair(6),
            curses.color_pair(7),
            curses.color_pair(4),
        ]

        wave_speed = 1.0
        wave_spacing = .25

        for i, char in enumerate(title):
            phase = game_time * wave_speed - i * wave_spacing

            # shifts each character between two rows to make the title wave
            vertical_offset = 0 if math.sin(phase) >= 0 else 1

            char_y = title_top_y + vertical_offset
            attr = title_colors[i % len(title_colors)]

            self._safe_addstr(
                char_y,
                title_x + i,
                char,
                attr,
            )

        controls_title = "CONTROLS"
        controls_title_x = self._center_x(
            controls_title, sidebar_left + 1, self.left_sidebar_width
        )

        self._safe_addstr(
            sidebar_top + 4,
            controls_title_x,
            controls_title,
            curses.color_pair(8),
        )

        controls = [
            ("A", "MOVE LEFT", 1),   # cyan
            ("D", "MOVE RIGHT", 4),  # blue
            ("S", "MOVE DOWN", 6),   # green
            ("J", "ROTATE L", 9),    # white
            ("K", "ROTATE R", 3),    # pink
            ("P", "PAUSE", 3),       # pink
            ("R", "RESTART", 2),     # yellow
            ("Q", "QUIT", 7),        # red
        ]

        start_y = sidebar_top + 5

        for i, (key, label, color_pair) in enumerate(controls):
            y = start_y + i * 2

            key_x = sidebar_left + 3
            label_x = sidebar_left + 7
            key_attr = curses.color_pair(color_pair)

            self._safe_addstr(
                y,
                key_x,
                key,
                key_attr,
            )
            self._safe_addstr(
                y,
                key_x + 1,
                ":",
                value_attr,
            )
            self._safe_addstr(
                y,
                label_x,
                label,
                value_attr,
            )

    def _draw_right_sidebar(
        self,
        next_piece: str,
        score: int,
        level: int,
        game_time: float,
    ) -> None:
        score_label_attr = curses.color_pair(7)
        level_label_attr = curses.color_pair(3)
        time_label_attr = curses.color_pair(6)
        value_attr = curses.color_pair(9)

        grid_pixel_width = self.visible_width * 2
        sidebar_left = self.left + 1 + grid_pixel_width + 1 + self.sidebar_gap

        sidebar_top = self.top + 1 + self.hidden_rows
        sidebar_height = self.visible_height - 1

        self._draw_box(sidebar_top, sidebar_left, self.sidebar_width, sidebar_height)

        title = "NEXT"
        title_x = self._center_x(title, sidebar_left + 1, self.sidebar_width)
        self._safe_addstr(sidebar_top + 2, title_x, title, curses.color_pair(8))

        shape = PIECE_DEFS[next_piece].shape
        piece_id = PIECE_TO_ID[next_piece]

        _, piece_width = shape.shape

        piece_start_y = sidebar_top + 4
        piece_start_x = sidebar_left + 1 + (self.sidebar_width - piece_width * 2) // 2
        if next_piece == "O":
            piece_start_x -= 1

        for local_y, row in enumerate(shape):
            for local_x, value in enumerate(row):
                if value == 0:
                    continue

                self._safe_addstr(
                    piece_start_y + local_y,
                    piece_start_x + local_x * 2,
                    "[]",
                    curses.color_pair(piece_id),
                )

        score_label = "SCORE"
        score_text = str(score)

        score_label_y = piece_start_y + 4
        score_value_y = score_label_y + 2

        score_label_x = self._center_x(score_label, sidebar_left + 1, self.sidebar_width)
        score_value_x = self._center_x(score_text, sidebar_left + 1, self.sidebar_width)

        self._safe_addstr(score_label_y, score_label_x, score_label, score_label_attr)
        self._safe_addstr(score_value_y, score_value_x, score_text, value_attr)

        level_label = "LEVEL"
        level_text = str(level)

        level_label_y = score_value_y + 2
        level_value_y = level_label_y + 2

        level_label_x = self._center_x(level_label, sidebar_left + 1, self.sidebar_width)
        level_value_x = self._center_x(level_text, sidebar_left + 1, self.sidebar_width)

        self._safe_addstr(level_label_y, level_label_x, level_label, level_label_attr)
        self._safe_addstr(level_value_y, level_value_x, level_text, value_attr)

        time_label = "TIME"
        time_text = self._format_time(game_time)

        time_value_y = sidebar_top + sidebar_height - 1
        time_label_y = time_value_y - 2

        time_label_x = self._center_x(time_label, sidebar_left + 1, self.sidebar_width)
        time_value_x = self._center_x(time_text, sidebar_left + 1, self.sidebar_width)

        self._safe_addstr(time_label_y, time_label_x, time_label, time_label_attr)
        self._safe_addstr(time_value_y, time_value_x, time_text, value_attr)
