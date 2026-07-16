# TUI Tetris

A quick terminal based tetris implementation. This was a nice weekend project that I developed to be rolled into a future Tetris reinforcement learning project.

![Demo of the application](assets/TetrisDemo.gif)

## Requirements

- Python 3.10 or newer
- `numpy`

## Run

From the project directory:

```bash
python3 run.py
```

The game will try to resize compatible terminals to fit the interface. If your terminal does not support that, enlarge it manually before starting.

## Controls

| Key | Action |
| --- | --- |
| `A` | Move left |
| `D` | Move right |
| `S` | Soft drop |
| `J` | Rotate left |
| `K` | Rotate right |
| `P` | Pause or resume |
| `R` | Restart |
| `Q` | Quit |

## Project Files

```text
run.py           starts the curses application
core/engine.py   handles timing, input, scoring, levels, and game flow
core/board.py    owns the grid, collision checks, locking, and line clears
core/pieces.py   defines tetromino shapes and rotation state
tui/renderer.py  draws the game interface in the terminal
```

## Headless / RL use

Create the engine with rendering disabled, then call `step()` with zero or more
control commands. Each call returns an `EngineState` dataclass containing a
snapshot of the board, the next piece, and the score.

```python
from core.engine import Engine

engine = Engine(render=False, tick_speed=1 / 60)
state = engine.state
state = engine.step("a")

locked_grid = state.board_state.grid
active_piece = state.board_state.curr_piece
next_piece = state.next_piece
score = state.score
```

`tick_speed` is measured in seconds per tick. `step()` never sleeps, letting a
training environment run as fast as it needs to; `tick_speed` controls the
interactive renderer's frame pacing and the engine's reported game time.
