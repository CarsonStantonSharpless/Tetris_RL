# TUI Tetris

A quick terminal based tetris implementation. This was a nice weekend project that I developed to be rolled into a future Tetris reinforcement learning project.

![Demo of the application](assets/TetrisDemo.gif)

## Requirements

- Python 3.10 or newer
- `numpy`
- `tqdm`

## Run

From the project directory:

```bash
python3 run.py
```

This runs the random policy with the BFS placer. Use `python3 run.py --help` to
control the player, display, timing, random seed, run length, and game-state
output.

For example, run headless for 1,000 ticks and save the game:

```bash
python3 run.py --no-display --max-ticks 1000 --filepath game.trs
```

For training or evaluation, apply each policy-selected placement in one tick:

```bash
python3 run.py --no-display --instant-placement --max-ticks 1000
```

Instant placement bypasses movement planning and soft-drop points. Each tick
locks one piece, clears rows, updates the score, and spawns the next piece.

Heuristic weights are optional and can be overridden individually:

```bash
python3 run.py --policy heuristic --alpha .3 --beta .4 --gamma .1 --delta .2
```

Run a genetic tournament and save its captain and lieutenant:

```bash
python3 -m training.genetic.run_tournament 10 100 --filepath leaders.json
```

Progress bars and leader fitness are shown by default; pass `--no-visualize` to
disable them. Episode evaluation uses one process per CPU by default; pass
`--workers 1` for serial evaluation or `--workers N` to set a limit. The saved
captain can then drive the genetic heuristic policy. Episodes are capped at
200 piece placements during the GA and the finalists are re-evaluated at 1,000.
Change the finalist horizon with `--max-ticks N`; GA rounds use one-fifth of it.

```bash
python3 run.py --policy genetic-heuristic --params-file leaders.json
```

Without `--params-file`, the genetic heuristic uses the regular heuristic's
default weights. Loading saved parameters never runs a tournament.

The display will try to resize compatible terminals to fit the interface. If
your terminal does not support that, enlarge it manually before starting.

## Controls

Movement keys are enabled with `--interactive`; pause, restart, and quit remain
available in the default RL-controlled display.

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
run.py                 configures and starts the player
core/                  Tetris rules and engine
agents/                policies, placements, and player control
training/genetic/      genetic optimizer code
training/reinforcement/ reinforcement learning code
models/                parameterized heuristic and neural models
storage/               game, checkpoint, and metric persistence
visualization/         board, game, and learning visualizations
tui/                   live terminal renderer
tests/                 automated tests
runs/                  generated experiment output
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
