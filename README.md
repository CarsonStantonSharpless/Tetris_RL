# TUI Tetris

A quick terminal based tetris implementation. This was a nice weekend project that I developed to be rolled into a future Tetris reinforcement learning project.

![Demo of the application](assets/TetrisDemo.gif)

## Requirements

- Python 3.10 or newer
- `numpy`
- `tqdm`
- `matplotlib`
- `torch` (for Double DQN and PPO training/inference)

Install them with:

```bash
python3 -m pip install -r requirements.txt
```

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

Train linear TD weights in parallel:

```bash
python3 -m training.reinforcement.run_linear_td 1000 --batch-size 8 --workers 8
```

Pass `--no-display` to disable the live loss plot. Results are written under
`runs/`, including the loss plot, checkpoints, and winning weights.

The display will try to resize compatible terminals to fit the interface. If
your terminal does not support that, enlarge it manually before starting.

## Double DQN

The Double DQN learns from the same instant-placement `Player` interface used
by the heuristic trainer. Its compact convolutional `DQN` scores every legal
placement's afterstate, current piece, preview piece, and level. Train
headlessly with:

```bash
python3 -m training.reinforcement.run_double_dqn 10000 --no-display
```

Training keeps the game's normal score reward and adds a `-1000` terminal
penalty when a placement causes game over. Override it with
`--terminal-penalty 0` to disable that training-only penalty.

Optionally constrain exploration and DQN decisions to the eight placements
ranked highest by the existing heuristic:

```bash
python3 -m training.reinforcement.run_double_dqn 10000 --heuristic-top-k
```

Pass a number such as `--heuristic-top-k 12` to choose a different limit.
The constraint is persistent: it applies to training choices, bootstrap
targets, and evaluation without changing or shaping the game's rewards.
Omitting the option runs an unconstrained Double DQN.

Each run writes `training.png` (loss, score, exploration, Q values, replay
size, and evaluation score), the most-recent playable `model.pt`, and a
protected `best_model.pt` whenever fixed-seed evaluation reaches a new high.
`latest.pt` and numbered checkpoints are resumable training state. Play the
best saved model with:

```bash
python3 run.py --policy dqn --params-file runs/double_dqn_.../best_model.pt \
  --no-display --instant-placement --max-ticks 1000
```

For a constrained model, add `--heuristic-top-k` when playing it (or pass the
explicit K used during training) to preserve the same action space.

Resume training from the most recent full checkpoint:

```bash
python3 -m training.reinforcement.run_double_dqn 5000 \
  --resume-from runs/double_dqn_.../latest.pt --no-display
```

For long headless runs, add `--plot-every 500` to avoid redrawing the plot
after every episode while still saving regular progress snapshots.

## PPO

PPO uses the same instant-placement environment interface as Double DQN, but
the thing it learns is different. For each environment state:

1. The environment emits the current board and all legal final placements.
2. The actor scores each legal placement afterstate and samples one action
   from the resulting categorical policy, `P(a | s)`.
3. The critic estimates `V(s)` from the current board before that action.
4. The environment applies the placement and emits the real score reward, the
   next state, and whether the game ended.
5. GAE combines the critic values with those rewards without crossing terminal
   boundaries. It emits advantages for the actor and return targets for the
   critic.
6. PPO compares `P_new(a | s) / P_old(a | s)` and clips overly large helpful
   changes before updating the shared actor-critic model.

The code follows those same boundaries: the model and playable policy live in
`agents/policies/ppo.py`, the standalone GAE calculation lives in
`training/reinforcement/ppo/gae.py`, and the commented rollout/update loop
lives in `training/reinforcement/ppo/trainer.py`.

Train headlessly with:

```bash
python3 -m training.reinforcement.run_ppo 10000 --no-display
```

By default, the actor remains unchanged while 1,024 fresh placement transitions
are collected. The rollout can contain several games or cut through one long
game. GAE stops at each terminal boundary and bootstraps the critic when the
fixed rollout boundary cuts through a game. Change the fixed compute budget
with `--rollout-steps N`.

An optional potential reward can provide a very small bottom-up hint during
early learning:

```bash
python3 -m training.reinforcement.run_ppo 10000 --bottom-up-bias --no-display
```

The hint values free headroom and penalizes covered holes. It is added as the
change in that board potential, not as a replacement for the environment
reward. Passing the flag alone uses a weight of one raw score point, so an
ordinary preference is only a few points beside a 40-point line clear. Pass an
explicit value such as `--bottom-up-bias 0.5` for an even lighter hint. It is
off by default.

The run writes `model.pt`, a resumable `latest.pt`, periodic checkpoints, and
`best_model.pt` selected by fixed-seed evaluation. Each point in its nine-panel
`training.png` represents one fixed-rollout update and separates mean environment
score, actor loss, critic loss, entropy, PPO clipping, TD residuals, and the
mean absolute GAE advantage.

Play the best actor greedily through the regular player CLI:

```bash
python3 run.py --policy ppo --params-file runs/ppo_.../best_model.pt \
  --no-display --instant-placement --max-ticks 1000
```

Resume a training run with its full checkpoint:

```bash
python3 -m training.reinforcement.run_ppo 5000 \
  --resume-from runs/ppo_.../latest.pt --no-display
```

As with DQN, `--heuristic-top-k` optionally limits the legal action set during
training, evaluation, and play. Use the same value in all three places.

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
