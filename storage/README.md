# Board State Storage

`games.py` writes lists of `BoardState` objects to a small binary file.

```python
from storage.games import read_board_states, write_board_states

write_board_states(states, "game.trs")
states = read_board_states("game.trs")
```

## Format

Each active piece uses two bytes: two bits for the record type, three for the
piece, two for its orientation, and nine for its position.

The first state stores the complete grid using three bits per cell. Later
states store no grid when it is unchanged, store changed cells when the change
is small, or store another complete grid when that is smaller. A complete state
uses 85 bytes; a state with an unchanged grid uses two.

Files begin with `TRS1` and the number of stored states. The versioned header
allows the format to change later without confusing old files.

## Genetic Parameters

`parameters.py` stores the captain and lieutenant from a genetic tournament as
JSON and loads either parameter set:

```python
from storage.parameters import read_genetic_parameters, write_genetic_parameters

write_genetic_parameters(captain, lieutenant, "leaders.json")
captain_parameters = read_genetic_parameters("leaders.json")
```

## Double DQN Models

`dqn.py` writes a compact latest playable `model.pt`, a protected
`best_model.pt` selected by fixed-seed evaluation, and full training
checkpoints. The latter also contain the target network, optimizer, replay
buffer, random state, hyperparameters, and best-model record, so they can
resume a Double DQN run exactly where the prior checkpoint stopped.
