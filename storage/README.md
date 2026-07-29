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
