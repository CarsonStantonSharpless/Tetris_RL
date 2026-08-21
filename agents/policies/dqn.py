"""The neural afterstate scorer used by the Double DQN player policy.

The model scores one legal placement at a time. Its board input is the grid
after that placement has locked and cleared rows; the current piece, preview
piece, and level retain the rest of the decision-state context. Scoring
afterstates keeps the variable number of legal Tetris placements out of the
network API.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from agents.placements import find_possible_states
from core.board import Board
from core.pieces import PIECE_TO_ID

if TYPE_CHECKING:
    from core.board import BoardState

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # Keep non-DQN policies usable without PyTorch.
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


BOARD_SHAPE = (22, 10)


def require_torch() -> Any:
    """Return PyTorch or raise a helpful error when DQN support is requested."""
    if torch is None:
        raise RuntimeError(
            "DQN support requires PyTorch. Install the project dependencies "
            "with `python3 -m pip install -r requirements.txt`."
        )
    return torch


def device_for(name: str = "auto") -> Any:
    """Choose an available device, with ``auto`` preferring accelerators."""
    torch_module = require_torch()
    if name == "auto":
        if torch_module.cuda.is_available():
            return torch_module.device("cuda")
        if getattr(torch_module.backends, "mps", None) and (
            torch_module.backends.mps.is_available()
        ):
            return torch_module.device("mps")
        return torch_module.device("cpu")
    return torch_module.device(name)


if nn is not None:
    class DQN(nn.Module):
        """A small convolutional network that scores a Tetris afterstate."""

        def __init__(self) -> None:
            super().__init__()
            self.convolutions = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Flatten(),
            )
            self.board_features = nn.Sequential(
                nn.Linear(32 * BOARD_SHAPE[0] * BOARD_SHAPE[1], 128),
                nn.ReLU(),
            )
            # Piece ids are one through seven; zero remains an unused pad id.
            self.piece_embedding = nn.Embedding(8, 12, padding_idx=0)
            self.head = nn.Sequential(
                nn.Linear(128 + 12 + 12 + 1, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

        def forward(
            self,
            boards: Any,
            current_pieces: Any,
            next_pieces: Any,
            levels: Any,
        ) -> Any:
            """Return one Q value for every legal-placement afterstate."""
            if boards.ndim == 3:
                boards = boards.unsqueeze(1)
            if boards.ndim != 4 or tuple(boards.shape[-2:]) != BOARD_SHAPE:
                raise ValueError(
                    "boards must have shape (batch, 22, 10) or "
                    "(batch, 1, 22, 10)"
                )
            context = (current_pieces, next_pieces, levels)
            if any(value.ndim != 1 for value in context):
                raise ValueError("piece and level inputs must have shape (batch,)")
            if any(boards.shape[0] != value.shape[0] for value in context):
                raise ValueError("board and context inputs must share a batch size")

            board_features = self.board_features(self.convolutions(boards.float()))
            current_features = self.piece_embedding(current_pieces.long())
            next_features = self.piece_embedding(next_pieces.long())
            # Level is bounded by the engine and only affects score multipliers.
            level_features = levels.float().unsqueeze(1) / 20.0
            features = torch.cat(
                (board_features, current_features, next_features, level_features),
                dim=1,
            )
            return self.head(features).squeeze(1)
else:
    class DQN:  # pragma: no cover - exercised only without the optional dependency
        """Placeholder that explains the missing optional dependency."""

        def __init__(self) -> None:
            require_torch()


def piece_id(piece: str) -> int:
    """Convert a Tetris piece name to the model's embedding id."""
    try:
        return PIECE_TO_ID[piece]
    except KeyError as error:
        raise ValueError(f"unknown Tetris piece: {piece!r}") from error


def placement_boards(placements: list[BoardState]) -> np.ndarray:
    """Encode legal placement afterstates as immutable occupancy grids."""
    if not placements:
        raise ValueError("at least one legal placement is required")
    return np.stack([
        (Board.simulate_placement(placement).grid != 0).astype(np.float32)
        for placement in placements
    ])


@lru_cache(maxsize=8)
def _stored_model(filepath: str, device_name: str) -> Any:
    """Load each requested inference model once per process."""
    from storage.dqn import load_dqn_model

    model = load_dqn_model(filepath, device_for(device_name))
    model.eval()
    return model


def clear_dqn_model_cache() -> None:
    """Forget cached inference models after a training run overwrites a file."""
    _stored_model.cache_clear()


def dqn_policy(
    start_state: BoardState,
    harddrop: bool = True,
    filepath: str | None = None,
    next_piece: str | None = None,
    level: int = 0,
    device: str = "auto",
) -> BoardState:
    """Choose the highest-scoring legal placement from a stored DQN model."""
    if filepath is None:
        raise ValueError("DQN policy requires a saved model filepath")
    if next_piece is None:
        raise ValueError("DQN policy requires the engine's next piece")

    torch_module = require_torch()
    placements = find_possible_states(start_state, harddrop)
    boards = placement_boards(placements)
    model_device = device_for(device)
    model = _stored_model(str(Path(filepath).resolve()), str(model_device))

    with torch_module.no_grad():
        board_tensor = torch_module.from_numpy(boards).to(model_device)
        current_piece_tensor = torch_module.full(
            (len(placements),),
            piece_id(start_state.curr_piece.kind),
            dtype=torch_module.long,
            device=model_device,
        )
        next_piece_tensor = torch_module.full(
            (len(placements),),
            piece_id(next_piece),
            dtype=torch_module.long,
            device=model_device,
        )
        level_tensor = torch_module.full(
            (len(placements),),
            level,
            dtype=torch.float32,
            device=model_device,
        )
        choice = int(torch_module.argmax(
            model(board_tensor, current_piece_tensor, next_piece_tensor, level_tensor)
        ).item())
    return placements[choice]
