"""The actor-critic network used by the PPO placement policy.

The Tetris environment has a different number of legal placements in every
state.  Just like the DQN, this model avoids a fixed-size action output by
scoring one legal *afterstate* at a time:

* The actor turns all legal afterstate scores into ``P(a | s)``.
* The critic assigns one value ``V(s)`` to the current pre-action state.

The actor and critic share the small board encoder.  Their final heads are
separate because an action preference and a state value answer different
questions.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agents.placements import find_possible_states
from agents.policies.dqn import BOARD_SHAPE, device_for, piece_id, require_torch
from agents.policies.heuristic import top_heuristic_placements

if TYPE_CHECKING:
    from core.board import BoardState

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # Keep non-neural policies usable without PyTorch.
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


if nn is not None:
    class PPO(nn.Module):
        """A small shared encoder with an actor head and a critic head."""

        def __init__(self) -> None:
            super().__init__()
            # This encoder deliberately mirrors the DQN model so the two
            # algorithms differ in their learning rule, not their capacity.
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
            feature_count = 128 + 12 + 12 + 1
            self.actor = nn.Sequential(
                nn.Linear(feature_count, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )
            self.critic = nn.Sequential(
                nn.Linear(feature_count, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

        def _encode(
            self,
            boards: Any,
            current_pieces: Any,
            next_pieces: Any,
            levels: Any,
        ) -> Any:
            """Encode either current-state boards or candidate afterstates."""
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
            level_features = levels.float().unsqueeze(1) / 20.0
            return torch.cat(
                (board_features, current_features, next_features, level_features),
                dim=1,
            )

        def policy_logits(
            self,
            afterstate_boards: Any,
            current_pieces: Any,
            next_pieces: Any,
            levels: Any,
        ) -> Any:
            """Return one unnormalized actor score for each legal action."""
            features = self._encode(
                afterstate_boards,
                current_pieces,
                next_pieces,
                levels,
            )
            return self.actor(features).squeeze(1)

        def state_values(
            self,
            state_boards: Any,
            current_pieces: Any,
            next_pieces: Any,
            levels: Any,
        ) -> Any:
            """Return the critic's estimate ``V(s)`` for each current state."""
            features = self._encode(
                state_boards,
                current_pieces,
                next_pieces,
                levels,
            )
            return self.critic(features).squeeze(1)

        def forward(
            self,
            afterstate_boards: Any,
            current_pieces: Any,
            next_pieces: Any,
            levels: Any,
        ) -> Any:
            """Alias the usual model call to the actor used while playing."""
            return self.policy_logits(
                afterstate_boards,
                current_pieces,
                next_pieces,
                levels,
            )
else:
    class PPO:  # type: ignore[no-redef]  # pragma: no cover
        """Placeholder that explains the missing optional dependency."""

        def __init__(self) -> None:
            require_torch()


@lru_cache(maxsize=8)
def _stored_model(filepath: str, device_name: str) -> Any:
    """Load each requested inference model once per process."""
    from storage.ppo import load_ppo_model

    model = load_ppo_model(filepath, device_for(device_name))
    model.eval()
    return model


def clear_ppo_model_cache() -> None:
    """Forget cached inference models after training overwrites a file."""
    _stored_model.cache_clear()


def ppo_policy(
    start_state: BoardState,
    harddrop: bool = True,
    filepath: str | None = None,
    next_piece: str | None = None,
    level: int = 0,
    device: str = "auto",
    heuristic_top_k: int | None = None,
) -> BoardState:
    """Choose the most probable legal placement from a saved PPO actor."""
    if filepath is None:
        raise ValueError("PPO policy requires a saved model filepath")
    if next_piece is None:
        raise ValueError("PPO policy requires the engine's next piece")

    from agents.policies.dqn import placement_boards

    torch_module = require_torch()
    placements = find_possible_states(start_state, harddrop)
    placements = top_heuristic_placements(placements, heuristic_top_k)
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
            dtype=torch_module.float32,
            device=model_device,
        )
        choice = int(torch_module.argmax(
            model(
                board_tensor,
                current_piece_tensor,
                next_piece_tensor,
                level_tensor,
            )
        ).item())
    return placements[choice]
