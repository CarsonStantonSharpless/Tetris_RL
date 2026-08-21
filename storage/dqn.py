"""Persistence helpers for DQN inference models and resumable training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agents.policies.dqn import DQN, require_torch


MODEL_FORMAT = "tetris-double-dqn-v1"


def write_dqn_model(model: Any, filepath: str | Path) -> None:
    """Save the online network in the compact format used for playing."""
    write_dqn_model_state(model.state_dict(), filepath)


def write_dqn_model_state(model_state: dict[str, Any], filepath: str | Path) -> None:
    """Save a captured network state in the compact inference format."""
    torch = require_torch()
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": MODEL_FORMAT,
            "model_state": model_state,
        },
        path,
    )


def load_dqn_model(filepath: str | Path, device: Any) -> Any:
    """Load an inference model or the online network from a training checkpoint."""
    checkpoint = _torch_load(filepath, device)
    if not isinstance(checkpoint, dict):
        raise ValueError("DQN file must contain a checkpoint dictionary")
    model_state = checkpoint.get("model_state")
    if model_state is None:
        raise ValueError("DQN file does not contain model_state")
    model = DQN().to(device)
    model.load_state_dict(model_state)
    return model


def write_dqn_checkpoint(checkpoint: dict[str, Any], filepath: str | Path) -> None:
    """Persist all state needed to resume a Double DQN training run."""
    torch = require_torch()
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": MODEL_FORMAT, **checkpoint}, path)


def read_dqn_checkpoint(filepath: str | Path, device: Any) -> dict[str, Any]:
    """Read and validate a previously saved training checkpoint."""
    checkpoint = _torch_load(filepath, device)
    if not isinstance(checkpoint, dict):
        raise ValueError("DQN checkpoint must contain a dictionary")
    required = {
        "model_state",
        "target_model_state",
        "optimizer_state",
        "episode",
        "environment_steps",
        "update_steps",
        "replay_buffer",
        "agent_rng_state",
    }
    missing = sorted(required - checkpoint.keys())
    if missing:
        raise ValueError(f"DQN checkpoint is missing: {', '.join(missing)}")
    return checkpoint


def _torch_load(filepath: str | Path, device: Any) -> Any:
    torch = require_torch()
    path = Path(filepath)
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch before the weights_only parameter.
        return torch.load(path, map_location=device)
