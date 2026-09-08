"""Persistence helpers for PPO inference models and resumable training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agents.policies.dqn import require_torch
from agents.policies.ppo import PPO


MODEL_FORMAT = "tetris-ppo-v1"


def write_ppo_model(model: Any, filepath: str | Path) -> None:
    """Save the actor-critic network in the compact format used for playing."""
    write_ppo_model_state(model.state_dict(), filepath)


def write_ppo_model_state(model_state: dict[str, Any], filepath: str | Path) -> None:
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


def write_ppo_checkpoint(checkpoint: dict[str, Any], filepath: str | Path) -> None:
    """Persist all state needed to resume a PPO training run."""
    torch = require_torch()
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": MODEL_FORMAT, **checkpoint}, path)


def read_ppo_checkpoint(filepath: str | Path, device: Any) -> dict[str, Any]:
    """Read and validate a previously saved PPO training checkpoint."""
    checkpoint = _torch_load(filepath, device)
    if not isinstance(checkpoint, dict):
        raise ValueError("PPO checkpoint must contain a dictionary")
    if checkpoint.get("format") != MODEL_FORMAT:
        raise ValueError("file is not a supported PPO checkpoint")
    required = {
        "config",
        "model_state",
        "optimizer_state",
        "episode",
        "environment_steps",
        "update_steps",
        "agent_rng_state",
    }
    missing = sorted(required - checkpoint.keys())
    if missing:
        raise ValueError(f"PPO checkpoint is missing: {', '.join(missing)}")
    return checkpoint


def load_ppo_model(filepath: str | Path, device: Any) -> Any:
    """Load an inference model or the model from a training checkpoint."""
    checkpoint = _torch_load(filepath, device)
    if not isinstance(checkpoint, dict):
        raise ValueError("PPO file must contain a checkpoint dictionary")
    if checkpoint.get("format") != MODEL_FORMAT:
        raise ValueError("file is not a supported PPO model or checkpoint")
    model_state = checkpoint.get("model_state")
    if model_state is None:
        raise ValueError("PPO file does not contain model_state")
    model = PPO().to(device)
    model.load_state_dict(model_state)
    return model


def _torch_load(filepath: str | Path, device: Any) -> Any:
    torch = require_torch()
    path = Path(filepath)
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch before the weights_only parameter.
        return torch.load(path, map_location=device)
