from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from training.genetic.fighter import Genome


PARAMETER_NAMES = ("alpha", "beta", "gamma", "delta", "epsilon")


def write_genetic_parameters(
    captain: Genome,
    lieutenant: Genome,
    filepath: str,
) -> None:
    """Save the two strongest parameter sets as JSON."""
    leaders = {
        "captain": _parameter_dict(captain),
        "lieutenant": _parameter_dict(lieutenant),
    }
    with Path(filepath).open("w", encoding="utf-8") as file:
        json.dump(leaders, file, indent=2)
        file.write("\n")


def read_genetic_parameters(
    filepath: str,
    leader: str = "captain",
) -> dict[str, float]:
    """Backward-compatible name for loading saved heuristic weights."""
    return read_heuristic_parameters(filepath, leader)


def read_heuristic_parameters(
    filepath: str,
    leader: str = "captain",
) -> dict[str, float]:
    """Load genetic-leader or trained heuristic weights."""
    with Path(filepath).open(encoding="utf-8") as file:
        leaders = json.load(file)

    if "weights" in leaders:
        parameters = leaders["weights"]
    elif leader in leaders:
        parameters = leaders[leader]
    else:
        raise ValueError(f"parameter file does not contain {leader!r}")
    if not isinstance(parameters, dict):
        raise ValueError(f"{leader!r} parameters must be a JSON object")

    try:
        return {
            name: float(parameters[name])
            for name in PARAMETER_NAMES
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"{leader!r} must contain numeric values for "
            f"{', '.join(PARAMETER_NAMES)}"
        ) from error


def _parameter_dict(parameters: Genome) -> dict[str, float]:
    return {
        name: float(getattr(parameters, name))
        for name in PARAMETER_NAMES
    }


def write_weights(weights: np.ndarray, filepath: str | Path) -> None:
    """Write one trained heuristic weight vector."""
    _write_json(filepath, {"weights": _weight_dict(weights)})


def write_checkpoint(
    weights: np.ndarray,
    progress: int,
    filepath: str | Path,
) -> None:
    """Append one weight checkpoint to a training-session JSON file."""
    path = Path(filepath)
    data = json.loads(path.read_text()) if path.exists() else {"checkpoints": []}
    data["checkpoints"].append({
        "progress": progress,
        "weights": _weight_dict(weights),
    })
    _write_json(path, data)


def _weight_dict(weights: np.ndarray) -> dict[str, float]:
    return dict(zip(PARAMETER_NAMES, map(float, weights), strict=True))


def _write_json(filepath: str | Path, data: dict) -> None:
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
