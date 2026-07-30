from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from training.genetic.fighter import Parameters


PARAMETER_NAMES = ("alpha", "beta", "gamma", "delta", "epsilon")


def write_genetic_parameters(
    captain: Parameters,
    lieutenant: Parameters,
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
    """Load one leader's heuristic parameters from a tournament JSON file."""
    with Path(filepath).open(encoding="utf-8") as file:
        leaders = json.load(file)

    if leader not in leaders:
        raise ValueError(f"parameter file does not contain {leader!r}")

    parameters = leaders[leader]
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


def _parameter_dict(parameters: Parameters) -> dict[str, float]:
    return {
        name: float(getattr(parameters, name))
        for name in PARAMETER_NAMES
    }
