import argparse
import random
import sys
from collections.abc import Iterator

import numpy as np

from training.genetic.population import Population
from training.genetic.fighter import Fighter, Parameters
from storage.parameters import write_genetic_parameters


def tournament(
    num_rounds: int,
    num_fighters: int,
    num_games: int = 1000,
    cull_percent: float = 0.3,
    tournament_percent: float = 0.1,
    mutation_chance: float = 0.5,
    mutation_amount: float = .1,
    filepath: str | None = None,
    visualize: bool = True,
) -> tuple[Parameters, Parameters]:
    if num_rounds < 0:
        raise ValueError("num_rounds must be zero or greater")
    if num_fighters < 2:
        raise ValueError("num_fighters must be at least two")
    if num_games < 0:
        raise ValueError("num_games must be zero or greater")
    if not 0.0 <= cull_percent <= 1.0:
        raise ValueError("cull_percent must be between zero and one")
    if not 0.0 < tournament_percent <= 1.0:
        raise ValueError("tournament_percent must be greater than zero and at most one")

    fighters = [
        Fighter(num_games=num_games)
        for _ in _progress(
            num_fighters,
            "Initial population",
            visualize,
        )
    ]
    seed_pop = Population(fighters=fighters)
    _show_leadership(seed_pop, "Initial population", visualize)

    for round_index in range(num_rounds):
        offspring: list[Fighter] = []
        offspring_count = int(len(seed_pop) * cull_percent)
        for _ in _progress(
            offspring_count,
            f"Round {round_index + 1}/{num_rounds}",
            visualize,
        ):
            sub_pop = seed_pop.subpopulation(tournament_percent)
            child = sub_pop.generate_offspring(
                mutation_chance,
                mutation_amount,
                num_games,
            )
            if child is None:
                raise RuntimeError("cannot generate offspring from an empty population")
            offspring.append(child)

        seed_pop.cull(cull_percent)
        seed_pop += Population(fighters=offspring)
        _show_leadership(
            seed_pop,
            f"Round {round_index + 1}/{num_rounds}",
            visualize,
        )

    if seed_pop.captain is None or seed_pop.lieutenant is None:
        raise RuntimeError("tournament finished without two leaders")

    captain = seed_pop.captain.params
    lieutenant = seed_pop.lieutenant.params
    if filepath is not None:
        write_genetic_parameters(captain, lieutenant, filepath)

    return captain, lieutenant


def _progress(
    count: int,
    description: str,
    enabled: bool,
) -> Iterator[int]:
    if not enabled:
        yield from range(count)
        return

    width = 30
    _draw_progress(description, 0, count, width)
    for index in range(count):
        yield index
        _draw_progress(description, index + 1, count, width)
    print(file=sys.stderr)


def _draw_progress(
    description: str,
    completed: int,
    total: int,
    width: int,
) -> None:
    fraction = completed / total if total else 1.0
    filled = int(width * fraction)
    bar = "#" * filled + "-" * (width - filled)
    print(
        f"\r{description}: [{bar}] {completed}/{total}",
        end="",
        file=sys.stderr,
        flush=True,
    )


def _show_leadership(
    population: Population,
    description: str,
    enabled: bool,
) -> None:
    if not enabled:
        return

    captain_fitness = (
        population.captain.fitness_score
        if population.captain is not None
        else "n/a"
    )
    lieutenant_fitness = (
        population.lieutenant.fitness_score
        if population.lieutenant is not None
        else "n/a"
    )
    print(
        f"{description}: captain fitness={captain_fitness}, "
        f"lieutenant fitness={lieutenant_fitness}",
        file=sys.stderr,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the genetic tournament.")
    parser.add_argument("num_rounds", type=int)
    parser.add_argument("num_fighters", type=int)
    parser.add_argument("--num-games", type=int, default=1000)
    parser.add_argument("--cull-percent", type=float, default=0.3)
    parser.add_argument("--tournament-percent", type=float, default=0.1)
    parser.add_argument("--mutation-chance", type=float, default=0.5)
    parser.add_argument("--mutation-amount", type=float, default=0.1)
    parser.add_argument("--filepath", help="save captain and lieutenant parameters as JSON")
    parser.add_argument(
        "--visualize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="show progress bars and leader fitness (default: on)",
    )
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    tournament(
        num_rounds=args.num_rounds,
        num_fighters=args.num_fighters,
        num_games=args.num_games,
        cull_percent=args.cull_percent,
        tournament_percent=args.tournament_percent,
        mutation_chance=args.mutation_chance,
        mutation_amount=args.mutation_amount,
        filepath=args.filepath,
        visualize=args.visualize,
    )


if __name__ == "__main__":
    main()
