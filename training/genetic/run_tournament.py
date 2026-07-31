"""Command-line entry point for the genetic tournament."""

import argparse
import random
import sys

import numpy as np

from training.genetic.tournament import tournament


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the genetic tournament.")
    parser.add_argument("num_rounds", type=int)
    parser.add_argument("num_fighters", type=int)
    parser.add_argument("--num-games", type=int, default=1000)
    parser.add_argument(
        "--max-ticks",
        type=int,
        default=1000,
        help="finalist horizon; GA rounds use one-fifth (default: 1000)",
    )
    parser.add_argument("--cull-percent", type=float, default=0.3)
    parser.add_argument("--tournament-percent", type=float, default=0.1)
    parser.add_argument("--mutation-chance", type=float, default=0.75)
    parser.add_argument("--mutation-amount", type=float, default=0.15)
    parser.add_argument(
        "--workers",
        type=int,
        help="episode worker processes (default: CPU count)",
    )
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

    try:
        tournament(
            num_rounds=args.num_rounds,
            num_fighters=args.num_fighters,
            num_games=args.num_games,
            max_ticks=args.max_ticks,
            cull_percent=args.cull_percent,
            tournament_percent=args.tournament_percent,
            mutation_chance=args.mutation_chance,
            mutation_amount=args.mutation_amount,
            workers=args.workers,
            filepath=args.filepath,
            visualize=args.visualize,
        )
    except KeyboardInterrupt:
        print("\nTournament interrupted.", file=sys.stderr)
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
