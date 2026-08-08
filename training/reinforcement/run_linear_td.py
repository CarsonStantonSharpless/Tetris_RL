"""Command-line entry point for linear TD training."""

import argparse
import sys

from training.reinforcement.linear_TD import linear_TD


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train linear TD weights.")
    parser.add_argument("num_games", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--max-placements", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-directory")
    parser.add_argument(
        "--display",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="show the live loss plot (default: on)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        linear_TD(
            num_games=args.num_games,
            batch_size=args.batch_size,
            workers=args.workers,
            learning_rate=args.learning_rate,
            discount=args.discount,
            epsilon=args.epsilon,
            max_placements=args.max_placements,
            seed=args.seed,
            run_directory=args.run_directory,
            display=args.display,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted.", file=sys.stderr)
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
