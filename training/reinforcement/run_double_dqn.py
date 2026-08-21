"""Command-line entry point for replay-based Double DQN training."""

from __future__ import annotations

import argparse
import sys

from agents.policies.dqn import device_for
from training.reinforcement.double_DQN import double_dqn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Tetris Double DQN.")
    parser.add_argument("num_games", type=int, help="number of new episodes")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--replay-size", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=1_000)
    parser.add_argument("--train-every", type=int, default=1)
    parser.add_argument("--target-update", type=int, default=1_000)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=50_000)
    parser.add_argument("--reward-scale", type=float, default=1_200.0)
    parser.add_argument(
        "--terminal-penalty",
        type=float,
        default=-1_000.0,
        help="raw reward added when a placement ends the game (default: -1000)",
    )
    parser.add_argument("--gradient-clip", type=float, default=10.0)
    parser.add_argument("--max-placements", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-directory")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda")
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--evaluation-games", type=int, default=10)
    parser.add_argument(
        "--plot-every",
        type=int,
        default=1,
        help="save the training graph every N episodes when headless (default: 1)",
    )
    parser.add_argument("--resume-from", help="a previous latest.pt or checkpoint file")
    parser.add_argument(
        "--display",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="show the live six-panel training plot (default: on)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        selected_device = device_for(args.device)
        print(f"Training device: {selected_device}")
        model_path = double_dqn(
            args.num_games,
            learning_rate=args.learning_rate,
            discount=args.discount,
            replay_size=args.replay_size,
            batch_size=args.batch_size,
            warmup_steps=args.warmup_steps,
            train_every=args.train_every,
            target_update=args.target_update,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay_steps=args.epsilon_decay_steps,
            reward_scale=args.reward_scale,
            terminal_penalty=args.terminal_penalty,
            gradient_clip=args.gradient_clip,
            max_placements=args.max_placements,
            seed=args.seed,
            run_directory=args.run_directory,
            display=args.display,
            device=str(selected_device),
            checkpoint_every=args.checkpoint_every,
            evaluation_games=args.evaluation_games,
            plot_every=args.plot_every,
            resume_from=args.resume_from,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted.", file=sys.stderr)
        raise SystemExit(130) from None
    except (RuntimeError, ValueError) as error:
        print(f"Double DQN training failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    print(f"Saved latest playable model to {model_path}")
    best_model_path = model_path.with_name("best_model.pt")
    if best_model_path.exists():
        print(f"Best fixed-seed evaluation model: {best_model_path}")


if __name__ == "__main__":
    main()
