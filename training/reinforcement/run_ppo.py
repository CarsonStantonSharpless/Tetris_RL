"""Command-line entry point for readable PPO training."""

from __future__ import annotations

import argparse
import sys

from agents.policies.dqn import device_for
from training.reinforcement.ppo import ppo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Tetris PPO agent.")
    parser.add_argument("num_games", type=int, help="number of new episodes")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="how far GAE carries future TD residuals backward (default: .95)",
    )
    parser.add_argument(
        "--clip-range",
        type=float,
        default=0.2,
        help="maximum helpful change in the PPO probability ratio (default: .2)",
    )
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument(
        "--episodes-per-update",
        type=int,
        default=16,
        help="fresh episodes collected before each PPO update (default: 16)",
    )
    parser.add_argument("--value-loss-coefficient", type=float, default=0.5)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--reward-scale", type=float, default=1_200.0)
    parser.add_argument(
        "--terminal-penalty",
        type=float,
        default=-1_000.0,
        help="raw training reward added on game-over (default: -1000)",
    )
    parser.add_argument(
        "--heuristic-top-k",
        type=int,
        nargs="?",
        const=8,
        help=(
            "restrict the actor to the top K heuristic placements "
            "(passing the flag without K uses 8)"
        ),
    )
    parser.add_argument("--gradient-clip", type=float, default=0.5)
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
        help="save the training graph every N PPO updates when headless (default: 1)",
    )
    parser.add_argument("--resume-from", help="a previous latest.pt checkpoint")
    parser.add_argument(
        "--display",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="show the live nine-panel PPO/GAE plot (default: on)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        selected_device = device_for(args.device)
        print(f"Training device: {selected_device}")
        model_path = ppo(
            args.num_games,
            learning_rate=args.learning_rate,
            discount=args.discount,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            update_epochs=args.update_epochs,
            minibatch_size=args.minibatch_size,
            episodes_per_update=args.episodes_per_update,
            value_loss_coefficient=args.value_loss_coefficient,
            entropy_coefficient=args.entropy_coefficient,
            reward_scale=args.reward_scale,
            terminal_penalty=args.terminal_penalty,
            heuristic_top_k=args.heuristic_top_k,
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
        print(f"PPO training failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    print(f"Saved latest playable model to {model_path}")
    best_model_path = model_path.with_name("best_model.pt")
    if best_model_path.exists():
        print(f"Best fixed-seed evaluation model: {best_model_path}")


if __name__ == "__main__":
    main()
