import argparse
import curses
import random

from agents.player import Placer, Player, Policy
from core.engine import TICK


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Tetris player.")
    parser.add_argument("--policy", choices=["random", "heuristic"], default="random")
    parser.add_argument("--placer", choices=["dfs"], default="dfs")
    parser.add_argument(
        "--display",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--interactive",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="allow keyboard moves alongside the automated player",
    )
    parser.add_argument("--tick-speed", type=float, default=TICK)
    parser.add_argument("--max-ticks", type=int)
    parser.add_argument("--filepath", help="save board states to this .trs file")
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def run(args: argparse.Namespace, stdscr=None) -> None:
    player = Player(
        policy=Policy[args.policy.upper()],
        placer=Placer[args.placer.upper()],
        display=args.display,
        interactive=args.interactive,
        stdscr=stdscr,
        filepath=args.filepath,
        tick_speed=args.tick_speed,
    )
    player.start(args.max_ticks)


def main() -> None:
    args = parse_args()
    if args.max_ticks is not None and args.max_ticks < 0:
        raise SystemExit("--max-ticks must be zero or greater")
    if args.seed is not None:
        random.seed(args.seed)

    if args.display:
        curses.wrapper(lambda stdscr: run(args, stdscr))
    else:
        run(args)


if __name__ == "__main__":
    main()
