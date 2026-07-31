import argparse
import curses
import random

from agents.player import Placer, Player, Policy
from core.engine import TICK


HEURISTIC_PARAMETERS = ("alpha", "beta", "gamma", "delta", "epsilon")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Tetris player.")
    parser.add_argument(
        "--policy",
        choices=["random", "heuristic", "genetic-heuristic"],
        default="random",
    )
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
    parser.add_argument(
        "--instant-placement",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="apply each policy placement in one tick (headless only)",
    )
    parser.add_argument("--filepath", help="save board states to this .trs file")
    parser.add_argument(
        "--params-file",
        help="captain/lieutenant JSON file for the genetic heuristic policy",
    )
    parser.add_argument("--seed", type=int)
    heuristic = parser.add_argument_group("heuristic parameters")
    for parameter in HEURISTIC_PARAMETERS:
        heuristic.add_argument(f"--{parameter}", type=float)
    return parser.parse_args()


def policy_params(args: argparse.Namespace) -> dict[str, float | str]:
    params: dict[str, float | str] = {
        parameter: value
        for parameter in HEURISTIC_PARAMETERS
        if (value := getattr(args, parameter, None)) is not None
    }
    if args.params_file is not None:
        params["filepath"] = args.params_file
    return params


def run(args: argparse.Namespace, stdscr=None) -> None:
    params = policy_params(args)
    if any(parameter in params for parameter in HEURISTIC_PARAMETERS) and (
        args.policy != "heuristic"
    ):
        raise SystemExit("heuristic parameters require --policy heuristic")
    if args.params_file is not None and args.policy != "genetic-heuristic":
        raise SystemExit("--params-file requires --policy genetic-heuristic")

    player = Player(
        policy=Policy[args.policy.replace("-", "_").upper()],
        placer=Placer[args.placer.upper()],
        display=args.display,
        interactive=args.interactive,
        stdscr=stdscr,
        filepath=args.filepath,
        tick_speed=args.tick_speed,
        policy_params=params,
        instant_placement=args.instant_placement,
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
