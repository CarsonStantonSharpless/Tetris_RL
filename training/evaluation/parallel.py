from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
import multiprocessing
import signal

from training.evaluation.episode import EpisodeResult, run_episode
from training.evaluation.specs import PolicySpec


@dataclass(frozen=True)
class PolicyEpisodeResult:
    policy_index: int
    episode: EpisodeResult


@dataclass(frozen=True)
class _EpisodeTask:
    policy_index: int
    policy: PolicySpec
    seed: int
    max_ticks: int


def evaluate_parallel(
    policies: Sequence[PolicySpec],
    seeds: Sequence[int],
    max_ticks: int = 1000,
    workers: int | None = None,
) -> Iterator[PolicyEpisodeResult]:
    """Evaluate every policy/seed pair using episode-level processes."""
    if max_ticks < 1:
        raise ValueError("max_ticks must be at least one")
    if workers is not None and workers < 1:
        raise ValueError("workers must be at least one")

    tasks = _episode_tasks(policies, seeds, max_ticks)
    if workers == 1:
        yield from map(_run_task, tasks)
        return

    task_count = len(policies) * len(seeds)
    if task_count == 0:
        return

    context = multiprocessing.get_context()
    pool = context.Pool(
        processes=workers,
        initializer=_ignore_sigint,
    )
    try:
        yield from pool.imap_unordered(
            _run_task,
            tasks,
            # Episode lengths vary widely. One task per chunk prevents a long
            # game from holding back completed results in the same chunk.
            chunksize=1,
        )
    except BaseException:
        pool.terminate()
        raise
    else:
        pool.close()
    finally:
        pool.join()


def _episode_tasks(
    policies: Sequence[PolicySpec],
    seeds: Sequence[int],
    max_ticks: int,
) -> Iterator[_EpisodeTask]:
    # Seed-major ordering distributes every policy across worker chunks.
    for seed in seeds:
        for policy_index, policy in enumerate(policies):
            yield _EpisodeTask(policy_index, policy, seed, max_ticks)


def _run_task(task: _EpisodeTask) -> PolicyEpisodeResult:
    return PolicyEpisodeResult(
        policy_index=task.policy_index,
        episode=run_episode(task.policy, task.seed, task.max_ticks),
    )


def _ignore_sigint() -> None:
    # The parent process handles Ctrl-C and terminates the whole pool.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
