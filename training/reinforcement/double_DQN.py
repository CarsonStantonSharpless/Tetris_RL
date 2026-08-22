"""Replay-based Double DQN training over the existing placement player.

Each action is a legal final placement, not an individual left/right/rotate
command.  The DQN therefore scores an *afterstate*: the board after a
candidate placement has locked and cleared rows, paired with the next piece.
This matches :class:`agents.player.Player`'s fast placement interface while
keeping all variable-size action handling in this training module.
"""

from __future__ import annotations

from collections import deque
import copy
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from agents.player import Player
from agents.policies.dqn import (
    DQN,
    clear_dqn_model_cache,
    device_for,
    piece_id,
    placement_boards,
    require_torch,
)
from agents.policies.heuristic import top_heuristic_placements
from storage.dqn import (
    read_dqn_checkpoint,
    write_dqn_checkpoint,
    write_dqn_model,
    write_dqn_model_state,
)
from training.evaluation.episode import evaluate
from training.evaluation.specs import DQNPolicySpec
from visualization.training import DQNTrainingPlot


@dataclass(frozen=True)
class Transition:
    """One placement decision stored in replay memory."""

    board: np.ndarray
    current_piece: int
    next_piece: int
    level: int
    reward: float
    next_boards: np.ndarray
    next_current_piece: int
    following_piece: int
    next_level: int
    done: bool


class ReplayBuffer:
    """A bounded collection of immutable placement transitions."""

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("replay capacity must be at least one")
        self.capacity = capacity
        self.transitions: deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.transitions)

    def append(self, transition: Transition) -> None:
        self.transitions.append(_frozen_transition(transition))

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator,
    ) -> list[Transition]:
        if batch_size > len(self):
            raise ValueError("cannot sample more transitions than are stored")
        indices = rng.choice(len(self), size=batch_size, replace=False)
        values = list(self.transitions)
        return [values[int(index)] for index in indices]

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "transitions": list(self.transitions),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> ReplayBuffer:
        buffer = cls(int(state["capacity"]))
        for transition in state["transitions"]:
            if not isinstance(transition, Transition):
                raise ValueError("replay checkpoint contains an invalid transition")
            buffer.append(transition)
        return buffer


@dataclass(frozen=True)
class DQNConfig:
    """The hyperparameters that define a Double DQN run."""

    learning_rate: float = 3e-4
    discount: float = 0.99
    replay_size: int = 10_000
    batch_size: int = 32
    warmup_steps: int = 1_000
    train_every: int = 1
    target_update: int = 1_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50_000
    reward_scale: float = 1_200.0
    terminal_penalty: float = -1_000.0
    heuristic_top_k: int | None = None
    gradient_clip: float = 10.0


class DQNAgent:
    """Online/target networks and the Double DQN optimization step."""

    def __init__(self, config: DQNConfig, seed: int, device: Any) -> None:
        torch = require_torch()
        self.config = config
        self.device = device
        torch.manual_seed(seed)
        self.online_model = DQN().to(device)
        self.target_model = DQN().to(device)
        self.target_model.load_state_dict(self.online_model.state_dict())
        self.target_model.eval()
        self.optimizer = torch.optim.Adam(
            self.online_model.parameters(),
            lr=config.learning_rate,
        )
        self.replay_buffer = ReplayBuffer(config.replay_size)
        self.rng = np.random.default_rng(seed)
        self.environment_steps = 0
        self.update_steps = 0
        self.episodes = 0
        self.best_evaluation = float("-inf")
        self.best_model_state: dict[str, Any] | None = None

    def epsilon(self) -> float:
        """Linearly anneal exploration as placement experience grows."""
        fraction = min(self.environment_steps / self.config.epsilon_decay_steps, 1.0)
        return self.config.epsilon_start + fraction * (
            self.config.epsilon_end - self.config.epsilon_start
        )

    def choose_action(
        self,
        boards: np.ndarray,
        current_piece: int,
        next_piece: int,
        level: int,
        epsilon: float,
    ) -> tuple[int, float]:
        """Return an epsilon-greedy legal placement index and its online value."""
        torch = require_torch()
        self.online_model.eval()
        with torch.no_grad():
            board_tensor = torch.from_numpy(boards).to(self.device)
            piece_tensor = torch.full(
                (len(boards),),
                current_piece,
                dtype=torch.long,
                device=self.device,
            )
            next_piece_tensor = torch.full(
                (len(boards),),
                next_piece,
                dtype=torch.long,
                device=self.device,
            )
            level_tensor = torch.full(
                (len(boards),),
                level,
                dtype=torch.float32,
                device=self.device,
            )
            values = self.online_model(
                board_tensor,
                piece_tensor,
                next_piece_tensor,
                level_tensor,
            )

        greedy_choice = int(torch.argmax(values).item())
        value = float(values[greedy_choice].item())
        if self.rng.random() < epsilon:
            return int(self.rng.integers(len(boards))), value
        return greedy_choice, value

    def update(self) -> tuple[float, float] | None:
        """Sample replay memory and perform one Double DQN optimizer update."""
        if len(self.replay_buffer) < self.config.batch_size:
            return None

        torch = require_torch()
        batch = self.replay_buffer.sample(self.config.batch_size, self.rng)
        boards = torch.from_numpy(np.stack([item.board for item in batch])).to(self.device)
        pieces = torch.tensor(
            [item.current_piece for item in batch],
            dtype=torch.long,
            device=self.device,
        )
        next_pieces = torch.tensor(
            [item.next_piece for item in batch],
            dtype=torch.long,
            device=self.device,
        )
        levels = torch.tensor(
            [item.level for item in batch],
            dtype=torch.float32,
            device=self.device,
        )
        rewards = torch.tensor(
            [item.reward for item in batch],
            dtype=torch.float32,
            device=self.device,
        )
        dones = torch.tensor(
            [item.done for item in batch],
            dtype=torch.bool,
            device=self.device,
        )

        self.online_model.train()
        current_values = self.online_model(boards, pieces, next_pieces, levels)
        targets = rewards.clone()
        active_indices = torch.nonzero(~dones, as_tuple=False).squeeze(1)
        if len(active_indices):
            (
                next_boards,
                next_current_pieces,
                following_pieces,
                next_levels,
                offsets,
            ) = _next_action_batch(
                batch,
                active_indices.tolist(),
            )
            next_board_tensor = torch.from_numpy(next_boards).to(self.device)
            next_current_piece_tensor = torch.from_numpy(next_current_pieces).to(self.device)
            following_piece_tensor = torch.from_numpy(following_pieces).to(self.device)
            next_level_tensor = torch.from_numpy(next_levels).to(self.device)
            with torch.no_grad():
                # Double DQN: online network selects, target network evaluates.
                online_next_values = self.online_model(
                    next_board_tensor,
                    next_current_piece_tensor,
                    following_piece_tensor,
                    next_level_tensor,
                )
                selected_actions = _segment_argmax(online_next_values, offsets)
                target_next_values = self.target_model(
                    next_board_tensor,
                    next_current_piece_tensor,
                    following_piece_tensor,
                    next_level_tensor,
                )
                bootstrap_values = target_next_values[selected_actions]
                targets[active_indices] += self.config.discount * bootstrap_values

        loss = torch.nn.functional.smooth_l1_loss(current_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.online_model.parameters(),
            self.config.gradient_clip,
        )
        self.optimizer.step()
        self.update_steps += 1
        if self.update_steps % self.config.target_update == 0:
            self.target_model.load_state_dict(self.online_model.state_dict())

        return float(loss.item()), float(current_values.detach().mean().item())

    def record_evaluation(self, score: float) -> bool:
        """Capture the online network when fixed-seed evaluation improves."""
        if score <= self.best_evaluation:
            return False
        self.best_evaluation = score
        self.best_model_state = copy.deepcopy(self.online_model.state_dict())
        return True

    def checkpoint(self) -> dict[str, Any]:
        """Collect all mutable state needed to resume training."""
        return {
            "config": asdict(self.config),
            "model_state": self.online_model.state_dict(),
            "target_model_state": self.target_model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "episode": self.episodes,
            "environment_steps": self.environment_steps,
            "update_steps": self.update_steps,
            "replay_buffer": self.replay_buffer.state_dict(),
            "agent_rng_state": self.rng.bit_generator.state,
            "best_evaluation": self.best_evaluation,
            "best_model_state": self.best_model_state,
        }

    def restore(self, checkpoint: dict[str, Any]) -> None:
        """Restore an agent from :func:`storage.dqn.read_dqn_checkpoint`."""
        self.online_model.load_state_dict(checkpoint["model_state"])
        self.target_model.load_state_dict(checkpoint["target_model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.replay_buffer = ReplayBuffer.from_state_dict(checkpoint["replay_buffer"])
        self.rng.bit_generator.state = checkpoint["agent_rng_state"]
        self.episodes = int(checkpoint["episode"])
        self.environment_steps = int(checkpoint["environment_steps"])
        self.update_steps = int(checkpoint["update_steps"])
        self.best_evaluation = float(checkpoint.get("best_evaluation", "-inf"))
        self.best_model_state = checkpoint.get("best_model_state")
        if self.best_model_state is None:
            # Older checkpoints did not retain the best evaluation snapshot.
            # Preserve their current learned policy until the next evaluation.
            self.best_model_state = copy.deepcopy(self.online_model.state_dict())


def double_dqn(
    num_games: int,
    *,
    learning_rate: float = 3e-4,
    discount: float = 0.99,
    replay_size: int = 10_000,
    batch_size: int = 32,
    warmup_steps: int = 1_000,
    train_every: int = 1,
    target_update: int = 1_000,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay_steps: int = 50_000,
    reward_scale: float = 1_200.0,
    terminal_penalty: float = -1_000.0,
    heuristic_top_k: int | None = None,
    gradient_clip: float = 10.0,
    max_placements: int = 1_000,
    seed: int = 0,
    run_directory: str | Path | None = None,
    display: bool = True,
    device: str = "auto",
    checkpoint_every: int = 100,
    evaluation_games: int = 10,
    plot_every: int = 1,
    resume_from: str | Path | None = None,
) -> Path:
    """Train a Double DQN for ``num_games`` new Tetris episodes.

    ``model.pt`` is the latest playable network, while ``best_model.pt`` is
    the highest fixed-seed evaluation snapshot. ``latest.pt`` and numbered
    files under ``checkpoints/`` additionally save optimizer and replay state
    and can be passed to ``resume_from``. Resuming keeps the checkpoint's
    training hyperparameters and best-model record.
    """
    _validate_arguments(
        num_games,
        max_placements,
        checkpoint_every,
        evaluation_games,
        plot_every,
    )
    require_torch()
    config = DQNConfig(
        learning_rate=learning_rate,
        discount=discount,
        replay_size=replay_size,
        batch_size=batch_size,
        warmup_steps=warmup_steps,
        train_every=train_every,
        target_update=target_update,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay_steps=epsilon_decay_steps,
        reward_scale=reward_scale,
        terminal_penalty=terminal_penalty,
        heuristic_top_k=heuristic_top_k,
        gradient_clip=gradient_clip,
    )
    model_device = device_for(device)
    run_path = Path(run_directory or _run_directory())
    if resume_from is not None:
        checkpoint = read_dqn_checkpoint(resume_from, model_device)
        try:
            config_data = dict(checkpoint["config"])
            # Reward-shaped heuristic checkpoints predate top-k filtering.
            config_data.pop("heuristic_feedback", None)
            config_data.pop("heuristic_feedback_scale", None)
            config = DQNConfig(**config_data)
        except (KeyError, TypeError) as error:
            raise ValueError("DQN checkpoint has an invalid training config") from error
    _validate_config(config)
    agent = DQNAgent(config, seed, model_device)
    if resume_from is not None:
        agent.restore(checkpoint)

    plot = DQNTrainingPlot(
        run_path / "training.png",
        display,
        save_every=plot_every,
    )
    model_path = run_path / "model.pt"
    best_model_path = run_path / "best_model.pt"
    latest_path = run_path / "latest.pt"
    latest_evaluation: float | None = None
    if agent.best_model_state is not None:
        write_dqn_model_state(agent.best_model_state, best_model_path)

    for _ in range(num_games):
        episode_seed = seed + agent.episodes
        score, losses, q_values = _play_training_episode(
            agent,
            episode_seed,
            max_placements,
        )
        agent.episodes += 1

        if agent.episodes % checkpoint_every == 0:
            write_dqn_model(agent.online_model, model_path)
            if evaluation_games:
                latest_evaluation = _evaluate_model(
                    model_path,
                    seed + 1_000_000,
                    evaluation_games,
                    max_placements,
                    agent.config.heuristic_top_k,
                )
                if agent.record_evaluation(latest_evaluation):
                    write_dqn_model_state(agent.best_model_state, best_model_path)
            _save_checkpoint(agent, latest_path)
            checkpoint_path = run_path / "checkpoints" / (
                f"episode_{agent.episodes:07d}.pt"
            )
            _save_checkpoint(agent, checkpoint_path)

        plot.update({
            "loss": _mean_or_none(losses),
            "score": score,
            "epsilon": agent.epsilon(),
            "mean_q": _mean_or_none(q_values),
            "replay_size": len(agent.replay_buffer),
            "evaluation_score": latest_evaluation,
        })

    write_dqn_model(agent.online_model, model_path)
    if evaluation_games:
        latest_evaluation = _evaluate_model(
            model_path,
            seed + 1_000_000,
            evaluation_games,
            max_placements,
            agent.config.heuristic_top_k,
        )
        if agent.record_evaluation(latest_evaluation):
            write_dqn_model_state(agent.best_model_state, best_model_path)
        plot.update({
            "loss": None,
            "score": None,
            "epsilon": agent.epsilon(),
            "mean_q": None,
            "replay_size": len(agent.replay_buffer),
            "evaluation_score": latest_evaluation,
        })
    _save_checkpoint(agent, latest_path)
    plot.save()
    return model_path


def _play_training_episode(
    agent: DQNAgent,
    seed: int,
    max_placements: int,
) -> tuple[int, list[float], list[float]]:
    """Collect one replay episode through the existing headless Player."""
    player = Player(
        display=False,
        instant_placement=True,
        seed=seed,
    )
    losses: list[float] = []
    q_values: list[float] = []

    while not player.engine.is_game_over and player.engine.ticks < max_placements:
        placements, boards, current_piece, next_piece, level = _legal_afterstates(
            player,
            agent.config.heuristic_top_k,
        )
        choice, q_value = agent.choose_action(
            boards,
            current_piece,
            next_piece,
            level,
            agent.epsilon(),
        )
        result = player.place(placements[choice])
        agent.environment_steps += 1
        q_values.append(q_value)

        if result.done:
            next_boards = np.empty((0, *boards.shape[1:]), dtype=np.float32)
            next_current_piece = 0
            following_piece = 0
            next_level = 0
        else:
            (
                _,
                next_boards,
                next_current_piece,
                following_piece,
                next_level,
            ) = _legal_afterstates(player, agent.config.heuristic_top_k)

        training_reward = _training_reward(
            result.reward,
            result.done,
            agent.config.terminal_penalty,
        )

        agent.replay_buffer.append(Transition(
            board=boards[choice],
            current_piece=current_piece,
            next_piece=next_piece,
            level=level,
            reward=training_reward / agent.config.reward_scale,
            next_boards=next_boards,
            next_current_piece=next_current_piece,
            following_piece=following_piece,
            next_level=next_level,
            done=result.done,
        ))

        if (
            agent.environment_steps >= agent.config.warmup_steps
            and agent.environment_steps % agent.config.train_every == 0
        ):
            update = agent.update()
            if update is not None:
                loss, _ = update
                losses.append(loss)

    return player.state.score, losses, q_values


def _legal_afterstates(
    player: Player,
    heuristic_top_k: int | None = None,
) -> tuple[list[Any], np.ndarray, int, int, int]:
    """Return legal placements encoded for the afterstate-value DQN."""
    placements = player.possible_placements()
    placements = top_heuristic_placements(placements, heuristic_top_k)
    boards = placement_boards(placements)
    return (
        placements,
        boards,
        piece_id(player.state.board_state.curr_piece.kind),
        piece_id(player.state.next_piece),
        player.engine.level,
    )


def _next_action_batch(
    batch: list[Transition],
    active_indices: Iterable[int],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[tuple[int, int]],
]:
    """Concatenate variable-length legal next-action sets for one network call."""
    boards: list[np.ndarray] = []
    current_pieces: list[np.ndarray] = []
    following_pieces: list[np.ndarray] = []
    levels: list[np.ndarray] = []
    offsets: list[tuple[int, int]] = []
    start = 0
    for index in active_indices:
        transition = batch[index]
        count = len(transition.next_boards)
        if count == 0:
            raise ValueError("non-terminal transitions must have next placements")
        boards.append(transition.next_boards)
        current_pieces.append(
            np.full(count, transition.next_current_piece, dtype=np.int64)
        )
        following_pieces.append(
            np.full(count, transition.following_piece, dtype=np.int64)
        )
        levels.append(np.full(count, transition.next_level, dtype=np.float32))
        offsets.append((start, start + count))
        start += count
    return (
        np.concatenate(boards),
        np.concatenate(current_pieces),
        np.concatenate(following_pieces),
        np.concatenate(levels),
        offsets,
    )


def _segment_argmax(values: Any, offsets: list[tuple[int, int]]) -> Any:
    """Return flattened indices of the best value in each action segment."""
    torch = require_torch()
    return torch.stack([
        start + torch.argmax(values[start:end])
        for start, end in offsets
    ])


def _training_reward(
    score_reward: int,
    done: bool,
    terminal_penalty: float,
) -> float:
    """Add the configured training penalty when a placement ends the game."""
    return score_reward + terminal_penalty if done else float(score_reward)


def _frozen_transition(transition: Transition) -> Transition:
    """Copy arrays so replay cannot observe later engine mutations."""
    # One byte per occupancy cell keeps replay memory feasible even though a
    # transition carries every legal next-placement afterstate.
    board = np.array(transition.board, dtype=np.uint8, copy=True)
    next_boards = np.array(transition.next_boards, dtype=np.uint8, copy=True)
    board.setflags(write=False)
    next_boards.setflags(write=False)
    return Transition(
        board=board,
        current_piece=int(transition.current_piece),
        next_piece=int(transition.next_piece),
        level=int(transition.level),
        reward=float(transition.reward),
        next_boards=next_boards,
        next_current_piece=int(transition.next_current_piece),
        following_piece=int(transition.following_piece),
        next_level=int(transition.next_level),
        done=bool(transition.done),
    )


def _save_checkpoint(agent: DQNAgent, filepath: Path) -> None:
    write_dqn_checkpoint(agent.checkpoint(), filepath)


def _evaluate_model(
    filepath: Path,
    seed: int,
    games: int,
    max_placements: int,
    heuristic_top_k: int | None,
) -> float:
    """Evaluate through the normal serial policy-spec pathway."""
    clear_dqn_model_cache()
    results = evaluate(
        DQNPolicySpec(str(filepath), heuristic_top_k=heuristic_top_k),
        range(seed, seed + games),
        max_placements,
    )
    return float(np.mean([result.score for result in results]))


def _mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _run_directory() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("runs") / f"double_dqn_{timestamp}"


def _validate_arguments(
    num_games: int,
    max_placements: int,
    checkpoint_every: int,
    evaluation_games: int,
    plot_every: int,
) -> None:
    if num_games < 1:
        raise ValueError("num_games must be at least one")
    if max_placements < 1:
        raise ValueError("max_placements must be at least one")
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be at least one")
    if evaluation_games < 0:
        raise ValueError("evaluation_games cannot be negative")
    if plot_every < 1:
        raise ValueError("plot_every must be at least one")


def _validate_config(config: DQNConfig) -> None:
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be greater than zero")
    if not 0 <= config.discount <= 1:
        raise ValueError("discount must be between zero and one")
    if config.batch_size < 1 or config.warmup_steps < 0:
        raise ValueError("batch_size must be positive and warmup_steps non-negative")
    if config.train_every < 1 or config.target_update < 1:
        raise ValueError("train_every and target_update must be at least one")
    if not 0 <= config.epsilon_end <= config.epsilon_start <= 1:
        raise ValueError("epsilon values must satisfy 0 <= end <= start <= 1")
    if config.epsilon_decay_steps < 1:
        raise ValueError("epsilon_decay_steps must be at least one")
    if config.reward_scale <= 0 or config.gradient_clip <= 0:
        raise ValueError("reward_scale and gradient_clip must be greater than zero")
    if config.terminal_penalty > 0:
        raise ValueError("terminal_penalty cannot be positive")
    if config.heuristic_top_k is not None and config.heuristic_top_k < 1:
        raise ValueError("heuristic_top_k must be at least one")
