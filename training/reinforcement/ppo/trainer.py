"""A deliberately direct PPO training loop for the Tetris placement player.

There are three boundaries worth keeping visible while reading this file:

1. ``Player`` is the environment.  It emits a state, legal placements, a
   score reward, the resulting next state, and a game-over flag.
2. ``PPO`` is the model.  Its actor emits action probabilities and its critic
   emits ``V(s)``.  It never changes the board or invents rewards.
3. ``generalized_advantage_estimate`` combines environment rewards with
   critic values.  It emits advantages for the actor and returns for the
   critic.

PPO is on-policy, so a fixed-size batch of fresh transitions is used for one
learning phase and then discarded. There is no replay buffer or target network.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from agents.player import Player
from agents.policies.dqn import device_for, piece_id, placement_boards, require_torch
from agents.policies.heuristic import top_heuristic_placements
from agents.policies.ppo import PPO, clear_ppo_model_cache
from storage.ppo import (
    read_ppo_checkpoint,
    write_ppo_checkpoint,
    write_ppo_model,
    write_ppo_model_state,
)
from training.evaluation.episode import evaluate
from training.evaluation.specs import PPOPolicySpec
from training.reinforcement.ppo.gae import generalized_advantage_estimate
from visualization.training import PPOTrainingPlot


@dataclass(frozen=True)
class PPOConfig:
    """The small set of hyperparameters used by the PPO equations."""

    learning_rate: float = 3e-4
    discount: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    update_epochs: int = 4
    minibatch_size: int = 64
    rollout_steps: int = 1_024
    value_loss_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    reward_scale: float = 1_200.0
    terminal_penalty: float = -1_000.0
    bottom_up_bias: float = 0.0
    heuristic_top_k: int | None = None
    gradient_clip: float = 0.5


@dataclass(frozen=True)
class PlacementObservation:
    """Everything emitted by the environment before one placement.

    ``state_board`` is the current board used by the critic.  ``action_boards``
    contains one simulated afterstate per legal placement and is used by the
    actor.  The model sees arrays and context ids; only the environment sees
    the concrete ``placements`` that can be applied to the game.
    """

    placements: list[Any]
    state_board: np.ndarray
    action_boards: np.ndarray
    current_piece: int
    next_piece: int
    level: int


@dataclass(frozen=True)
class PolicyDecision:
    """The actor and critic outputs recorded before the environment acts."""

    action: int
    log_probability: float
    value: float


@dataclass(frozen=True)
class RolloutStep:
    """One on-policy state/action/result tuple kept until the PPO update."""

    observation: PlacementObservation
    action: int
    old_log_probability: float
    old_value: float
    reward: float
    done: bool


@dataclass(frozen=True)
class RolloutSegment:
    """Contiguous steps from one game plus the value immediately afterward.

    A segment ends at game-over, the placement cap, or the fixed PPO rollout
    boundary. A non-terminal segment uses its bootstrap value for GAE.
    """

    steps: list[RolloutStep]
    bootstrap_value: float


@dataclass(frozen=True)
class PPOMetrics:
    """Values shown in the PPO training visualization."""

    loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    clip_fraction: float
    mean_absolute_advantage: float
    mean_absolute_td_residual: float


class PPOAgent:
    """Own the actor-critic model and explain one clipped PPO update."""

    def __init__(self, config: PPOConfig, seed: int, device: Any) -> None:
        torch = require_torch()
        self.config = config
        self.device = device
        torch.manual_seed(seed)
        self.model = PPO().to(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
        )
        self.rng = np.random.default_rng(seed)
        self.environment_steps = 0
        self.update_steps = 0
        self.episodes = 0
        self.best_evaluation = float("-inf")
        self.best_model_state: dict[str, Any] | None = None

    def choose_action(self, observation: PlacementObservation) -> PolicyDecision:
        """Sample ``a`` from the old policy and record ``log P_old(a | s)``."""
        torch = require_torch()
        self.model.eval()
        with torch.no_grad():
            # The actor emits one logit for every action the environment says
            # is legal in this particular state.
            logits = self._policy_logits(observation)
            # A categorical distribution turns those logits into P(a | s).
            distribution = torch.distributions.Categorical(logits=logits)
            # NumPy performs the sample so its saved RNG makes resume repeatable.
            probabilities = distribution.probs.detach().cpu().numpy()
            action = int(self.rng.choice(len(probabilities), p=probabilities))
            action_tensor = torch.tensor(action, device=self.device)
            # PPO must retain the probability from collection time as P_old.
            old_log_probability = distribution.log_prob(action_tensor)
            # The critic separately predicts the value of the state before a.
            value = self._state_value(observation)
        return PolicyDecision(
            action=action,
            log_probability=float(old_log_probability.item()),
            value=float(value.item()),
        )

    def value_of(self, observation: PlacementObservation) -> float:
        """Ask only the critic for ``V(s)`` when bootstrapping a truncation."""
        torch = require_torch()
        self.model.eval()
        with torch.no_grad():
            return float(self._state_value(observation).item())

    def learn(self, segments: list[RolloutSegment]) -> PPOMetrics:
        """Turn one fixed transition rollout into GAE targets and updates."""
        torch = require_torch()

        if not segments:
            raise ValueError("PPO learning requires at least one rollout segment")

        steps: list[RolloutStep] = []
        segment_advantages: list[np.ndarray] = []
        segment_returns: list[np.ndarray] = []
        segment_td_residuals: list[np.ndarray] = []

        # GAE must restart at every terminal or fixed-rollout boundary. A segment
        # cut mid-game bootstraps V(s_next), so its unfinished future is not lost.
        for segment in segments:
            # The environment supplied this segment's rewards and done flags.
            rewards = np.array(
                [step.reward for step in segment.steps],
                dtype=np.float32,
            )
            dones = np.array(
                [step.done for step in segment.steps],
                dtype=np.bool_,
            )
            # The unchanged old critic supplied V(s) while collecting the rollout.
            old_values = np.array(
                [step.old_value for step in segment.steps],
                dtype=np.float32,
            )
            # GAE emits TD residuals, actor advantages, and critic return targets
            # for this uninterrupted segment only.
            gae = generalized_advantage_estimate(
                rewards,
                old_values,
                dones,
                segment.bootstrap_value,
                self.config.discount,
                self.config.gae_lambda,
            )
            # Keep the transitions and GAE outputs in matching concatenated order.
            steps.extend(segment.steps)
            segment_advantages.append(gae.advantages)
            segment_returns.append(gae.returns)
            segment_td_residuals.append(gae.td_residuals)

        # The optimizer sees one consistently sized batch of fresh transitions.
        raw_advantages = np.concatenate(segment_advantages)
        critic_returns = np.concatenate(segment_returns)
        td_residuals = np.concatenate(segment_td_residuals)
        # Normalize across the complete batch so one unusual game dominates less.
        advantages = _normalized_advantages(raw_advantages)

        losses: list[float] = []
        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []
        clip_fractions: list[float] = []
        indices = np.arange(len(steps))

        # PPO reuses this one fresh on-policy rollout for a few learning passes.
        for _ in range(self.config.update_epochs):
            # Shuffling removes any special meaning from trajectory order.
            self.rng.shuffle(indices)
            # Each slice is small enough to form one optimizer update.
            for start in range(0, len(indices), self.config.minibatch_size):
                batch_indices = indices[start:start + self.config.minibatch_size]
                batch_steps = [steps[int(index)] for index in batch_indices]

                # Re-evaluate the recorded action with the changing new policy.
                new_log_probabilities, new_values, entropy = (
                    self._evaluate_actions(batch_steps)
                )
                # These log probabilities were frozen when the rollout was made.
                old_log_probabilities = torch.tensor(
                    [step.old_log_probability for step in batch_steps],
                    dtype=torch.float32,
                    device=self.device,
                )
                # These are GAE's normalized answers to "was action a better
                # or worse than the critic expected in state s?"
                batch_advantages = torch.from_numpy(advantages[batch_indices]).to(
                    self.device
                )
                # These are GAE's V(s) + advantage targets for the critic.
                batch_returns = torch.from_numpy(critic_returns[batch_indices]).to(
                    self.device
                )

                # Given action a and state s, the ratio is exactly
                # P_new(a | s) / P_old(a | s).  Subtracting log probabilities
                # and exponentiating computes that division stably.
                probability_ratio = torch.exp(
                    new_log_probabilities - old_log_probabilities
                )
                # The ordinary policy-gradient objective trusts the full ratio.
                ordinary_objective = probability_ratio * batch_advantages
                # PPO's second objective limits how far that ratio may help.
                clipped_ratio = torch.clamp(
                    probability_ratio,
                    1.0 - self.config.clip_range,
                    1.0 + self.config.clip_range,
                )
                clipped_objective = clipped_ratio * batch_advantages
                # Taking the smaller objective is PPO's conservative promise.
                policy_loss = -torch.min(
                    ordinary_objective,
                    clipped_objective,
                ).mean()
                # The critic learns to reproduce the return targets from GAE.
                value_loss = torch.nn.functional.mse_loss(
                    new_values,
                    batch_returns,
                )
                # Entropy rewards a broad policy so exploration disappears slowly.
                mean_entropy = entropy.mean()
                # One shared loss trains the actor, critic, and shared encoder.
                loss = (
                    policy_loss
                    + self.config.value_loss_coefficient * value_loss
                    - self.config.entropy_coefficient * mean_entropy
                )

                # Clear gradients left by the preceding minibatch.
                self.optimizer.zero_grad()
                # Differentiate the readable PPO objective above.
                loss.backward()
                # Prevent one surprising trajectory from making a huge update.
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip,
                )
                # Apply this minibatch's actor and critic gradients.
                self.optimizer.step()
                # Count optimizer updates for checkpoint inspection.
                self.update_steps += 1

                # The clip fraction shows how often PPO rejected a large ratio.
                was_clipped = (
                    torch.abs(probability_ratio.detach() - 1.0)
                    > self.config.clip_range
                )
                losses.append(float(loss.detach().item()))
                policy_losses.append(float(policy_loss.detach().item()))
                value_losses.append(float(value_loss.detach().item()))
                entropies.append(float(mean_entropy.detach().item()))
                clip_fractions.append(float(was_clipped.float().mean().item()))

        return PPOMetrics(
            loss=float(np.mean(losses)),
            policy_loss=float(np.mean(policy_losses)),
            value_loss=float(np.mean(value_losses)),
            entropy=float(np.mean(entropies)),
            clip_fraction=float(np.mean(clip_fractions)),
            mean_absolute_advantage=float(np.mean(np.abs(raw_advantages))),
            mean_absolute_td_residual=float(np.mean(np.abs(td_residuals))),
        )

    def _policy_logits(self, observation: PlacementObservation) -> Any:
        """Run the actor on every legal afterstate in one observation."""
        torch = require_torch()
        count = len(observation.action_boards)
        return self.model.policy_logits(
            torch.from_numpy(observation.action_boards).to(self.device),
            torch.full(
                (count,),
                observation.current_piece,
                dtype=torch.long,
                device=self.device,
            ),
            torch.full(
                (count,),
                observation.next_piece,
                dtype=torch.long,
                device=self.device,
            ),
            torch.full(
                (count,),
                observation.level,
                dtype=torch.float32,
                device=self.device,
            ),
        )

    def _state_value(self, observation: PlacementObservation) -> Any:
        """Run the critic on the single current state in an observation."""
        torch = require_torch()
        return self.model.state_values(
            torch.from_numpy(observation.state_board).unsqueeze(0).to(self.device),
            torch.tensor(
                [observation.current_piece],
                dtype=torch.long,
                device=self.device,
            ),
            torch.tensor(
                [observation.next_piece],
                dtype=torch.long,
                device=self.device,
            ),
            torch.tensor(
                [observation.level],
                dtype=torch.float32,
                device=self.device,
            ),
        )[0]

    def _evaluate_actions(
        self,
        steps: list[RolloutStep],
    ) -> tuple[Any, Any, Any]:
        """Get new actor probabilities and critic values for a minibatch.

        Action sets are ragged, so the most readable implementation evaluates
        each categorical action set separately.  Current-state critic inputs
        have a fixed shape and are evaluated together.
        """
        torch = require_torch()
        self.model.train()
        log_probabilities = []
        entropies = []
        for step in steps:
            logits = self._policy_logits(step.observation)
            distribution = torch.distributions.Categorical(logits=logits)
            action = torch.tensor(step.action, device=self.device)
            log_probabilities.append(distribution.log_prob(action))
            entropies.append(distribution.entropy())

        observations = [step.observation for step in steps]
        state_boards = torch.from_numpy(np.stack([
            observation.state_board for observation in observations
        ])).to(self.device)
        current_pieces = torch.tensor(
            [observation.current_piece for observation in observations],
            dtype=torch.long,
            device=self.device,
        )
        next_pieces = torch.tensor(
            [observation.next_piece for observation in observations],
            dtype=torch.long,
            device=self.device,
        )
        levels = torch.tensor(
            [observation.level for observation in observations],
            dtype=torch.float32,
            device=self.device,
        )
        values = self.model.state_values(
            state_boards,
            current_pieces,
            next_pieces,
            levels,
        )
        return torch.stack(log_probabilities), values, torch.stack(entropies)

    def record_evaluation(self, score: float) -> bool:
        """Capture the model when fixed-seed evaluation improves."""
        if score <= self.best_evaluation:
            return False
        self.best_evaluation = score
        self.best_model_state = copy.deepcopy(self.model.state_dict())
        return True

    def checkpoint(self) -> dict[str, Any]:
        """Collect all mutable state needed to resume training."""
        return {
            "config": asdict(self.config),
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "episode": self.episodes,
            "environment_steps": self.environment_steps,
            "update_steps": self.update_steps,
            "agent_rng_state": self.rng.bit_generator.state,
            "best_evaluation": self.best_evaluation,
            "best_model_state": self.best_model_state,
        }

    def restore(self, checkpoint: dict[str, Any]) -> None:
        """Restore model, optimizer, counters, and sampling state."""
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.episodes = int(checkpoint["episode"])
        self.environment_steps = int(checkpoint["environment_steps"])
        self.update_steps = int(checkpoint["update_steps"])
        self.rng.bit_generator.state = checkpoint["agent_rng_state"]
        self.best_evaluation = float(checkpoint.get("best_evaluation", "-inf"))
        self.best_model_state = checkpoint.get("best_model_state")
        if self.best_model_state is None:
            self.best_model_state = copy.deepcopy(self.model.state_dict())


def ppo(
    num_games: int,
    *,
    learning_rate: float = 3e-4,
    discount: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    update_epochs: int = 4,
    minibatch_size: int = 64,
    rollout_steps: int = 1_024,
    value_loss_coefficient: float = 0.5,
    entropy_coefficient: float = 0.01,
    reward_scale: float = 1_200.0,
    terminal_penalty: float = -1_000.0,
    bottom_up_bias: float = 0.0,
    heuristic_top_k: int | None = None,
    gradient_clip: float = 0.5,
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
    """Train PPO with fixed-size transition rollouts until ``num_games`` end."""
    _validate_arguments(
        num_games,
        max_placements,
        checkpoint_every,
        evaluation_games,
        plot_every,
    )
    require_torch()
    config = PPOConfig(
        learning_rate=learning_rate,
        discount=discount,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        update_epochs=update_epochs,
        minibatch_size=minibatch_size,
        rollout_steps=rollout_steps,
        value_loss_coefficient=value_loss_coefficient,
        entropy_coefficient=entropy_coefficient,
        reward_scale=reward_scale,
        terminal_penalty=terminal_penalty,
        bottom_up_bias=bottom_up_bias,
        heuristic_top_k=heuristic_top_k,
        gradient_clip=gradient_clip,
    )
    model_device = device_for(device)
    run_path = Path(run_directory or _run_directory())
    if resume_from is not None:
        checkpoint = read_ppo_checkpoint(resume_from, model_device)
        try:
            config_data = dict(checkpoint["config"])
            # Older checkpoints used complete episodes as their batch unit.
            config_data.pop("episodes_per_update", None)
            config_data.setdefault("rollout_steps", rollout_steps)
            config_data.setdefault("bottom_up_bias", bottom_up_bias)
            config = PPOConfig(**config_data)
        except (KeyError, TypeError) as error:
            raise ValueError("PPO checkpoint has an invalid training config") from error
    _validate_config(config)
    agent = PPOAgent(config, seed, model_device)
    if resume_from is not None:
        agent.restore(checkpoint)

    plot = PPOTrainingPlot(
        run_path / "training.png",
        display,
        save_every=plot_every,
    )
    model_path = run_path / "model.pt"
    best_model_path = run_path / "best_model.pt"
    latest_path = run_path / "latest.pt"
    latest_evaluation: float | None = None
    if agent.best_model_state is not None:
        write_ppo_model_state(agent.best_model_state, best_model_path)

    target_episodes = agent.episodes + num_games
    # The same environment continues when a fixed rollout cuts through a game.
    player = Player(
        display=False,
        instant_placement=True,
        seed=seed + agent.episodes,
    )

    # One trip through this loop collects fresh transitions, then updates once.
    while agent.episodes < target_episodes:
        # End a rollout at checkpoint boundaries so resume always starts cleanly
        # at the beginning of the following episode.
        next_checkpoint_episode = (
            (agent.episodes // checkpoint_every) + 1
        ) * checkpoint_every
        stop_after_episode = min(target_episodes, next_checkpoint_episode)

        # The actor and critic remain unchanged throughout this collection call.
        segments, scores = _collect_rollout(
            agent,
            player,
            seed,
            max_placements,
            stop_after_episode,
        )
        # GAE handles each segment boundary before all transitions are combined.
        metrics = agent.learn(segments)
        # The complete fixed-step rollout is now discarded: PPO stays on-policy.

        checkpoint_due = agent.episodes >= next_checkpoint_episode
        if checkpoint_due:
            write_ppo_model(agent.model, model_path)
            if evaluation_games:
                latest_evaluation = _evaluate_model(
                    model_path,
                    seed + 1_000_000,
                    evaluation_games,
                    max_placements,
                    agent.config.heuristic_top_k,
                )
                if agent.record_evaluation(latest_evaluation):
                    assert agent.best_model_state is not None
                    write_ppo_model_state(agent.best_model_state, best_model_path)
            _save_checkpoint(agent, latest_path)
            checkpoint_path = run_path / "checkpoints" / (
                f"episode_{agent.episodes:07d}.pt"
            )
            _save_checkpoint(agent, checkpoint_path)

        # Each point makes the actor, critic, GAE, and environment score visible.
        plot.update({
            "loss": metrics.loss,
            "score": float(np.mean(scores)) if scores else None,
            "policy_loss": metrics.policy_loss,
            "value_loss": metrics.value_loss,
            "gae_advantage": metrics.mean_absolute_advantage,
            "td_residual": metrics.mean_absolute_td_residual,
            "entropy": metrics.entropy,
            "clip_fraction": metrics.clip_fraction,
            "evaluation_score": latest_evaluation,
        })

    # Always leave a current playable model and resumable training checkpoint.
    write_ppo_model(agent.model, model_path)
    if evaluation_games:
        latest_evaluation = _evaluate_model(
            model_path,
            seed + 1_000_000,
            evaluation_games,
            max_placements,
            agent.config.heuristic_top_k,
        )
        if agent.record_evaluation(latest_evaluation):
            assert agent.best_model_state is not None
            write_ppo_model_state(agent.best_model_state, best_model_path)
        plot.update({
            "loss": None,
            "score": None,
            "policy_loss": None,
            "value_loss": None,
            "gae_advantage": None,
            "td_residual": None,
            "entropy": None,
            "clip_fraction": None,
            "evaluation_score": latest_evaluation,
        })
    _save_checkpoint(agent, latest_path)
    plot.save()
    return model_path


def _collect_rollout(
    agent: PPOAgent,
    player: Player,
    seed: int,
    max_placements: int,
    stop_after_episode: int,
) -> tuple[list[RolloutSegment], list[int]]:
    """Collect at most ``rollout_steps`` while preserving game boundaries."""
    segments: list[RolloutSegment] = []
    completed_scores: list[int] = []
    current_steps: list[RolloutStep] = []
    collected_steps = 0

    # A PPO update now receives a consistent amount of experience regardless
    # of whether the current policy survives for ten placements or one thousand.
    while (
        collected_steps < agent.config.rollout_steps
        and agent.episodes < stop_after_episode
    ):
        # A prior segment may have ended exactly at the previous rollout boundary.
        if player.engine.is_game_over or player.engine.ticks >= max_placements:
            player.restart(seed=seed + agent.episodes)

        # The environment emits current state data and every legal action.
        observation = _observe_environment(player, agent.config.heuristic_top_k)
        # The model emits a sampled action, P_old(a | s), and V_old(s).
        decision = agent.choose_action(observation)
        # The environment applies that action and emits reward, next state, done.
        result = player.place(observation.placements[decision.action])
        # The optional game-over penalty belongs to training, not the Tetris score.
        training_reward = _training_reward(
            result.reward,
            result.done,
            agent.config.terminal_penalty,
        )
        # This optional potential difference gently favors low, hole-free boards.
        if agent.config.bottom_up_bias:
            next_board = (result.state.board_state.grid != 0).astype(np.uint8)
            training_reward += agent.config.bottom_up_bias * _bottom_up_reward(
                observation.state_board,
                next_board,
                result.done,
                agent.config.discount,
            )
        # Reward scaling keeps critic targets near the network's initial scale.
        scaled_reward = training_reward / agent.config.reward_scale
        # Store only what the PPO and GAE equations need from this fresh step.
        current_steps.append(RolloutStep(
            observation=observation,
            action=decision.action,
            old_log_probability=decision.log_probability,
            old_value=decision.value,
            reward=scaled_reward,
            done=result.done,
        ))
        # Count environment interactions independently from optimizer updates.
        agent.environment_steps += 1
        collected_steps += 1

        # Game-over and the per-game placement cap both close this trajectory.
        episode_finished = result.done or player.engine.ticks >= max_placements
        if episode_finished:
            # Game-over has no future value; a capped game is merely truncated.
            bootstrap_value = 0.0
            if not result.done:
                next_observation = _observe_environment(
                    player,
                    agent.config.heuristic_top_k,
                )
                bootstrap_value = agent.value_of(next_observation)
            segments.append(RolloutSegment(current_steps, bootstrap_value))
            current_steps = []
            completed_scores.append(player.state.score)
            agent.episodes += 1

    # If the fixed transition budget cuts through a game, bootstrap its last
    # state and continue that same environment after this PPO update.
    if current_steps:
        final_observation = _observe_environment(
            player,
            agent.config.heuristic_top_k,
        )
        bootstrap_value = agent.value_of(final_observation)
        segments.append(RolloutSegment(current_steps, bootstrap_value))

    return segments, completed_scores


def _observe_environment(
    player: Player,
    heuristic_top_k: int | None = None,
) -> PlacementObservation:
    """Translate a Player state and its legal placements into model arrays."""
    placements = player.possible_placements()
    placements = top_heuristic_placements(placements, heuristic_top_k)
    board_state = player.state.board_state
    return PlacementObservation(
        placements=placements,
        # The critic sees the board before the selected piece is placed.
        # One-byte occupancy keeps a multi-game rollout reasonably small; the
        # model converts it to floating point immediately before computation.
        state_board=(board_state.grid != 0).astype(np.uint8),
        # The actor sees one board after each legal action locks and clears rows.
        action_boards=placement_boards(placements).astype(np.uint8),
        current_piece=piece_id(board_state.curr_piece.kind),
        next_piece=piece_id(player.state.next_piece),
        level=player.engine.level,
    )


def _normalized_advantages(advantages: np.ndarray) -> np.ndarray:
    """Center and scale a non-trivial advantage batch."""
    if len(advantages) < 2:
        return advantages.copy()
    return (advantages - advantages.mean()) / (advantages.std() + 1e-8)


def _training_reward(
    score_reward: int,
    done: bool,
    terminal_penalty: float,
) -> float:
    """Add the configured training-only penalty when the game ends."""
    return score_reward + terminal_penalty if done else float(score_reward)


def _bottom_up_reward(
    current_board: np.ndarray,
    next_board: np.ndarray,
    done: bool,
    discount: float,
) -> float:
    """Return a small potential difference favoring bottom-up construction.

    This is deliberately measured in raw score-point units. At the CLI's
    flag-only weight of one, an ordinary preference is a few points while even
    a single cleared line is worth forty. Both actor and critic see this reward;
    it only gives GAE a gentle early hint about which boards are promising.
    """
    current_potential = _bottom_up_potential(current_board)
    next_potential = 0.0 if done else _bottom_up_potential(next_board)
    return discount * next_potential - current_potential


def _bottom_up_potential(board: np.ndarray) -> float:
    """Score free headroom and subtract two points for every covered hole."""
    filled = board != 0
    occupied_columns = filled.any(axis=0)
    first_filled_rows = np.argmax(filled, axis=0)
    column_heights = np.where(
        occupied_columns,
        board.shape[0] - first_filled_rows,
        0,
    )
    covered_cells = np.maximum.accumulate(filled, axis=0)
    holes = int(np.count_nonzero(covered_cells & ~filled))
    headroom = board.shape[0] - int(column_heights.max())
    return max(0.0, float(headroom - 2 * holes))


def _save_checkpoint(agent: PPOAgent, filepath: Path) -> None:
    write_ppo_checkpoint(agent.checkpoint(), filepath)


def _evaluate_model(
    filepath: Path,
    seed: int,
    games: int,
    max_placements: int,
    heuristic_top_k: int | None,
) -> float:
    """Evaluate the actor greedily through the normal policy-spec pathway."""
    clear_ppo_model_cache()
    results = evaluate(
        PPOPolicySpec(str(filepath), heuristic_top_k=heuristic_top_k),
        range(seed, seed + games),
        max_placements,
    )
    return float(np.mean([result.score for result in results]))


def _run_directory() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("runs") / f"ppo_{timestamp}"


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


def _validate_config(config: PPOConfig) -> None:
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be greater than zero")
    if not 0 <= config.discount <= 1:
        raise ValueError("discount must be between zero and one")
    if not 0 <= config.gae_lambda <= 1:
        raise ValueError("gae_lambda must be between zero and one")
    if config.clip_range <= 0:
        raise ValueError("clip_range must be greater than zero")
    if config.update_epochs < 1 or config.minibatch_size < 1:
        raise ValueError("update_epochs and minibatch_size must be at least one")
    if config.rollout_steps < 1:
        raise ValueError("rollout_steps must be at least one")
    if config.value_loss_coefficient < 0 or config.entropy_coefficient < 0:
        raise ValueError("loss coefficients cannot be negative")
    if config.reward_scale <= 0 or config.gradient_clip <= 0:
        raise ValueError("reward_scale and gradient_clip must be greater than zero")
    if config.terminal_penalty > 0:
        raise ValueError("terminal_penalty cannot be positive")
    if config.bottom_up_bias < 0:
        raise ValueError("bottom_up_bias cannot be negative")
    if config.heuristic_top_k is not None and config.heuristic_top_k < 1:
        raise ValueError("heuristic_top_k must be at least one")
