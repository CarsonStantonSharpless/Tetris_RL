from pathlib import Path
from typing import Mapping


class TrainingPlot:
    """Display and save loss and score curves while training."""

    def __init__(self, filepath: str | Path, display: bool = True) -> None:
        import matplotlib.pyplot as plt

        self.plt = plt
        self.filepath = Path(filepath)
        self.display = display
        self.losses: list[float] = []
        self.scores: list[int] = []
        self.figure, self.axes = plt.subplots(2, 1)
        if display:
            plt.ion()
            plt.show(block=False)

    def update(self, losses: list[float], scores: list[int]) -> None:
        self.losses.extend(losses)
        self.scores.extend(scores)
        loss_axes, score_axes = self.axes
        loss_axes.clear()
        score_axes.clear()
        loss_axes.scatter(
            range(len(self.losses)),
            self.losses,
            s=8,
            color="tab:blue",
        )
        score_axes.scatter(
            range(len(self.scores)),
            self.scores,
            s=8,
            color="tab:orange",
        )
        loss_axes.set(ylabel="Loss", title="Training loss")
        score_axes.set(xlabel="Game", ylabel="Score", title="Game score")
        self.figure.tight_layout()
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.figure.savefig(self.filepath)
        if self.display:
            self.plt.pause(0.001)


class DQNTrainingPlot:
    """Display and save the metrics collected by a Double DQN run."""

    _PLOTS = (
        ("loss", "Training loss", "Episode", "Mean loss", "tab:blue"),
        ("score", "Episode score", "Episode", "Score", "tab:orange"),
        ("epsilon", "Exploration rate", "Episode", "Epsilon", "tab:green"),
        ("mean_q", "Mean selected Q value", "Episode", "Q value", "tab:red"),
        (
            "replay_size",
            "Replay-buffer size",
            "Episode",
            "Transitions",
            "tab:purple",
        ),
        (
            "evaluation_score",
            "Evaluation score",
            "Episode",
            "Mean score",
            "tab:brown",
        ),
    )

    def __init__(
        self,
        filepath: str | Path,
        display: bool = True,
        save_every: int = 1,
    ) -> None:
        import matplotlib.pyplot as plt

        if save_every < 1:
            raise ValueError("save_every must be at least one")
        self.plt = plt
        self.filepath = Path(filepath)
        self.display = display
        self.save_every = save_every
        self.updates = 0
        self.metrics: dict[str, list[float | int]] = {
            name: [] for name, *_ in self._PLOTS
        }
        self.figure, self.axes = plt.subplots(3, 2, figsize=(10, 9))
        self.figure.subplots_adjust(hspace=0.45, wspace=0.3)
        if display:
            plt.ion()
            plt.show(block=False)

    def update(self, values: Mapping[str, float | int | None]) -> None:
        """Append one metric point and redraw at the configured interval."""
        for name in self.metrics:
            value = values.get(name)
            self.metrics[name].append(float("nan") if value is None else value)
        self.updates += 1
        if not self.display and self.updates % self.save_every:
            return
        self.save()

    def save(self) -> None:
        """Redraw and write the accumulated metrics immediately."""
        for axes, (name, title, xlabel, ylabel, color) in zip(
            self.axes.flat,
            self._PLOTS,
            strict=True,
        ):
            axes.clear()
            values = self.metrics[name]
            axes.scatter(range(len(values)), values, s=8, color=color)
            axes.set(title=title, xlabel=xlabel, ylabel=ylabel)

        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.figure.savefig(self.filepath)
        if self.display:
            self.plt.pause(0.001)


class PPOTrainingPlot:
    """Display one point for each fixed-transition PPO update."""

    _PLOTS = (
        (
            "score",
            "Mean completed-episode score",
            "PPO update",
            "Score",
            "tab:blue",
        ),
        ("loss", "Combined PPO loss", "PPO update", "Loss", "tab:orange"),
        (
            "policy_loss",
            "Actor policy loss",
            "PPO update",
            "Loss",
            "tab:green",
        ),
        ("value_loss", "Critic value loss", "PPO update", "Loss", "tab:red"),
        (
            "gae_advantage",
            "Mean absolute GAE advantage",
            "PPO update",
            "|Advantage|",
            "tab:purple",
        ),
        (
            "td_residual",
            "Mean absolute TD residual",
            "PPO update",
            "|TD residual|",
            "tab:brown",
        ),
        ("entropy", "Actor entropy", "PPO update", "Entropy", "tab:pink"),
        (
            "clip_fraction",
            "PPO clip fraction",
            "PPO update",
            "Fraction",
            "tab:gray",
        ),
        (
            "evaluation_score",
            "Fixed-seed evaluation score",
            "PPO update",
            "Mean score",
            "tab:olive",
        ),
    )

    def __init__(
        self,
        filepath: str | Path,
        display: bool = True,
        save_every: int = 1,
    ) -> None:
        import matplotlib.pyplot as plt

        if save_every < 1:
            raise ValueError("save_every must be at least one")
        self.plt = plt
        self.filepath = Path(filepath)
        self.display = display
        self.save_every = save_every
        self.updates = 0
        self.metrics: dict[str, list[float | int]] = {
            name: [] for name, *_ in self._PLOTS
        }
        self.figure, self.axes = plt.subplots(3, 3, figsize=(13, 10))
        self.figure.subplots_adjust(hspace=0.5, wspace=0.35)
        if display:
            plt.ion()
            plt.show(block=False)

    def update(self, values: Mapping[str, float | int | None]) -> None:
        """Append one fixed-rollout update and redraw when configured."""
        for name in self.metrics:
            value = values.get(name)
            self.metrics[name].append(float("nan") if value is None else value)
        self.updates += 1
        if not self.display and self.updates % self.save_every:
            return
        self.save()

    def save(self) -> None:
        """Redraw and write every PPO panel immediately."""
        for axes, (name, title, xlabel, ylabel, color) in zip(
            self.axes.flat,
            self._PLOTS,
            strict=True,
        ):
            axes.clear()
            values = self.metrics[name]
            axes.scatter(range(len(values)), values, s=8, color=color)
            axes.set(title=title, xlabel=xlabel, ylabel=ylabel)

        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.figure.savefig(self.filepath)
        if self.display:
            self.plt.pause(0.001)
