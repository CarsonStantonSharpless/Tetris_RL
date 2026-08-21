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
        loss_axes.plot(self.losses)
        score_axes.plot(self.scores)
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
        ("loss", "Training loss", "Episode", "Mean loss"),
        ("score", "Episode score", "Episode", "Score"),
        ("epsilon", "Exploration rate", "Episode", "Epsilon"),
        ("mean_q", "Mean selected Q value", "Episode", "Q value"),
        ("replay_size", "Replay-buffer size", "Episode", "Transitions"),
        ("evaluation_score", "Evaluation score", "Episode", "Mean score"),
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
        self.metrics = {name: [] for name, *_ in self._PLOTS}
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
        for axes, (name, title, xlabel, ylabel) in zip(
            self.axes.flat,
            self._PLOTS,
            strict=True,
        ):
            axes.clear()
            axes.plot(self.metrics[name])
            axes.set(title=title, xlabel=xlabel, ylabel=ylabel)

        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.figure.savefig(self.filepath)
        if self.display:
            self.plt.pause(0.001)
