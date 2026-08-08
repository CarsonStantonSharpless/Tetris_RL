from pathlib import Path


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
