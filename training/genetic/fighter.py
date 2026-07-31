"""
https://codemyroad.wordpress.com/2013/04/14/tetris-ai-the-near-perfect-player/
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Genome:
    alpha: float
    beta: float
    gamma: float
    delta: float
    epsilon: float

    @property
    def vector(self) -> np.ndarray:
        return np.array(
            [
                self.alpha,
                self.beta,
                self.gamma,
                self.delta,
                self.epsilon,
            ]
        )


@dataclass(frozen=True)
class Evaluation:
    genome: Genome
    fitness: float


@dataclass(frozen=True)
class Fighter:
    evaluation: Evaluation


def random_genome() -> Genome:
    alpha, beta, gamma, delta, epsilon = np.random.dirichlet(
        np.ones(5)
    )

    return Genome(
        alpha=float(alpha),
        beta=float(beta),
        gamma=float(gamma),
        delta=float(delta),
        epsilon=float(epsilon),
    )
