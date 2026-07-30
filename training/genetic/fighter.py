"""
https://codemyroad.wordpress.com/2013/04/14/tetris-ai-the-near-perfect-player/
"""
from dataclasses import dataclass
import numpy as np

from agents.player import Player, Policy

@dataclass
class Parameters():
    alpha: float
    beta: float
    gamma: float
    delta: float
    epsilon: float
    
    @property
    def vector(self) -> np.ndarray:
        return np.array(
            [self.alpha,
             self.beta,
             self.gamma,
             self.delta,
             self.epsilon]
        )

class Fighter():
    params: Parameters
    fitness_score: int

    def __init__(
            self,
            params: Parameters | None = None,
            num_games: int = 100
        ) -> None:
        self.params = params if params is not None else self.random_params()
        self.fitness_score = self.fitness(num_games)

    def fitness(
        self,
        num_games: int = 100,
    ) -> int:
        """
        the score is the sum of all games
        """
        score: int = 0
        for _ in range(num_games):
            player = Player(
                policy=Policy.HEURISTIC,
                display=False,
                policy_params={
                    "alpha": self.params.alpha,
                    "beta": self.params.beta,
                    "gamma": self.params.gamma,
                    "delta": self.params.delta,
                    "epsilon": self.params.epsilon,
                },
            )
            score += player.start().score
        return score
    
    def random_params(self) -> Parameters:
        numbers = np.random.multinomial(5, [1/5]*5)
        return Parameters(
            alpha = numbers[0],
            beta = numbers[1],
            gamma = numbers[2],
            delta = numbers[3],
            epsilon = numbers[4]
        )
    
    
