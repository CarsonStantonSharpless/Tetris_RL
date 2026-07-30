"""
https://codemyroad.wordpress.com/2013/04/14/tetris-ai-the-near-perfect-player/
"""
import random
import numpy as np

from training.genetic.fighter import Fighter, Parameters


class Population:
    fighters: list[Fighter]
    captain: Fighter | None
    lieutenant: Fighter | None

    def __init__(
        self,
        fighters: list[Fighter] | None = None,
        n: int = 0,
        num_games: int = 100,
    ) -> None:
        if fighters is not None:
            self.fighters = list(fighters)
        else:
            self.fighters = [
                Fighter(num_games)
                for _ in range(n)
            ]

        self._update_leadership()

    def __len__(self) -> int:
        return len(self.fighters)

    def __add__(self, other: "Population") -> "Population":
        """
        Return a new population containing fighters from both populations.

        Neither original population is modified.
        """
        if not isinstance(other, Population):
            return NotImplemented

        return Population(
            fighters=self.fighters + other.fighters
        )

    def __iadd__(self, other: "Population") -> "Population":
        """
        Add another population's fighters to this population in place.
        """
        if not isinstance(other, Population):
            return NotImplemented

        self.fighters.extend(other.fighters)
        self._update_leadership()

        return self

    def subpopulation(self, val: float) -> "Population":
        if not 0.0 <= val <= 1.0:
            raise ValueError("val must be between 0.0 and 1.0")

        count = int(len(self.fighters) * val)
        if self.fighters and val > 0.0:
            count = max(1, count)
        selected = random.sample(self.fighters, count)

        return Population(fighters=selected)

    def cull(self, val: float) -> "Population":
        """
        Remove the weakest percentage of the population.
        """
        self._validate_percentage(val)

        count = int(len(self.fighters) * val)

        if count == 0:
            return Population()

        # Lowest fitness scores appear first.
        ranked = sorted(
            self.fighters,
            key=lambda fighter: fighter.fitness_score,
        )

        culled = ranked[:count]
        self.fighters = ranked[count:]

        self._update_leadership()

        return Population(fighters=culled)

    def fittest(self) -> tuple[Fighter | None, Fighter | None]:
        """
        Return the strongest and second-strongest fighters.
        """
        if not self.fighters:
            return None, None

        strongest = sorted(
            self.fighters,
            key=lambda fighter: fighter.fitness_score,
            reverse=True,
        )

        captain = strongest[0]
        lieutenant = strongest[1] if len(strongest) > 1 else None

        return captain, lieutenant
    
    def generate_offspring(
        self,
        mutation_chance: float,
        mutation_amount: float,
        num_games: int = 100,
    ) -> Fighter | None:
        if not 0.0 <= mutation_chance <= 1.0:
            raise ValueError("mutation_chance must be between 0 and 1")

        if mutation_amount < 0.0:
            raise ValueError("mutation_amount must be non-negative")

        if self.captain is None:
            return None

        # A population with one fighter can only clone/mutate that fighter.
        if self.lieutenant is None:
            child_vector = self.captain.params.vector.astype(float).copy()
        else:
            captain_score = max(0.0, float(self.captain.fitness_score))
            lieutenant_score = max(0.0, float(self.lieutenant.fitness_score))
            total_score = captain_score + lieutenant_score

            if total_score == 0.0:
                # Neither parent has a useful fitness weight.
                child_vector = (
                    self.captain.params.vector.astype(float)
                    + self.lieutenant.params.vector.astype(float)
                ) / 2.0
            else:
                child_vector = (
                    self.captain.params.vector.astype(float) * captain_score
                    + self.lieutenant.params.vector.astype(float) * lieutenant_score
                ) / total_score

        # mutation_chance determines whether this child mutates at all.
        if np.random.random() < mutation_chance:
            mutation = np.random.uniform(
                low=-mutation_amount,
                high=mutation_amount,
                size=child_vector.shape,
            )
            child_vector += mutation

        # Parameters cannot be negative.
        child_vector = np.clip(child_vector, 0.0, None)

        # Ensure the parameters add up to 1.
        vector_sum = child_vector.sum()

        if vector_sum <= 0.0:
            child_vector = np.full(5, 1.0 / 5.0)
        else:
            child_vector /= vector_sum

        child_params = Parameters(
            alpha=float(child_vector[0]),
            beta=float(child_vector[1]),
            gamma=float(child_vector[2]),
            delta=float(child_vector[3]),
            epsilon=float(child_vector[4]),
        )

        return Fighter(
            params=child_params,
            num_games=num_games,
        )
        

    def _update_leadership(self) -> None:
        self.captain, self.lieutenant = self.fittest()

    @staticmethod
    def _validate_percentage(val: float) -> None:
        if not isinstance(val, (int, float)):
            raise TypeError("Percentage must be an int or float.")

        if not 0.0 <= val <= 1.0:
            raise ValueError("Percentage must be between 0.0 and 1.0.")
