import sys
from statistics import fmean, pstdev

from tqdm import tqdm

from training.evaluation.parallel import evaluate_parallel
from training.evaluation.specs import HeuristicSpec
from training.genetic.population import Population
from training.genetic.fighter import Evaluation, Fighter, Genome, random_genome
from storage.parameters import write_genetic_parameters


def tournament(
    num_rounds: int,
    num_fighters: int,
    num_games: int = 1000,
    max_ticks: int = 1000,
    cull_percent: float = 0.3,
    tournament_percent: float = 0.1,
    mutation_chance: float = 0.75,
    mutation_amount: float = 0.15,
    workers: int | None = None,
    filepath: str | None = None,
    visualize: bool = True,
) -> tuple[Genome, Genome]:
    if num_rounds < 0:
        raise ValueError("num_rounds must be zero or greater")
    if num_fighters < 2:
        raise ValueError("num_fighters must be at least two")
    if num_games < 1:
        raise ValueError("num_games must be at least one")
    if max_ticks < 1:
        raise ValueError("max_ticks must be at least one")
    if not 0.0 <= cull_percent <= 1.0:
        raise ValueError("cull_percent must be between zero and one")
    if not 0.0 < tournament_percent <= 1.0:
        raise ValueError(
            "tournament_percent must be greater than zero and at most one"
        )
    if workers is not None and workers < 1:
        raise ValueError("workers must be at least one")

    training_max_ticks = max(1, max_ticks // 5)
    initial_genomes = [random_genome() for _ in range(num_fighters)]
    fighters = _evaluate_genomes(
        initial_genomes,
        num_games,
        training_max_ticks,
        workers,
        "Initial population",
        visualize,
    )
    seed_pop = Population(fighters=fighters)
    _finish_generation(
        seed_pop,
        "Initial population",
        visualize,
        filepath,
    )

    for round_index in range(num_rounds):
        offspring_genomes: list[Genome] = []
        offspring_count = int(len(seed_pop) * cull_percent)
        for _ in range(offspring_count):
            sub_pop = seed_pop.subpopulation(tournament_percent)
            child_genome = sub_pop.breed(
                mutation_chance,
                mutation_amount,
            )
            if child_genome is None:
                raise RuntimeError(
                    "cannot breed from an empty population"
                )
            offspring_genomes.append(child_genome)

        offspring = _evaluate_genomes(
            offspring_genomes,
            num_games,
            training_max_ticks,
            workers,
            f"Round {round_index + 1}/{num_rounds}",
            visualize,
        )

        seed_pop.cull(cull_percent)
        seed_pop += Population(fighters=offspring)
        _finish_generation(
            seed_pop,
            f"Round {round_index + 1}/{num_rounds}",
            visualize,
            filepath,
        )

    finalist_count = max(
        2,
        int(len(seed_pop) * tournament_percent),
    )
    finalists = sorted(
        seed_pop.fighters,
        key=lambda fighter: fighter.evaluation.fitness,
        reverse=True,
    )[:finalist_count]
    final_pop = Population(
        fighters=_evaluate_genomes(
            [fighter.evaluation.genome for fighter in finalists],
            num_games,
            max_ticks,
            workers,
            "Finalists",
            visualize,
        )
    )
    _finish_generation(final_pop, "Finalists", visualize, filepath)

    if final_pop.captain is None or final_pop.lieutenant is None:
        raise RuntimeError("tournament finished without two leaders")

    captain = final_pop.captain.evaluation.genome
    lieutenant = final_pop.lieutenant.evaluation.genome
    return captain, lieutenant


def _evaluate_genomes(
    genomes: list[Genome],
    num_games: int,
    max_ticks: int,
    workers: int | None,
    description: str,
    visualize: bool,
) -> list[Fighter]:
    scores: list[list[int]] = [[] for _ in genomes]
    results = evaluate_parallel(
        [HeuristicSpec(genome) for genome in genomes],
        list(range(num_games)),
        max_ticks,
        workers,
    )

    for result in tqdm(
        results,
        total=len(genomes) * num_games,
        desc=description,
        disable=not visualize,
        unit="game",
    ):
        scores[result.policy_index].append(result.episode.score)

    return [
        Fighter(
            Evaluation(
                genome=genome,
                fitness=fmean(genome_scores),
            )
        )
        for genome, genome_scores in zip(genomes, scores, strict=True)
    ]


def _finish_generation(
    population: Population,
    description: str,
    enabled: bool,
    filepath: str | None,
) -> None:
    captain = population.captain
    lieutenant = population.lieutenant
    if captain is None or lieutenant is None:
        return

    if enabled:
        fitnesses = [
            fighter.evaluation.fitness
            for fighter in population.fighters
        ]
        unique_genomes = len({
            fighter.evaluation.genome
            for fighter in population.fighters
        })
        print(
            f"{description}: captain fitness={captain.evaluation.fitness}, "
            f"params={_format_genome(captain.evaluation.genome)}",
            file=sys.stderr,
        )
        print(
            f"{description}: lieutenant fitness={lieutenant.evaluation.fitness}, "
            f"params={_format_genome(lieutenant.evaluation.genome)}",
            file=sys.stderr,
        )
        print(
            f"{description}: unique_genomes={unique_genomes}/{len(population)}, "
            f"fitness_stddev={pstdev(fitnesses)}",
            file=sys.stderr,
        )

    if filepath is not None:
        write_genetic_parameters(
            captain.evaluation.genome,
            lieutenant.evaluation.genome,
            filepath,
        )


def _format_genome(genome: Genome) -> str:
    return (
        f"alpha={genome.alpha:.6f}, beta={genome.beta:.6f}, "
        f"gamma={genome.gamma:.6f}, delta={genome.delta:.6f}, "
        f"epsilon={genome.epsilon:.6f}"
    )
