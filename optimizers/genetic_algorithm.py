import random
from typing import Any, Callable

from .random_search import sample_parameters


def run(search_space: dict[str, tuple], evaluate: Callable, trials: int, seed: int) -> list[dict[str, Any]]:
    try:
        from deap import algorithms, base, creator, tools
    except ImportError as error:
        raise RuntimeError("DEAP is unavailable. Install with: py -3.10 -m pip install deap") from error

    rng = random.Random(seed)
    fitness_name = f"FitnessMax_{seed}"
    individual_name = f"Individual_{seed}"
    if not hasattr(creator, fitness_name):
        creator.create(fitness_name, base.Fitness, weights=(1.0,))
        creator.create(individual_name, dict, fitness=getattr(creator, fitness_name))
    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, getattr(creator, individual_name),
                     lambda: sample_parameters(search_space, rng))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    records: list[dict[str, Any]] = []
    population = toolbox.population(n=max(2, min(4, trials)))
    for trial_id in range(trials):
        individual = population[trial_id % len(population)]
        result = evaluate(dict(individual))
        result.update(optimizer="genetic_algorithm", trial_id=trial_id)
        records.append(result)
        if result["status"] == "completed":
            individual.fitness.values = (result["validation_macro_f1"],)
        if trial_id + 1 < trials:
            parent = max(population, key=lambda item: getattr(item.fitness, "values", (float("-inf"),)))
            child = type(parent)(parent)
            keys = list(search_space)
            key = rng.choice(keys)
            child[key] = sample_parameters({key: search_space[key]}, rng)[key]
            population[trial_id % len(population)] = child
    return records

