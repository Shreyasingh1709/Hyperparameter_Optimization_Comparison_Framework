from typing import Any, Callable

from .random_search import sample_parameters


def run(search_space: dict[str, tuple], evaluate: Callable, trials: int, seed: int) -> list[dict[str, Any]]:
    import random
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    candidates = [sample_parameters(search_space, rng) for _ in range(max(2, trials // 3))]
    for bracket, resource in enumerate((25, 50, 100)):
        for parameters in candidates:
            if len(records) >= trials:
                return records
            parameters = dict(parameters)
            parameters["n_estimators"] = min(parameters["n_estimators"], resource)
            result = evaluate(parameters)
            result.update(optimizer="hyperband", trial_id=len(records), resource=resource, bracket=bracket)
            records.append(result)
        candidates = candidates[:max(1, len(candidates) // 2)]
    while len(records) < trials:
        parameters = dict(candidates[0])
        parameters["n_estimators"] = min(parameters["n_estimators"], 100)
        result = evaluate(parameters)
        result.update(
            optimizer="hyperband",
            trial_id=len(records),
            resource=100,
            bracket=2,
        )
        records.append(result)
    return records
