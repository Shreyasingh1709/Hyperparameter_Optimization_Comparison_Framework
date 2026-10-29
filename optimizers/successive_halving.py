from typing import Any, Callable

from .random_search import sample_parameters


def run(search_space: dict[str, tuple], evaluate: Callable, trials: int, seed: int) -> list[dict[str, Any]]:
    import random
    rng = random.Random(seed)
    candidates = [sample_parameters(search_space, rng) for _ in range(max(2, trials // 2))]
    records: list[dict[str, Any]] = []
    resource_levels = [max(10, 50 // max(1, trials // 3)), 50, 100]
    for level in resource_levels:
        scored = []
        for parameters in candidates:
            parameters = dict(parameters)
            parameters["n_estimators"] = min(parameters["n_estimators"], level)
            result = evaluate(parameters)
            result.update(optimizer="successive_halving", trial_id=len(records), resource=level)
            records.append(result)
            scored.append((result["validation_macro_f1"] or 0.0, parameters))
        scored.sort(reverse=True, key=lambda item: item[0])
        candidates = [parameters for _, parameters in scored[:max(1, len(scored) // 2)]]
        if len(records) >= trials:
            return records
    while len(records) < trials:
        parameters = dict(candidates[0])
        parameters["n_estimators"] = min(parameters["n_estimators"], resource_levels[-1])
        result = evaluate(parameters)
        result.update(
            optimizer="successive_halving",
            trial_id=len(records),
            resource=resource_levels[-1],
        )
        records.append(result)
    return records
