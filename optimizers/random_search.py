from __future__ import annotations

import random
from typing import Any, Callable


def sample_parameters(search_space: dict[str, tuple], rng: random.Random) -> dict[str, Any]:
    parameters = {}
    for name, specification in search_space.items():
        kind = specification[0]
        if kind == "int":
            parameters[name] = rng.randint(specification[1], specification[2])
        elif kind == "choice":
            parameters[name] = rng.choice(specification[1])
        else:
            raise ValueError(f"Unsupported search-space kind: {kind!r}")
    return parameters


def run(
    search_space: dict[str, tuple],
    evaluate: Callable[[dict[str, Any]], dict[str, Any]],
    trials: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    results = []
    for trial_id in range(trials):
        parameters = sample_parameters(search_space, rng)
        result = evaluate(parameters)
        result.update(optimizer="random_search", trial_id=trial_id)
        results.append(result)
    return results

