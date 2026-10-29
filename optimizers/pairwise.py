from typing import Any, Callable

from .common import pairwise_configurations


def run(search_space: dict[str, tuple], evaluate: Callable, trials: int, seed: int) -> list[dict[str, Any]]:
    del seed
    records = []
    for trial_id, parameters in enumerate(pairwise_configurations(search_space, trials)):
        result = evaluate(parameters)
        result.update(optimizer="pairwise", trial_id=trial_id)
        records.append(result)
    return records

