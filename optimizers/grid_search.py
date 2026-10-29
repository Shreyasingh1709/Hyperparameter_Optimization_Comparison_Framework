from itertools import islice, product
from typing import Any, Callable

from .common import values_for


def run(search_space: dict[str, tuple], evaluate: Callable, trials: int, seed: int) -> list[dict[str, Any]]:
    del seed
    names = list(search_space)
    combinations = product(*(values_for(search_space[name]) for name in names))
    records = []
    for trial_id, values in enumerate(islice(combinations, trials)):
        result = evaluate(dict(zip(names, values)))
        result.update(optimizer="grid_search", trial_id=trial_id)
        records.append(result)
    return records

