from typing import Any, Callable

from .common import decode_vector


def run(search_space: dict[str, tuple], evaluate: Callable, trials: int, seed: int) -> list[dict[str, Any]]:
    try:
        import cma
    except ImportError as error:
        raise RuntimeError("CMA-ES is unavailable. Install with: py -3.10 -m pip install cma") from error

    names = list(search_space)
    lower = [specification[1] if specification[0] == "int" else 0 for specification in search_space.values()]
    upper = [specification[2] if specification[0] == "int" else len(specification[1]) - 1 for specification in search_space.values()]
    initial = [(low + high) / 2 for low, high in zip(lower, upper)]
    records: list[dict[str, Any]] = []
    options = {
        "bounds": [lower, upper],
        "maxfevals": trials,
        "seed": seed,
        "verb_disp": 0,
        "popsize": max(2, min(4, trials)),
        "tolflatfitness": trials,
        "tolfun": 0,
        "tolfunhist": 0,
        "tolx": 0,
        "tolxstagnation": False,
        "tolstagnation": trials,
    }

    def objective(vector):
        if len(records) >= trials:
            return 0.0
        result = evaluate(decode_vector(search_space, vector))
        result.update(optimizer="cma_es", trial_id=len(records))
        records.append(result)
        return -(result["validation_macro_f1"] or 0.0)

    cma.fmin(objective, initial, 0.2, options=options)
    return records
