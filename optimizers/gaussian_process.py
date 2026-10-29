from typing import Any, Callable


def run(search_space: dict[str, tuple], evaluate: Callable, trials: int, seed: int) -> list[dict[str, Any]]:
    try:
        from skopt import Optimizer
        from skopt.space import Categorical, Integer
    except ImportError as error:
        raise RuntimeError("Gaussian Process optimization is unavailable. Install with: py -3.10 -m pip install scikit-optimize") from error

    dimensions = []
    names = list(search_space)
    for name in names:
        specification = search_space[name]
        dimensions.append(
            Integer(specification[1], specification[2], name=name)
            if specification[0] == "int"
            else Categorical(specification[1], name=name)
        )
    optimizer = Optimizer(dimensions, base_estimator="GP", acq_func="EI", random_state=seed)
    records = []
    for trial_id in range(trials):
        vector = optimizer.ask()
        result = evaluate(dict(zip(names, vector)))
        result.update(optimizer="gaussian_process", trial_id=trial_id)
        records.append(result)
        optimizer.tell(vector, -(result["validation_macro_f1"] or 0.0))
    return records

