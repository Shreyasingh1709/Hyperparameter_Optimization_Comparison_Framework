from typing import Any, Callable


def run(search_space: dict[str, tuple], evaluate: Callable, trials: int, seed: int) -> list[dict[str, Any]]:
    try:
        from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
    except ImportError as error:
        raise RuntimeError("Hyperopt TPE is unavailable. Install with: py -3.10 -m pip install hyperopt") from error

    space = {}
    for name, specification in search_space.items():
        if specification[0] == "int":
            space[name] = hp.quniform(name, specification[1], specification[2], 1)
        elif specification[0] == "choice":
            space[name] = hp.choice(name, specification[1])
        else:
            raise ValueError(f"Unsupported search-space kind: {specification[0]!r}")
    records: list[dict[str, Any]] = []
    hyperopt_trials = Trials()

    def objective(parameters: dict[str, Any]) -> dict[str, Any]:
        for name, specification in search_space.items():
            if specification[0] == "int":
                parameters[name] = int(parameters[name])
        result = evaluate(parameters)
        result.update(optimizer="hyperopt_tpe", trial_id=len(records))
        records.append(result)
        return {"loss": 1.0 - (result["validation_macro_f1"] or 0.0), "status": STATUS_OK}

    fmin(objective, space, algo=tpe.suggest, max_evals=trials, trials=hyperopt_trials,
         rstate=__import__("numpy").random.default_rng(seed), show_progressbar=False)
    return records

