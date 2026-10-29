from __future__ import annotations

from typing import Any, Callable


def run(
    search_space: dict[str, tuple],
    evaluate: Callable[[dict[str, Any]], dict[str, Any]],
    trials: int,
    seed: int,
) -> list[dict[str, Any]]:
    try:
        import optuna
    except ImportError as error:
        raise RuntimeError(
            "Optuna TPE is unavailable. Install it with: py -3.10 -m pip install optuna"
        ) from error

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    records: list[dict[str, Any]] = []

    def objective(trial: Any) -> float:
        parameters = {}
        for name, specification in search_space.items():
            if specification[0] == "int":
                parameters[name] = trial.suggest_int(name, specification[1], specification[2])
            elif specification[0] == "choice":
                parameters[name] = trial.suggest_categorical(name, specification[1])
            else:
                raise ValueError(f"Unsupported search-space kind: {specification[0]!r}")
        result = evaluate(parameters)
        result.update(optimizer="optuna_tpe", trial_id=trial.number)
        records.append(result)
        return result["validation_macro_f1"] if result["status"] == "completed" else 0.0

    study.optimize(objective, n_trials=trials, n_jobs=1)
    return records

