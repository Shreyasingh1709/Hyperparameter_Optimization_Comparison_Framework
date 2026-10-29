from typing import Any, Callable


def run(search_space: dict[str, tuple], evaluate: Callable, trials: int, seed: int) -> list[dict[str, Any]]:
    try:
        import ConfigSpace as cs
        from smac import HyperparameterOptimizationFacade, Scenario
    except ImportError as error:
        raise RuntimeError("SMAC3 is unavailable. Install with: py -3.10 -m pip install smac") from error

    config_space = cs.ConfigurationSpace(seed=seed)
    hyperparameters = []
    for name, specification in search_space.items():
        if specification[0] == "int":
            hyperparameters.append(cs.Integer(name, (specification[1], specification[2])))
        elif specification[0] == "choice":
            hyperparameters.append(cs.Categorical(name, specification[1]))
    config_space.add(hyperparameters)
    records: list[dict[str, Any]] = []

    def target(configuration, seed: int = 0) -> float:
        del seed
        result = evaluate(dict(configuration))
        result.update(optimizer="smac3", trial_id=len(records))
        records.append(result)
        return -(result["validation_macro_f1"] or 0.0)

    scenario = Scenario(config_space, name="hpo_pilot_smac", n_trials=trials, seed=seed,
                        output_directory=__import__("pathlib").Path("results") / "smac3")
    smac = HyperparameterOptimizationFacade(scenario, target, overwrite=True)
    smac.optimize()
    return records

