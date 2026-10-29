import pandas as pd

from experiment_runner import _json_default
from objective import ExperimentContext, evaluate_configuration, evaluate_test_configuration
from models.random_forest import SEARCH_SPACE
from models.xgboost import SEARCH_SPACE as XGBOOST_SEARCH_SPACE
from models.lightgbm import SEARCH_SPACE as LIGHTGBM_SEARCH_SPACE
from optimizers.random_search import run
from optimizers.common import pairwise_configurations
from optimizers.grid_search import run as grid_run
from optimizers.cma_es import run as cma_run
from optimizers.hyperband import run as hyperband_run
from optimizers.successive_halving import run as successive_halving_run


def _context():
    X = pd.DataFrame(
        {"age": [20, 30, 40, 50, 25, 35, 45, 55], "work": ["a", "b"] * 4}
    )
    y = pd.Series(["low", "high"] * 4)
    return ExperimentContext("synthetic", "random_forest", X, X, y, y, ("work",))


def test_shared_objective_returns_metrics():
    result = evaluate_configuration(
        {
            "n_estimators": 5,
            "max_depth": 4,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "bootstrap": True,
            "criterion": "gini",
        },
        _context(),
    )
    assert result["status"] == "completed"
    assert 0.0 <= result["validation_macro_f1"] <= 1.0


def test_test_objective_returns_test_metrics():
    context = _context()
    context.X_test = context.X_validation
    context.y_test = context.y_validation
    result = evaluate_test_configuration(
        {
            "n_estimators": 5,
            "max_depth": 4,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "bootstrap": True,
            "criterion": "gini",
        },
        context,
    )
    assert result["status"] == "completed"
    assert 0.0 <= result["test_macro_f1"] <= 1.0


def test_xgboost_objective_returns_metrics():
    result = evaluate_configuration(
        {
            "n_estimators": 5,
            "max_depth": 3,
            "learning_rate": 0.1,
            "min_child_weight": 1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
        },
        ExperimentContext(
            "synthetic",
            "xgboost",
            _context().X_train,
            _context().X_validation,
            _context().y_train,
            _context().y_validation,
            ("work",),
        ),
    )
    assert result["status"] == "completed"
    assert 0.0 <= result["validation_macro_f1"] <= 1.0
    assert set(XGBOOST_SEARCH_SPACE) == {
        "n_estimators",
        "max_depth",
        "learning_rate",
        "min_child_weight",
        "subsample",
        "colsample_bytree",
    }


def test_lightgbm_objective_returns_metrics():
    context = _context()
    context.model_name = "lightgbm"
    result = evaluate_configuration(
        {
            "n_estimators": 5,
            "max_depth": 3,
            "learning_rate": 0.1,
            "num_leaves": 15,
            "min_child_samples": 10,
            "subsample": 1.0,
            "subsample_freq": 1,
            "colsample_bytree": 1.0,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
        },
        context,
    )
    assert result["status"] == "completed"
    assert 0.0 <= result["validation_macro_f1"] <= 1.0
    assert set(LIGHTGBM_SEARCH_SPACE) == {
        "n_estimators",
        "max_depth",
        "learning_rate",
        "num_leaves",
        "min_child_samples",
        "subsample",
        "subsample_freq",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
    }


def test_random_search_uses_requested_budget():
    results = run(
        SEARCH_SPACE,
        lambda parameters: {"status": "completed", "parameters": parameters},
        trials=3,
        seed=7,
    )
    assert len(results) == 3
    assert {result["optimizer"] for result in results} == {"random_search"}


def test_grid_and_pairwise_are_distinct_search_strategies():
    grid = grid_run(SEARCH_SPACE, lambda parameters: {"status": "completed"}, 3, 7)
    pairwise = pairwise_configurations(SEARCH_SPACE, 3)
    assert len(grid) == 3
    assert len(pairwise) == 3
    assert {result["optimizer"] for result in grid} == {"grid_search"}


def test_json_default_serializes_numpy_scalars():
    import numpy as np

    assert _json_default(np.int64(7)) == 7
    assert _json_default(np.bool_(True)) is True


def test_cma_es_does_not_evaluate_beyond_requested_budget():
    evaluations = []

    def evaluator(parameters):
        evaluations.append(parameters)
        return {"status": "completed", "validation_macro_f1": 0.5}

    results = cma_run(SEARCH_SPACE, evaluator, trials=3, seed=7)

    assert len(evaluations) == 3
    assert len(results) == 3


def test_cma_es_honors_ten_trial_budget_with_flat_fitness():
    evaluator = lambda parameters: {
        "status": "completed",
        "validation_macro_f1": 0.5,
    }

    results = cma_run(SEARCH_SPACE, evaluator, trials=10, seed=7)

    assert len(results) == 10


def test_resource_optimizers_use_the_requested_budget():
    evaluator = lambda parameters: {
        "status": "completed",
        "validation_macro_f1": 0.5,
    }

    assert len(hyperband_run(SEARCH_SPACE, evaluator, trials=10, seed=7)) == 10
    assert len(successive_halving_run(SEARCH_SPACE, evaluator, trials=10, seed=7)) == 10
