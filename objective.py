from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from models.random_forest import build_model as build_random_forest
from models.xgboost import build_model as build_xgboost


@dataclass
class ExperimentContext:
    dataset_name: str
    model_name: str
    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    categorical_columns: tuple[str, ...]
    seed: int = 42
    X_test: pd.DataFrame | None = None
    y_test: pd.Series | None = None


def _build_model(
    parameters: dict[str, Any], context: ExperimentContext
) -> tuple[Any, pd.Series, pd.Series, pd.Series | None]:
    clean_parameters = dict(parameters)
    clean_parameters["_numeric_columns"] = list(
        context.X_train.select_dtypes(exclude=["object", "category"]).columns
    )
    clean_parameters["_seed"] = context.seed
    if context.model_name == "random_forest":
        return (
            build_random_forest(clean_parameters, context.categorical_columns),
            context.y_train,
            context.y_validation,
            context.y_test,
        )
    if context.model_name == "xgboost":
        labels = sorted(context.y_train.astype(str).unique())
        label_to_int = {label: index for index, label in enumerate(labels)}
        y_train = context.y_train.astype(str).map(label_to_int)
        y_validation = context.y_validation.astype(str).map(label_to_int)
        y_test = (
            context.y_test.astype(str).map(label_to_int)
            if context.y_test is not None
            else None
        )
        return (
            build_xgboost(clean_parameters, context.categorical_columns),
            y_train,
            y_validation,
            y_test,
        )
    raise ValueError(f"Unsupported model: {context.model_name}")


def evaluate_configuration(
    parameters: dict[str, Any], context: ExperimentContext
) -> dict[str, Any]:
    started = time.perf_counter()
    result: dict[str, Any] = {
        "dataset": context.dataset_name,
        "model": context.model_name,
        "parameters": dict(parameters),
        "status": "failed",
        "validation_macro_f1": None,
        "validation_accuracy": None,
        "validation_balanced_accuracy": None,
        "error": None,
    }
    try:
        model, y_train, y_validation, _ = _build_model(parameters, context)
        model.fit(context.X_train, y_train)
        predictions = model.predict(context.X_validation)
        result.update(
            status="completed",
            validation_macro_f1=float(
                f1_score(y_validation, predictions, average="macro")
            ),
            validation_accuracy=float(accuracy_score(y_validation, predictions)),
            validation_balanced_accuracy=float(
                balanced_accuracy_score(y_validation, predictions)
            ),
        )
    except (ValueError, TypeError, RuntimeError) as error:
        result["error"] = f"{type(error).__name__}: {error}"
    result["elapsed_seconds"] = time.perf_counter() - started
    return result


def evaluate_test_configuration(
    parameters: dict[str, Any], context: ExperimentContext
) -> dict[str, Any]:
    if context.X_test is None or context.y_test is None:
        raise ValueError("Test data is required for final evaluation")

    started = time.perf_counter()
    result: dict[str, Any] = {
        "dataset": context.dataset_name,
        "model": context.model_name,
        "parameters": dict(parameters),
        "status": "failed",
        "test_macro_f1": None,
        "test_accuracy": None,
        "test_balanced_accuracy": None,
        "error": None,
    }
    try:
        model, y_train, _, y_test = _build_model(parameters, context)
        model.fit(context.X_train, y_train)
        predictions = model.predict(context.X_test)
        if y_test is not None:
            context_y_test = y_test
        else:
            context_y_test = context.y_test
        result.update(
            status="completed",
            test_macro_f1=float(
                f1_score(context_y_test, predictions, average="macro")
            ),
            test_accuracy=float(accuracy_score(context_y_test, predictions)),
            test_balanced_accuracy=float(
                balanced_accuracy_score(context_y_test, predictions)
            ),
        )
    except (ValueError, TypeError, RuntimeError) as error:
        result["error"] = f"{type(error).__name__}: {error}"
    result["elapsed_seconds"] = time.perf_counter() - started
    return result
