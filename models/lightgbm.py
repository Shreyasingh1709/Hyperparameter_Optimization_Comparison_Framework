from __future__ import annotations

from typing import Any

from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


SEARCH_SPACE = {
    "n_estimators": ("int", 50, 500),
    "max_depth": ("choice", [-1, 3, 5, 8, 12, 16]),
    "learning_rate": ("choice", [0.01, 0.03, 0.05, 0.1, 0.2]),
    "num_leaves": ("int", 15, 127),
    "min_child_samples": ("int", 5, 100),
    "subsample": ("choice", [0.7, 0.85, 1.0]),
    "subsample_freq": ("choice", [1, 5]),
    "colsample_bytree": ("choice", [0.7, 0.85, 1.0]),
    "reg_alpha": ("choice", [0.0, 0.1, 1.0]),
    "reg_lambda": ("choice", [0.0, 0.1, 1.0]),
}


def build_model(
    parameters: dict[str, Any], categorical_columns: tuple[str, ...]
) -> Pipeline:
    model_parameters = dict(parameters)
    numeric_columns = model_parameters.pop("_numeric_columns", [])
    seed = model_parameters.pop("_seed", 42)
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), numeric_columns),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                list(categorical_columns),
            ),
        ],
        remainder="drop",
    )
    model = LGBMClassifier(
        **model_parameters,
        objective="binary",
        random_state=seed,
        n_jobs=1,
        verbosity=-1,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])
