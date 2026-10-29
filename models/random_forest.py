from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


SEARCH_SPACE = {
    "n_estimators": ("int", 50, 150),
    "max_depth": ("choice", [None, 8, 16, 24]),
    "min_samples_split": ("int", 2, 12),
    "min_samples_leaf": ("int", 1, 6),
    "max_features": ("choice", ["sqrt", "log2", None]),
    "bootstrap": ("choice", [True, False]),
    "criterion": ("choice", ["gini", "entropy"]),
}


def build_model(parameters: dict[str, Any], categorical_columns: tuple[str, ...]) -> Pipeline:
    model_parameters = dict(parameters)
    numeric_columns = model_parameters.pop("_numeric_columns", [])
    seed = model_parameters.pop("_seed", 42)
    model_parameters.pop("_num_classes", None)
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
    model = RandomForestClassifier(
        **model_parameters,
        random_state=seed,
        n_jobs=1,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])
