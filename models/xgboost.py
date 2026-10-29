from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


SEARCH_SPACE = {
    "n_estimators": ("int", 50, 200),
    "max_depth": ("int", 3, 10),
    "learning_rate": ("choice", [0.03, 0.1, 0.2]),
    "min_child_weight": ("int", 1, 10),
    "subsample": ("choice", [0.7, 0.85, 1.0]),
    "colsample_bytree": ("choice", [0.7, 0.85, 1.0]),
}


def build_model(parameters: dict[str, Any], categorical_columns: tuple[str, ...]) -> Pipeline:
    model_parameters = dict(parameters)
    numeric_columns = model_parameters.pop("_numeric_columns", [])
    seed = model_parameters.pop("_seed", 42)
    num_classes = model_parameters.pop("_num_classes", 2)
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
    model = XGBClassifier(
        **model_parameters,
        objective="binary:logistic" if num_classes == 2 else "multi:softprob",
        num_class=num_classes if num_classes > 2 else None,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=1,
        tree_method="hist",
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])
