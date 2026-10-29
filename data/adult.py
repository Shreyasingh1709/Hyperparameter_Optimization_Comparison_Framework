from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DatasetSplits:
    name: str
    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series
    categorical_columns: tuple[str, ...]


def load_adult(seed: int = 42, max_rows: int | None = None) -> DatasetSplits:
    """Load Adult and create a stratified 70/15/15 split.

    ``max_rows`` is an explicit pilot-only cap applied before splitting.
    """
    adult = fetch_openml("adult", version=2, as_frame=True, parser="auto")
    frame = adult.frame.copy()
    target_column = "class"
    if target_column not in frame.columns:
        raise ValueError(
            f"Adult schema mismatch: expected target {target_column!r}; "
            f"found {list(frame.columns)!r}"
        )

    X = frame.drop(columns=[target_column])
    y = frame[target_column].astype(str).str.strip()
    if y.nunique() != 2:
        raise ValueError(f"Adult target must be binary, found labels {sorted(y.unique())!r}")
    if max_rows is not None:
        if max_rows < 2:
            raise ValueError("max_rows must be at least 2")
        if max_rows < len(X):
            X, _, y, _ = train_test_split(
                X, y, train_size=max_rows, stratify=y, random_state=seed
            )

    X_train, X_remainder, y_train, y_remainder = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=seed
    )
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_remainder, y_remainder, test_size=0.50, stratify=y_remainder, random_state=seed
    )
    categorical_columns = tuple(X.select_dtypes(include=["object", "category"]).columns)
    return DatasetSplits(
        "adult", X_train, X_validation, X_test, y_train, y_validation, y_test,
        categorical_columns,
    )
