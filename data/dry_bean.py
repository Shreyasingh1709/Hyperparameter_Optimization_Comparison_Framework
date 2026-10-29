from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd
from sklearn.model_selection import train_test_split

from config import PROJECT_ROOT


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


_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00602/DryBeanDataset.zip"
_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Dry_Bean_Dataset.xlsx"


def _ensure_data_file() -> Path:
    if _DATA_PATH.exists():
        return _DATA_PATH
    _DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(_DATA_URL, timeout=60) as response:
        archive = ZipFile(BytesIO(response.read()))
        member = "DryBeanDataset/Dry_Bean_Dataset.xlsx"
        if member not in archive.namelist():
            raise ValueError(f"Dry Bean archive is missing {member!r}")
        _DATA_PATH.write_bytes(archive.read(member))
    return _DATA_PATH


def load_dry_bean(seed: int = 42, max_rows: int | None = None) -> DatasetSplits:
    """Load UCI Dry Bean and create a stratified 70/15/15 split."""
    frame = pd.read_excel(_ensure_data_file())
    target_column = "Class"
    if target_column not in frame.columns:
        raise ValueError(
            f"Dry Bean schema mismatch: expected target {target_column!r}; "
            f"found {list(frame.columns)!r}"
        )
    X = frame.drop(columns=[target_column])
    y = frame[target_column].astype(str).str.strip()
    if y.nunique() < 2:
        raise ValueError(f"Dry Bean target must have multiple classes, found {sorted(y.unique())!r}")
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
        "dry_bean",
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        categorical_columns,
    )
