from __future__ import annotations

from pathlib import Path
import hashlib
import pandas as pd

TARGET_COLUMN = "Survived"
REQUIRED_COLUMNS = {
    "PassengerId",
    "Survived",
    "Pclass",
    "Name",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Ticket",
    "Fare",
    "Cabin",
    "Embarked",
}

FEATURE_COLUMNS = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
]


def sha256_file(path: str | Path) -> str:
    """Return a SHA-256 hash so we can tell exactly which data version trained a model."""
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_titanic(path: str | Path) -> pd.DataFrame:
    """Load the CSV and check that it has the columns the pipeline expects."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df[TARGET_COLUMN].isna().any():
        raise ValueError(f"Target column {TARGET_COLUMN!r} contains missing values.")

    return df


def split_features_target(df: pd.DataFrame):
    """Return X and y using only simple, safe beginner-friendly features."""
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(int)
    return X, y
