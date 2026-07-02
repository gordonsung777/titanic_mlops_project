from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
CATEGORICAL_FEATURES = ["Sex", "Embarked"]


def build_model(C: float = 1.0, max_iter: int = 1000, random_state: int = 42) -> Pipeline:
    """Build a preprocessing + model pipeline.

    Why a Pipeline?
    - It keeps preprocessing and the model together.
    - The same cleaning steps are used during training and prediction.
    - It prevents accidental differences between notebook code and production code.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    model = LogisticRegression(C=C, max_iter=max_iter, random_state=random_state)

    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("model", model),
        ]
    )
