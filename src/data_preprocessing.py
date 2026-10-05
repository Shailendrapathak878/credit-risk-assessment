"""Leakage-safe splitting and preprocessing for credit-risk classification."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data_audit import identify_target, load_dataset

LOGGER = logging.getLogger(__name__)
RANDOM_STATE = 42


def build_preprocessor(
    numerical_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    """Build transformations that can be fitted only on training data."""
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, numerical_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def prepare_data(
    path: str | Path,
    target_column: str | None = None,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Deduplicate, identify the target, and split before fitting preprocessing."""
    data = load_dataset(path)
    target = identify_target(data, target_column)
    duplicate_count = int(data.duplicated().sum())
    if duplicate_count:
        data = data.drop_duplicates().reset_index(drop=True)
        LOGGER.info("Removed %d exact duplicate rows before splitting", duplicate_count)

    if data[target].isna().any():
        raise ValueError(f"Target column contains missing values: {target}")
    if data[target].nunique() != 2:
        raise ValueError(f"Target must be binary; found {data[target].nunique()} classes")

    X = data.drop(columns=target)
    y = data[target]
    numerical_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(exclude="number").columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    preprocessor = build_preprocessor(numerical_features, categorical_features)
    LOGGER.info(
        "Prepared split: train=%d rows, test=%d rows, target=%s",
        len(X_train),
        len(X_test),
        target,
    )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "target_column": target,
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "removed_duplicate_rows": duplicate_count,
    }
