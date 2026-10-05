"""Cross-validated baseline models for Phase 2 experimentation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from .data_preprocessing import RANDOM_STATE, build_preprocessor, prepare_data
from .feature_engineering import FeatureEngineer, add_features

LOGGER = logging.getLogger(__name__)
SCENARIOS = {
    "all_valid_features": (),
    "pre_decision_only": ("loan_grade", "loan_int_rate"),
}


def calculate_scale_pos_weight(y: pd.Series) -> float:
    """Return the negative-to-positive ratio calculated from training labels."""
    counts = y.value_counts()
    if set(counts.index) != {0, 1} or counts[1] == 0:
        raise ValueError("scale_pos_weight requires binary labels containing both 0 and 1")
    return float(counts[0] / counts[1])


def make_xgboost(
    scale_pos_weight: float = 1.0,
    **overrides: Any,
) -> ClassifierMixin:
    """Create a reproducible XGBoost binary classifier."""
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("XGBoost is required for XGBoost experiments") from exc
    parameters: dict[str, Any] = {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.08,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "scale_pos_weight": scale_pos_weight,
    }
    parameters.update(overrides)
    return XGBClassifier(**parameters)


def make_pipeline(
    X_train: pd.DataFrame,
    estimator: ClassifierMixin,
    excluded_features: tuple[str, ...] = (),
) -> Pipeline:
    """Create a pipeline whose every learned step is fitted inside each fold."""
    engineered = add_features(X_train).drop(columns=list(excluded_features), errors="ignore")
    numerical = engineered.select_dtypes(include="number").columns.tolist()
    categorical = engineered.select_dtypes(exclude="number").columns.tolist()
    return Pipeline(
        steps=[
            ("features", FeatureEngineer()),
            ("preprocessor", build_preprocessor(numerical, categorical)),
            ("model", estimator),
        ]
    )


def model_definitions(y: pd.Series | None = None) -> dict[str, ClassifierMixin]:
    """Return intentionally small, interpretable baseline candidates."""
    models: dict[str, ClassifierMixin] = {
        "DummyClassifier": DummyClassifier(strategy="prior", random_state=RANDOM_STATE),
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "LogisticRegression_balanced": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "RandomForest_balanced": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }
    try:
        models["XGBoost"] = make_xgboost()
        models["XGBoost_balanced"] = make_xgboost(
            calculate_scale_pos_weight(y) if y is not None else 1.0
        )
    except ImportError:
        LOGGER.info("XGBoost is not installed; skipping optional model.")
    return models


def evaluate_model(
    name: str,
    estimator: ClassifierMixin,
    X: pd.DataFrame,
    y: pd.Series,
    excluded_features: tuple[str, ...] = (),
    cv_splits: int = 5,
) -> tuple[dict[str, Any], list[list[int]]]:
    """Evaluate one model using only stratified training folds."""
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    values: dict[str, list[float]] = {
        key: [] for key in ("Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC")
    }
    matrices: list[list[int]] = []
    for train_index, validation_index in cv.split(X, y):
        pipeline = make_pipeline(X.iloc[train_index], estimator, excluded_features)
        X_fold = X.drop(columns=list(excluded_features), errors="ignore")
        pipeline.fit(X_fold.iloc[train_index], y.iloc[train_index])
        predictions = pipeline.predict(X_fold.iloc[validation_index])
        probabilities = pipeline.predict_proba(X_fold.iloc[validation_index])[:, 1]
        values["Accuracy"].append(accuracy_score(y.iloc[validation_index], predictions))
        values["Precision"].append(precision_score(y.iloc[validation_index], predictions, zero_division=0))
        values["Recall"].append(recall_score(y.iloc[validation_index], predictions, zero_division=0))
        values["F1"].append(f1_score(y.iloc[validation_index], predictions, zero_division=0))
        values["ROC-AUC"].append(roc_auc_score(y.iloc[validation_index], probabilities))
        values["PR-AUC"].append(average_precision_score(y.iloc[validation_index], probabilities))
        matrices.append(confusion_matrix(y.iloc[validation_index], predictions).tolist())
    return (
        {
            "Model": name,
            **{
                key: round(sum(scores) / len(scores), 4)
                for key, scores in values.items()
            },
            **{
                f"{key}_std": round(float(pd.Series(scores).std(ddof=1)), 4)
                for key, scores in values.items()
            },
        },
        matrices,
    )


def compare_models(
    X: pd.DataFrame,
    y: pd.Series,
    scenario: str = "all_valid_features",
    cv_splits: int = 5,
) -> tuple[pd.DataFrame, dict[str, list[list[int]]]]:
    """Return mean cross-validation metrics and fold confusion matrices."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    rows = []
    matrices = {}
    for name, model in model_definitions(y).items():
        row, model_matrices = evaluate_model(
            name, model, X, y, SCENARIOS[scenario], cv_splits
        )
        rows.append(row)
        matrices[name] = model_matrices
        LOGGER.info("Completed %s in scenario %s", name, scenario)
    return pd.DataFrame(rows), matrices


def run_baselines(
    dataset_path: str | Path = "data/credit_risk_dataset.csv",
    scenario: str = "all_valid_features",
    cv_splits: int = 5,
) -> tuple[pd.DataFrame, dict[str, list[list[int]]]]:
    """Prepare the Phase 1 split and evaluate models on its training data only."""
    parts = prepare_data(dataset_path)
    return compare_models(parts["X_train"], parts["y_train"], scenario, cv_splits)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    comparison, _ = run_baselines()
    print(comparison.to_string(index=False))
