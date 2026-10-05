"""Training-only XGBoost tuning and out-of-fold threshold analysis."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from .data_preprocessing import RANDOM_STATE
from .train_baseline import (
    SCENARIOS,
    calculate_scale_pos_weight,
    make_pipeline,
    make_xgboost,
)

LOGGER = logging.getLogger(__name__)
THRESHOLDS = tuple(round(value, 2) for value in np.arange(0.10, 0.71, 0.05))


def xgboost_search_space() -> dict[str, list[Any]]:
    """A bounded search space suitable for a first PR-AUC optimization."""
    return {
        "model__n_estimators": [150, 250, 350, 450],
        "model__max_depth": [3, 4, 5, 6],
        "model__learning_rate": [0.03, 0.05, 0.08, 0.12],
        "model__subsample": [0.75, 0.9, 1.0],
        "model__colsample_bytree": [0.75, 0.9, 1.0],
        "model__min_child_weight": [1, 3, 5, 8],
        "model__gamma": [0.0, 0.1, 0.3, 0.5],
        "model__reg_alpha": [0.0, 0.01, 0.1],
        "model__reg_lambda": [1.0, 2.0, 5.0],
    }


def tune_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    scenario: str = "all_valid_features",
    n_iter: int = 8,
    cv_splits: int = 5,
) -> RandomizedSearchCV:
    """Tune XGBoost using only training data and average precision."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    estimator = make_xgboost(calculate_scale_pos_weight(y))
    pipeline = make_pipeline(X, estimator, SCENARIOS[scenario])
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=xgboost_search_space(),
        n_iter=n_iter,
        scoring="average_precision",
        cv=StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
    )
    X_scenario = X.drop(columns=list(SCENARIOS[scenario]), errors="ignore")
    search.fit(X_scenario, y)
    LOGGER.info(
        "Tuned scenario %s: PR-AUC=%.4f, params=%s",
        scenario,
        search.best_score_,
        search.best_params_,
    )
    return search


def out_of_fold_probabilities(
    X: pd.DataFrame,
    y: pd.Series,
    estimator: Any,
    excluded_features: tuple[str, ...] = (),
    cv_splits: int = 5,
) -> np.ndarray:
    """Generate one validation probability per row using a non-training fold."""
    probabilities = np.full(len(y), np.nan, dtype=float)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    X_scenario = X.drop(columns=list(excluded_features), errors="ignore")
    for train_index, validation_index in cv.split(X_scenario, y):
        pipeline = make_pipeline(X, estimator, excluded_features)
        pipeline.fit(X_scenario.iloc[train_index], y.iloc[train_index])
        probabilities[validation_index] = pipeline.predict_proba(
            X_scenario.iloc[validation_index]
        )[:, 1]
    if np.isnan(probabilities).any():
        raise RuntimeError("Out-of-fold prediction did not cover every training row")
    return probabilities


def threshold_metrics(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    thresholds: tuple[float, ...] = THRESHOLDS,
) -> pd.DataFrame:
    """Calculate threshold trade-offs without selecting a business threshold."""
    rows = []
    y_array = np.asarray(y_true)
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_array, predictions, labels=[0, 1]).ravel()
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_array, predictions, average="binary", zero_division=0
        )
        rows.append(
            {
                "Threshold": threshold,
                "Precision": round(float(precision), 4),
                "Recall": round(float(recall), 4),
                "F1": round(float(f1), 4),
                "False Positive Rate": round(float(fp / (fp + tn)), 4),
                "False Negative Rate": round(float(fn / (fn + tp)), 4),
            }
        )
    return pd.DataFrame(rows)
