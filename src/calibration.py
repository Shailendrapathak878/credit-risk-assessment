"""Nested out-of-fold probability calibration for the tuned XGBoost model."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from .data_preprocessing import RANDOM_STATE, prepare_data
from .train_baseline import SCENARIOS, make_pipeline, make_xgboost
from .threshold_optimization import threshold_metrics as calculate_threshold_metrics

LOGGER = logging.getLogger(__name__)
TUNED_PARAMS: dict[str, Any] = {
    "subsample": 0.9,
    "reg_lambda": 2.0,
    "reg_alpha": 0.1,
    "n_estimators": 150,
    "min_child_weight": 1,
    "max_depth": 6,
    "learning_rate": 0.12,
    "gamma": 0.5,
    "colsample_bytree": 0.9,
}
CALIBRATION_METHODS = ("uncalibrated", "sigmoid", "isotonic")


def _new_model() -> Any:
    return make_xgboost(**TUNED_PARAMS)


def _fit_outer_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    method: str,
    calibration_cv: int,
) -> Any:
    base = make_pipeline(X_train, _new_model())
    if method == "uncalibrated":
        return base
    if method not in {"sigmoid", "isotonic"}:
        raise ValueError(f"Unknown calibration method: {method}")
    return CalibratedClassifierCV(
        estimator=base,
        method=method,
        cv=calibration_cv,
        n_jobs=-1,
    )


def out_of_fold_calibration_predictions(
    X: pd.DataFrame,
    y: pd.Series,
    scenario: str = "all_valid_features",
    outer_splits: int = 5,
    calibration_cv: int = 3,
) -> dict[str, np.ndarray]:
    """Return nested OOF probabilities for all calibration methods.

    The outer validation row is never used to fit its base model or calibrator.
    The inner calibration CV is fitted only within the outer training fold.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    excluded = SCENARIOS[scenario]
    X_scenario = X.drop(columns=list(excluded), errors="ignore")
    predictions = {
        method: np.full(len(y), np.nan, dtype=float) for method in CALIBRATION_METHODS
    }
    outer = StratifiedKFold(
        n_splits=outer_splits, shuffle=True, random_state=RANDOM_STATE
    )
    for train_index, validation_index in outer.split(X_scenario, y):
        X_outer_train = X_scenario.iloc[train_index]
        X_outer_validation = X_scenario.iloc[validation_index]
        y_outer_train = y.iloc[train_index]
        for method in CALIBRATION_METHODS:
            fitted = _fit_outer_model(
                X_outer_train, y_outer_train, method, calibration_cv
            )
            fitted.fit(X_outer_train, y_outer_train)
            predictions[method][validation_index] = fitted.predict_proba(
                X_outer_validation
            )[:, 1]
    for method, values in predictions.items():
        if np.isnan(values).any():
            raise RuntimeError(f"OOF calibration predictions incomplete for {method}")
        if ((values < 0) | (values > 1)).any():
            raise RuntimeError(f"OOF probabilities out of range for {method}")
    return predictions


def calibration_metrics(
    y: pd.Series | np.ndarray,
    predictions: dict[str, np.ndarray],
    thresholds: tuple[float, ...],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Calculate probability metrics and threshold summaries for OOF outputs."""
    labels = np.asarray(y)
    rows = []
    threshold_tables = {}
    for method, probabilities in predictions.items():
        rows.append(
            {
                "Model": method,
                "Brier Score": brier_score_loss(labels, probabilities),
                "Log Loss": log_loss(labels, probabilities, labels=[0, 1]),
                "ROC-AUC": roc_auc_score(labels, probabilities),
                "PR-AUC": average_precision_score(labels, probabilities),
            }
        )
        threshold_tables[method] = calculate_threshold_metrics(
            labels, probabilities, thresholds
        )
        best = threshold_tables[method].loc[
            threshold_tables[method]["F1"].idxmax()
        ]
        rows[-1]["Best F1"] = best["F1"]
        rows[-1]["Best F1 Threshold"] = best["Threshold"]
    return pd.DataFrame(rows), threshold_tables


def reliability_table(
    y: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Build a quantile-binned reliability table from OOF probabilities."""
    labels = np.asarray(y)
    bins = pd.qcut(probabilities, q=n_bins, duplicates="drop")
    frame = pd.DataFrame(
        {"Probability bin": bins, "probability": probabilities, "default": labels}
    )
    return (
        frame.groupby("Probability bin", observed=True)
        .agg(
            mean_predicted_probability=("probability", "mean"),
            observed_default_rate=("default", "mean"),
            number_of_samples=("default", "size"),
        )
        .reset_index()
    )


def save_calibration_curve(
    y: pd.Series | np.ndarray,
    predictions: dict[str, np.ndarray],
    path: str | Path,
    n_bins: int = 10,
) -> None:
    """Save a training-OOF calibration curve."""
    figure_path = Path(path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], "--", label="Perfect calibration")
    for method, probabilities in predictions.items():
        observed, predicted = calibration_curve(
            y, probabilities, n_bins=n_bins, strategy="quantile"
        )
        plt.plot(predicted, observed, marker="o", label=method)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed default rate")
    plt.title("Training OOF probability calibration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=150)
    plt.close()


def run_calibration_reports(
    dataset_path: str | Path = "data/credit_risk_dataset.csv",
    scenario: str = "all_valid_features",
    thresholds: tuple[float, ...] | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Generate Phase 4 reports from the Phase 1 training split only."""
    parts = prepare_data(dataset_path)
    predictions = out_of_fold_calibration_predictions(
        parts["X_train"], parts["y_train"], scenario
    )
    thresholds = thresholds or tuple(round(x, 2) for x in np.arange(0.10, 0.71, 0.05))
    metrics, tables = calibration_metrics(parts["y_train"], predictions, thresholds)
    best_method = metrics.sort_values(["Brier Score", "Log Loss"]).iloc[0]["Model"]
    output_dir = Path("reports")
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    save_calibration_curve(
        parts["y_train"],
        predictions,
        output_dir / "figures" / "calibration_curve.png",
    )
    reliability_table(
        parts["y_train"], predictions[best_method]
    ).to_csv(output_dir / "calibration_reliability.csv", index=False)
    tables[best_method].to_csv(output_dir / "threshold_comparison.csv", index=False)
    metrics.to_csv(output_dir / "calibration_metrics.csv", index=False)
    LOGGER.info("Best calibration method by Brier/Log Loss: %s", best_method)
    return metrics, tables
