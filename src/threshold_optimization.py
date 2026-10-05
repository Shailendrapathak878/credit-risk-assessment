"""Threshold metrics and configurable cost-sensitive analysis."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def validate_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Validate and return one-dimensional probabilities in [0, 1]."""
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Probabilities must be a finite one-dimensional array")
    if ((values < 0) | (values > 1)).any():
        raise ValueError("Probabilities must be within [0, 1]")
    return values


def threshold_metrics(
    y_true: Iterable[int],
    probabilities: np.ndarray,
    thresholds: Iterable[float],
) -> pd.DataFrame:
    """Calculate classification trade-offs for each supplied threshold."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    values = validate_probabilities(probabilities)
    labels = np.asarray(list(y_true))
    rows = []
    for threshold in thresholds:
        if not 0 <= threshold <= 1:
            raise ValueError("Thresholds must be within [0, 1]")
        predictions = (values >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
        rows.append(
            {
                "Threshold": float(threshold),
                "Precision": float(precision_score(labels, predictions, zero_division=0)),
                "Recall": float(recall_score(labels, predictions, zero_division=0)),
                "F1": float(f1_score(labels, predictions, zero_division=0)),
                "Specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
                "False Positive Rate": float(fp / (tn + fp)) if tn + fp else 0.0,
                "False Negative Rate": float(fn / (fn + tp)) if fn + tp else 0.0,
                "Accuracy": float(accuracy_score(labels, predictions)),
                "True Negatives": int(tn),
                "False Positives": int(fp),
                "False Negatives": int(fn),
                "True Positives": int(tp),
            }
        )
    return pd.DataFrame(rows)


def add_expected_cost(
    metrics: pd.DataFrame,
    false_positive_cost: float,
    false_negative_cost: float,
) -> pd.DataFrame:
    """Add expected per-observation cost using rates from threshold metrics."""
    if false_positive_cost < 0 or false_negative_cost < 0:
        raise ValueError("Costs must be non-negative")
    result = metrics.copy()
    if {"False Positives", "False Negatives"}.issubset(result.columns):
        denominator = result[
            ["True Negatives", "False Positives", "False Negatives", "True Positives"]
        ].sum(axis=1)
        result["Expected Cost"] = (
            result["False Positives"] * false_positive_cost
            + result["False Negatives"] * false_negative_cost
        ) / denominator
    else:
        result["Expected Cost"] = (
            result["False Positive Rate"] * false_positive_cost
            + result["False Negative Rate"] * false_negative_cost
        )
    return result


def select_lowest_cost_threshold(
    metrics: pd.DataFrame,
    false_positive_cost: float,
    false_negative_cost: float,
) -> pd.Series:
    """Return the lowest-cost row; this is analysis, not a business decision."""
    scored = add_expected_cost(metrics, false_positive_cost, false_negative_cost)
    return scored.loc[scored["Expected Cost"].idxmin()]
