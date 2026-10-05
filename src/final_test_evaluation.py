"""One-time final evaluation on the untouched held-out test split."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .calibration import TUNED_PARAMS
from .data_preprocessing import prepare_data
from .train_baseline import make_pipeline, make_xgboost

LOGGER = logging.getLogger(__name__)
EVALUATION_THRESHOLD = 0.50
PRE_DECISION_PARAMS = {
    "subsample": 0.75,
    "reg_lambda": 1.0,
    "reg_alpha": 0.01,
    "n_estimators": 450,
    "min_child_weight": 5,
    "max_depth": 5,
    "learning_rate": 0.12,
    "gamma": 0.5,
    "colsample_bytree": 1.0,
}


def candidate_definitions() -> dict[str, tuple[str, Any, tuple[str, ...]]]:
    """Return the fixed candidates chosen before test evaluation."""
    return {
        "Tuned XGBoost - All Valid": ("all_valid_features", make_xgboost(**TUNED_PARAMS), ()),
        "Tuned XGBoost - Pre-Decision": (
            "pre_decision_only",
            make_xgboost(**PRE_DECISION_PARAMS),
            ("loan_grade", "loan_int_rate"),
        ),
        "Random Forest": (
            "all_valid_features",
            RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
            (),
        ),
        "Baseline XGBoost": (
            "all_valid_features",
            make_xgboost(),
            (),
        ),
    }


def evaluate_candidate(
    name: str,
    feature_set: str,
    estimator: Any,
    excluded_features: tuple[str, ...],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = EVALUATION_THRESHOLD,
) -> tuple[dict[str, Any], Any, str]:
    """Fit one candidate on development data and evaluate once on test data."""
    X_train_scenario = X_train.drop(columns=list(excluded_features), errors="ignore")
    X_test_scenario = X_test.drop(columns=list(excluded_features), errors="ignore")
    pipeline = make_pipeline(X_train, estimator, excluded_features)
    pipeline.fit(X_train_scenario, y_train)
    probabilities = pipeline.predict_proba(X_test_scenario)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    row = {
        "model": name,
        "feature_set": feature_set,
        "threshold": threshold,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "pr_auc": average_precision_score(y_test, probabilities),
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "fpr": fp / (tn + fp) if tn + fp else 0.0,
        "fnr": fn / (fn + tp) if fn + tp else 0.0,
        "brier_score": brier_score_loss(y_test, probabilities),
        "log_loss": log_loss(y_test, probabilities, labels=[0, 1]),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }
    report = classification_report(y_test, predictions, labels=[0, 1], zero_division=0)
    return row, pipeline, report


def _save_confusion_matrix(row: dict[str, Any], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(5, 4))
    matrix = [[row["tn"], row["fp"]], [row["fn"], row["tp"]]]
    axis.imshow(matrix, cmap="Blues")
    axis.set_xticks([0, 1], ["Predicted 0", "Predicted 1"])
    axis.set_yticks([0, 1], ["Actual 0", "Actual 1"])
    axis.set_title(f"{row['model']} (test set)")
    for i in range(2):
        for j in range(2):
            axis.text(j, i, matrix[i][j], ha="center", va="center")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def run_final_evaluation(
    dataset_path: str | Path = "data/credit_risk_dataset.csv",
    output_dir: str | Path = "reports",
) -> pd.DataFrame:
    """Run final test evaluation; no test result is used to alter candidates."""
    parts = prepare_data(dataset_path)
    reports = Path(output_dir)
    figures = reports / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    rows = []
    reports_text = [
        "# Final Model Comparison",
        "",
        "The 6,484-row held-out test set was used exactly once for final evaluation.",
        "Candidates, preprocessing, features, threshold, and calibration policy were fixed beforehand.",
        "",
    ]
    roc_data = []
    pr_data = []
    for name, (feature_set, estimator, excluded) in candidate_definitions().items():
        row, pipeline, report = evaluate_candidate(
            name,
            feature_set,
            estimator,
            excluded,
            parts["X_train"],
            parts["y_train"],
            parts["X_test"],
            parts["y_test"],
        )
        rows.append(row)
        safe_name = {
            "Tuned XGBoost - All Valid": "tuned_xgboost_all_valid",
            "Tuned XGBoost - Pre-Decision": "tuned_xgboost_pre_decision",
            "Random Forest": "random_forest",
            "Baseline XGBoost": "baseline_xgboost",
        }[name]
        _save_confusion_matrix(row, figures / f"final_test_confusion_matrix_{safe_name}.png")
        probabilities = pipeline.predict_proba(
            parts["X_test"].drop(columns=list(excluded), errors="ignore")
        )[:, 1]
        fpr, tpr, _ = roc_curve(parts["y_test"], probabilities)
        precision, recall, _ = precision_recall_curve(parts["y_test"], probabilities)
        roc_data.append((name, fpr, tpr))
        pr_data.append((name, recall, precision))
        reports_text.extend(
            [f"## {name}", "", "```text", report, "```", ""]
        )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(reports / "final_test_metrics.csv", index=False)
    figure, axis = plt.subplots(figsize=(8, 6))
    for name, fpr, tpr in roc_data:
        row = metrics.loc[metrics["model"] == name].iloc[0]
        axis.plot(fpr, tpr, label=f"{name} (AUC={row['roc_auc']:.4f})")
    axis.plot([0, 1], [0, 1], "--", color="gray")
    axis.set(xlabel="False positive rate", ylabel="True positive rate", title="Final test ROC curves")
    axis.legend()
    figure.tight_layout()
    figure.savefig(figures / "final_test_roc_curves.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 6))
    for name, recall, precision in pr_data:
        row = metrics.loc[metrics["model"] == name].iloc[0]
        axis.plot(recall, precision, label=f"{name} (AP={row['pr_auc']:.4f})")
    axis.set(xlabel="Recall", ylabel="Precision", title="Final test precision-recall curves")
    axis.legend()
    figure.tight_layout()
    figure.savefig(figures / "final_test_pr_curves.png", dpi=150)
    plt.close(figure)

    winner = metrics.sort_values(["pr_auc", "roc_auc", "f1"], ascending=False).iloc[0]
    reports_text.extend(
        [
            "## Final metrics",
            "",
            "```text",
            metrics.to_string(index=False),
            "```",
            "",
            f"Predictive winner by PR-AUC, ROC-AUC, and F1 tie-break: **{winner['model']}**.",
            "The all-valid winner is not automatically production-safe because grade and interest rate availability requires domain validation.",
            "Calibration: uncalibrated probabilities, as selected using training-only Phase 4 OOF results.",
            "Threshold: 0.50 evaluation threshold, not a business-approved production threshold.",
            "",
            "The held-out test set was not used for tuning, feature selection, threshold selection, calibration selection, or retraining.",
        ]
    )
    (reports / "final_model_comparison.md").write_text("\n".join(reports_text), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(run_final_evaluation().to_string(index=False))
