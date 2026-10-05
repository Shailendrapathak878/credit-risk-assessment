"""Training-only SHAP explainability for the tuned XGBoost pipeline."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap

from .calibration import TUNED_PARAMS
from .data_preprocessing import RANDOM_STATE, prepare_data
from .train_baseline import make_pipeline, make_xgboost

LOGGER = logging.getLogger(__name__)


def build_interpretation_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    excluded_features: tuple[str, ...] = (),
) -> Any:
    """Fit the exact tuned training pipeline used for interpretation."""
    X_scenario = X_train.drop(columns=list(excluded_features), errors="ignore")
    pipeline = make_pipeline(
        X_train,
        make_xgboost(**TUNED_PARAMS),
        excluded_features,
    )
    pipeline.fit(X_scenario, y_train)
    return pipeline


def transform_for_shap(
    pipeline: Any,
    X: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    """Transform records and verify names align with model input columns."""
    engineered = pipeline.named_steps["features"].transform(X)
    transformed = pipeline.named_steps["preprocessor"].transform(engineered)
    names = list(pipeline.named_steps["preprocessor"].get_feature_names_out())
    if transformed.shape[1] != len(names):
        raise RuntimeError(
            f"Feature-name mismatch: {len(names)} names for {transformed.shape[1]} columns"
        )
    return np.asarray(transformed), names


def get_transformed_feature_names(pipeline: Any) -> list[str]:
    """Return the exact feature names supplied to XGBoost."""
    names = list(pipeline.named_steps["preprocessor"].get_feature_names_out())
    if not names:
        raise ValueError("Pipeline has no transformed feature names")
    return names


def calculate_shap_values(
    pipeline: Any,
    X: pd.DataFrame,
) -> tuple[np.ndarray, float, np.ndarray, list[str]]:
    """Calculate TreeExplainer raw-margin SHAP values.

    Values are additive in the model's raw margin (log-odds) output space,
    not direct probability changes.
    """
    transformed, names = transform_for_shap(pipeline, X)
    model = pipeline.named_steps["model"]
    explainer = shap.TreeExplainer(model, model_output="raw")
    explanation = explainer(transformed)
    values = np.asarray(explanation.values)
    if values.ndim == 3:
        values = values[:, :, 1]
    base = np.asarray(explanation.base_values)
    base_value = float(base.reshape(-1)[0])
    if values.shape != transformed.shape or not np.isfinite(values).all():
        raise RuntimeError("SHAP values do not align with finite transformed input")
    return values, base_value, transformed, names


def create_global_feature_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """Create sorted mean absolute SHAP importance for transformed features."""
    if shap_values.shape[1] != len(feature_names):
        raise ValueError("SHAP feature count does not match feature-name count")
    result = pd.DataFrame(
        {
            "feature_name": feature_names,
            "mean_absolute_shap": np.abs(shap_values).mean(axis=0),
        }
    )
    return result.sort_values("mean_absolute_shap", ascending=False).reset_index(drop=True)


def _logical_feature(name: str) -> str:
    clean = re.sub(r"^(numerical|categorical)__", "", name)
    for prefix in (
        "person_home_ownership",
        "loan_intent",
        "loan_grade",
        "cb_person_default_on_file",
    ):
        if clean.startswith(f"{prefix}_"):
            return prefix
    return clean


def create_grouped_feature_importance(importance: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one-hot columns by their original logical feature."""
    grouped = importance.assign(
        logical_feature=importance["feature_name"].map(_logical_feature)
    )
    return (
        grouped.groupby("logical_feature", as_index=False)["mean_absolute_shap"]
        .sum()
        .sort_values("mean_absolute_shap", ascending=False)
        .reset_index(drop=True)
    )


def create_shap_summary_plot(
    shap_values: np.ndarray,
    transformed: np.ndarray,
    feature_names: list[str],
    path: str | Path,
    plot_type: str = "dot",
) -> None:
    """Save a SHAP summary plot without displaying it."""
    import matplotlib.pyplot as plt

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 9))
    shap.summary_plot(
        shap_values,
        transformed,
        feature_names=feature_names,
        plot_type=plot_type,
        show=False,
        max_display=20,
    )
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()


def create_shap_dependence_plot(
    shap_values: np.ndarray,
    transformed: np.ndarray,
    feature_names: list[str],
    feature_name: str,
    path: str | Path,
) -> None:
    """Save a dependence plot for one transformed numeric feature."""
    import matplotlib.pyplot as plt

    if feature_name not in feature_names:
        raise ValueError(f"Unknown transformed feature: {feature_name}")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    shap.dependence_plot(
        feature_names.index(feature_name),
        shap_values,
        transformed,
        feature_names=feature_names,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()


def explain_single_prediction(
    pipeline: Any,
    applicant: pd.DataFrame,
    threshold: float = 0.5,
    top_n: int = 5,
) -> dict[str, Any]:
    """Explain one applicant using raw-margin SHAP contributions."""
    if len(applicant) != 1:
        raise ValueError("applicant must contain exactly one row")
    values, base_value, transformed, names = calculate_shap_values(pipeline, applicant)
    model = pipeline.named_steps["model"]
    probability = float(pipeline.predict_proba(applicant)[:, 1][0])
    contributions = values[0]
    raw_output = float(model.predict(transformed, output_margin=True)[0])
    if not np.isclose(base_value + contributions.sum(), raw_output, atol=1e-4):
        raise RuntimeError("SHAP additivity check failed in raw-margin output space")
    entries = []
    for index, contribution in enumerate(contributions):
        entries.append(
            {
                "feature": names[index],
                "feature_value": float(transformed[0, index]),
                "shap_value": float(contribution),
                "direction": (
                    "increases default risk"
                    if contribution > 0
                    else "decreases default risk"
                ),
            }
        )
    positive = sorted(
        [entry for entry in entries if entry["shap_value"] > 0],
        key=lambda item: item["shap_value"],
        reverse=True,
    )[:top_n]
    negative = sorted(
        [entry for entry in entries if entry["shap_value"] < 0],
        key=lambda item: item["shap_value"],
    )[:top_n]
    return {
        "predicted_probability": probability,
        "predicted_class": int(probability >= threshold),
        "threshold": threshold,
        "base_value": base_value,
        "output_space": "raw_margin_log_odds",
        "positive_contributors": positive,
        "negative_contributors": negative,
    }


def generate_shap_artifacts(
    dataset_path: str | Path = "data/credit_risk_dataset.csv",
    sample_size: int = 3000,
    random_state: int = RANDOM_STATE,
    output_dir: str | Path = "reports",
) -> dict[str, Any]:
    """Fit on training data and generate all requested SHAP artifacts."""
    parts = prepare_data(dataset_path)
    pipeline = build_interpretation_model(parts["X_train"], parts["y_train"])
    X_train = parts["X_train"]
    sample = X_train.sample(
        n=min(sample_size, len(X_train)), random_state=random_state
    )
    values, base_value, transformed, names = calculate_shap_values(pipeline, sample)
    importance = create_global_feature_importance(values, names)
    grouped = create_grouped_feature_importance(importance)
    reports = Path(output_dir)
    figures = reports / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    importance.to_csv(reports / "shap_feature_importance.csv", index=False)
    grouped.to_csv(reports / "shap_grouped_feature_importance.csv", index=False)
    create_shap_summary_plot(
        values, transformed, names, figures / "shap_summary_beeswarm.png"
    )
    create_shap_summary_plot(
        values, transformed, names, figures / "shap_summary_bar.png", plot_type="bar"
    )

    numeric_names = [
        name
        for name in names
        if name.startswith("numerical__")
        and not name.endswith("loan_status")
    ]
    top_numeric = [
        name
        for name in importance["feature_name"]
        if name in numeric_names
    ][:5]
    for feature_name in top_numeric:
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", feature_name.replace("__", "_"))
        create_shap_dependence_plot(
            values,
            transformed,
            names,
            feature_name,
            figures / f"shap_dependence_{safe_name}.png",
        )

    train_probabilities = pipeline.predict_proba(X_train)[:, 1]
    high_index = int(np.argmax(train_probabilities))
    low_index = int(np.argmin(train_probabilities))
    high = explain_single_prediction(pipeline, X_train.iloc[[high_index]])
    low = explain_single_prediction(pipeline, X_train.iloc[[low_index]])
    (reports / "example_high_risk_explanation.json").write_text(
        json.dumps(high, indent=2), encoding="utf-8"
    )
    (reports / "example_low_risk_explanation.json").write_text(
        json.dumps(low, indent=2), encoding="utf-8"
    )
    return {
        "shap_version": shap.__version__,
        "transformed_feature_count": len(names),
        "sample_count": len(sample),
        "top_features": importance.head(15),
        "top_dependence_features": top_numeric,
        "high_probability": high["predicted_probability"],
        "low_probability": low["predicted_probability"],
        "base_value": base_value,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    result = generate_shap_artifacts()
    print(f"SHAP version: {result['shap_version']}")
    print(f"Transformed features: {result['transformed_feature_count']}")
    print(f"SHAP samples: {result['sample_count']}")
    print("Top 15 features:")
    print(result["top_features"].to_string(index=False))
    print(f"Top dependence features: {result['top_dependence_features']}")
    print(f"High-risk probability: {result['high_probability']:.8f}")
    print(f"Low-risk probability: {result['low_probability']:.8f}")
