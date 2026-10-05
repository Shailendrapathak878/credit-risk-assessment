"""SHAP explanation service for one applicant at a time."""

from __future__ import annotations

import logging
import re
from typing import Any

import pandas as pd
import shap

from app.schemas import ApplicantRequest
from src.explainability import calculate_shap_values

LOGGER = logging.getLogger(__name__)


def _readable_feature(name: str) -> str:
    clean = re.sub(r"^(numerical|categorical)__", "", name)
    categorical_prefixes = (
        "person_home_ownership",
        "loan_intent",
        "loan_grade",
        "cb_person_default_on_file",
    )
    for prefix in categorical_prefixes:
        if clean.startswith(f"{prefix}_"):
            return f"{prefix} = {clean[len(prefix) + 1:]}"
    return clean


class ExplanationService:
    """Reuse a TreeExplainer-compatible model pipeline for explanations."""

    def __init__(self, model_service: Any) -> None:
        self.model_service = model_service
        self.explainer = None
        if model_service.model is not None:
            model = model_service.model.named_steps["model"]
            self.explainer = shap.TreeExplainer(model, model_output="raw")

    def explain(
        self,
        request: ApplicantRequest,
        threshold: float,
        top_n: int = 5,
    ) -> dict[str, Any]:
        if self.model_service.model is None or self.explainer is None:
            raise RuntimeError("Explanation model is unavailable")
        pipeline = self.model_service.model
        record = pd.DataFrame([request.to_record()])
        values, base_value, transformed, names = calculate_shap_values(pipeline, record)
        model = pipeline.named_steps["model"]
        raw_output = float(model.predict(transformed, output_margin=True)[0])
        contributions = values[0]
        if abs(base_value + contributions.sum() - raw_output) > 1e-4:
            raise RuntimeError("SHAP additivity validation failed")
        probability = float(pipeline.predict_proba(record)[0, 1])
        entries = [
            {
                "feature": _readable_feature(name),
                "value": float(transformed[0, index]),
                "shap_value": float(contribution),
                "direction": "increases_risk" if contribution > 0 else "decreases_risk",
            }
            for index, (name, contribution) in enumerate(zip(names, contributions))
            if contribution != 0
        ]
        positives = sorted(
            [item for item in entries if item["shap_value"] > 0],
            key=lambda item: abs(item["shap_value"]),
            reverse=True,
        )[:top_n]
        negatives = sorted(
            [item for item in entries if item["shap_value"] < 0],
            key=lambda item: abs(item["shap_value"]),
            reverse=True,
        )[:top_n]
        risk_class = int(probability >= threshold)
        return {
            "risk_probability": probability,
            "risk_class": risk_class,
            "risk_label": "High Risk" if risk_class else "Low Risk",
            "threshold": threshold,
            "model_name": "Tuned XGBoost",
            "model": "tuned_xgboost_all_valid",
            "base_value": base_value,
            "explanation_space": "raw_margin_log_odds",
            "top_positive_factors": positives,
            "top_negative_factors": negatives,
        }
