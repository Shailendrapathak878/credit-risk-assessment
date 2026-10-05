"""In-memory access to the development-trained model artifact."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from app.schemas import ApplicantRequest

LOGGER = logging.getLogger(__name__)


class ModelUnavailableError(RuntimeError):
    """Raised when the model artifact cannot be used."""


class ModelService:
    """Load and serve the serialized pipeline without retraining."""

    def __init__(
        self,
        model_path: str | Path = "models/final_credit_risk_model.joblib",
        metadata_path: str | Path = "models/final_model_metadata.json",
    ) -> None:
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.model: Any | None = None
        self.metadata: dict[str, Any] = {}

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        try:
            self.model = joblib.load(self.model_path)
            self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            if not hasattr(self.model, "predict_proba"):
                raise TypeError("Loaded artifact does not expose predict_proba")
            LOGGER.info("Model artifact loaded successfully")
        except Exception as exc:
            self.model = None
            self.metadata = {}
            LOGGER.exception("Model artifact failed to load: %s", type(exc).__name__)
            raise ModelUnavailableError("Model artifact is unavailable") from exc

    def _require_model(self) -> Any:
        if self.model is None:
            raise ModelUnavailableError("Model artifact is unavailable")
        return self.model

    def predict(self, request: ApplicantRequest, threshold: float) -> dict[str, Any]:
        model = self._require_model()
        probability = float(model.predict_proba(pd.DataFrame([request.to_record()]))[0, 1])
        risk_class = int(probability >= threshold)
        return {
            "risk_probability": probability,
            "risk_class": risk_class,
            "risk_label": "High Risk" if risk_class else "Low Risk",
            "threshold": threshold,
            "model_name": "Tuned XGBoost",
            "model": "tuned_xgboost_all_valid",
        }
