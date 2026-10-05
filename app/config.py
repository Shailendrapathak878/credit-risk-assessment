"""Environment-backed runtime configuration with safe local defaults."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", PROJECT_ROOT / "models/final_credit_risk_model.joblib"))
METADATA_PATH = Path(os.getenv("METADATA_PATH", PROJECT_ROOT / "models/final_model_metadata.json"))
STATIC_PATH = Path(os.getenv("STATIC_PATH", PROJECT_ROOT / "app/static"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "phase_9")

try:
    THRESHOLD = float(os.getenv("PREDICTION_THRESHOLD", "0.50"))
except ValueError as exc:
    raise ValueError("PREDICTION_THRESHOLD must be a number between 0 and 1") from exc

if not 0 <= THRESHOLD <= 1:
    raise ValueError("PREDICTION_THRESHOLD must be between 0 and 1")
