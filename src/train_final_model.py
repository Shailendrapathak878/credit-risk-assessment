"""Prepare the development-trained model artifact for the next API phase."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn
import xgboost

from .calibration import TUNED_PARAMS
from .data_preprocessing import prepare_data
from .final_test_evaluation import EVALUATION_THRESHOLD
from .train_baseline import make_pipeline, make_xgboost


def train_and_save_final_model(
    dataset_path: str | Path = "data/credit_risk_dataset.csv",
    model_path: str | Path = "models/final_credit_risk_model.joblib",
    metadata_path: str | Path = "models/final_model_metadata.json",
) -> dict:
    """Fit the selected all-feature model on development data only."""
    parts = prepare_data(dataset_path)
    model = make_pipeline(parts["X_train"], make_xgboost(**TUNED_PARAMS))
    model.fit(parts["X_train"], parts["y_train"])
    output_model = Path(model_path)
    output_metadata = Path(metadata_path)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_model)
    metadata = {
        "model_name": "Tuned XGBoost",
        "feature_set": "all_valid_features",
        "hyperparameters": TUNED_PARAMS,
        "training_rows": len(parts["X_train"]),
        "test_rows": len(parts["X_test"]),
        "random_state": 42,
        "threshold": EVALUATION_THRESHOLD,
        "calibration_method": "uncalibrated",
        "feature_names": list(parts["X_train"].columns),
        "transformed_feature_names": list(
            model.named_steps["preprocessor"].get_feature_names_out()
        ),
        "package_versions": {
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "joblib": joblib.__version__,
        },
        "training_date_utc": datetime.now(timezone.utc).isoformat(),
        "test_metrics_reference": "reports/final_test_metrics.csv",
        "feature_availability_caveat": (
            "loan_grade and loan_int_rate require business validation before deployment."
        ),
        "known_limitations": [
            "The artifact is development-trained and has not been retrained on the held-out test set.",
            "Threshold 0.50 is an evaluation threshold, not a business-approved threshold.",
        ],
    }
    output_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    print(json.dumps(train_and_save_final_model(), indent=2))
