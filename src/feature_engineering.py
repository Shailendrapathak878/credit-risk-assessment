"""Leakage-safe, domain-motivated feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


ENGINEERED_FEATURES = (
    "income_to_loan_ratio",
    "credit_history_age_ratio",
    "employment_stability_ratio",
)


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add ratios available at application time without using the target."""
    result = data.copy()
    result["income_to_loan_ratio"] = result["person_income"] / result["loan_amnt"].replace(0, np.nan)
    result["credit_history_age_ratio"] = (
        result["cb_person_cred_hist_length"] / result["person_age"].replace(0, np.nan)
    )
    result["employment_stability_ratio"] = (
        result["person_emp_length"] / result["person_age"].replace(0, np.nan)
    )
    return result


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """scikit-learn transformer wrapper for :func:`add_features`."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return add_features(X)
