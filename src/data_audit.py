"""Dataset quality auditing for the credit-risk dataset."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a CSV and fail with a useful error when it is unavailable/invalid."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {csv_path}")
    try:
        data = pd.read_csv(csv_path)
    except (pd.errors.ParserError, UnicodeDecodeError, OSError) as exc:
        raise ValueError(f"Unable to read dataset CSV: {csv_path}") from exc
    if data.empty:
        raise ValueError(f"Dataset is empty: {csv_path}")
    LOGGER.info("Loaded dataset from %s with shape %s", csv_path, data.shape)
    return data


def identify_target(data: pd.DataFrame, target_column: str | None = None) -> str:
    """Identify an explicit target or a unique conventional binary target."""
    if target_column is not None:
        if target_column not in data.columns:
            raise ValueError(f"Target column not found: {target_column}")
        return target_column

    candidates = [
        column
        for column in data.columns
        if data[column].nunique(dropna=True) == 2
        and set(data[column].dropna().unique()).issubset({0, 1, "0", "1"})
        and ("status" in column.lower() or "target" in column.lower())
    ]
    if len(candidates) != 1:
        raise ValueError(
            "Target column is ambiguous; pass target_column explicitly. "
            f"Candidates: {candidates}"
        )
    return candidates[0]


def _outlier_counts(data: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    for column in data.select_dtypes(include="number").columns:
        series = data[column].dropna()
        if series.empty:
            counts[column] = 0
            continue
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        counts[column] = int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())
    return counts


def audit_dataset(data: pd.DataFrame, target_column: str | None = None) -> dict[str, Any]:
    """Return a serializable, non-mutating data quality report."""
    target = identify_target(data, target_column)
    feature_data = data.drop(columns=target)
    numeric = feature_data.select_dtypes(include="number").columns.tolist()
    categorical = feature_data.select_dtypes(exclude="number").columns.tolist()
    correlations = data[numeric].corr().abs()
    highly_correlated: list[dict[str, Any]] = []
    for index, first in enumerate(numeric):
        for second in numeric[index + 1 :]:
            value = correlations.loc[first, second]
            if value >= 0.8:
                highly_correlated.append(
                    {"feature_1": first, "feature_2": second, "absolute_correlation": round(float(value), 4)}
                )

    suspicious: dict[str, int] = {}
    if "person_age" in data:
        suspicious["person_age_over_100"] = int((data["person_age"] > 100).sum())
    if {"person_emp_length", "person_age"}.issubset(data.columns):
        suspicious["employment_length_over_age"] = int(
            (data["person_emp_length"] > data["person_age"]).sum()
        )
    if "loan_percent_income" in data:
        suspicious["loan_percent_income_out_of_range"] = int(
            ((data["loan_percent_income"] < 0) | (data["loan_percent_income"] > 1)).sum()
        )

    target_counts = data[target].value_counts(dropna=False)
    return {
        "shape": {"rows": int(data.shape[0]), "columns": int(data.shape[1])},
        "columns": data.columns.tolist(),
        "dtypes": {column: str(dtype) for column, dtype in data.dtypes.items()},
        "target_column": target,
        "missing_values": data.isna().sum().astype(int).to_dict(),
        "duplicate_rows": int(data.duplicated().sum()),
        "unique_values": data.nunique(dropna=False).astype(int).to_dict(),
        "numeric_columns": numeric,
        "categorical_columns": categorical,
        "categorical_values": {
            column: feature_data[column].value_counts(dropna=False).to_dict()
            for column in categorical
        },
        "suspicious_values": suspicious,
        "outlier_counts_iqr": _outlier_counts(data),
        "target_distribution": target_counts.astype(int).to_dict(),
        "target_proportions": target_counts.div(len(data)).round(6).to_dict(),
        "highly_correlated_numeric_features": highly_correlated,
    }
