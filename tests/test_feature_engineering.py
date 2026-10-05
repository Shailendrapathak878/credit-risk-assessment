import pandas as pd
import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from src.feature_engineering import ENGINEERED_FEATURES, FeatureEngineer, add_features
from src.train_baseline import calculate_scale_pos_weight, make_pipeline, make_xgboost
from src.tune_models import (
    THRESHOLDS,
    out_of_fold_probabilities,
    threshold_metrics,
    xgboost_search_space,
)
from src.calibration import calibration_metrics, reliability_table
from src.explainability import (
    calculate_shap_values,
    create_global_feature_importance,
    explain_single_prediction,
    get_transformed_feature_names,
)
from src.final_test_evaluation import candidate_definitions
from src.threshold_optimization import (
    add_expected_cost,
    select_lowest_cost_threshold,
    threshold_metrics as cost_threshold_metrics,
    validate_probabilities,
)


def test_add_features_are_domain_ratios_and_do_not_change_input():
    data = pd.DataFrame(
        {
            "person_income": [100_000],
            "loan_amnt": [10_000],
            "cb_person_cred_hist_length": [5],
            "person_age": [25],
            "person_emp_length": [5],
        }
    )
    result = add_features(data)
    assert list(data.columns) == [
        "person_income", "loan_amnt", "cb_person_cred_hist_length",
        "person_age", "person_emp_length",
    ]
    assert set(ENGINEERED_FEATURES).issubset(result.columns)
    assert result.loc[0, "income_to_loan_ratio"] == 10
    assert result.loc[0, "credit_history_age_ratio"] == 0.2
    assert result.loc[0, "employment_stability_ratio"] == 0.2


def test_feature_engineer_transformer_and_scenario_exclusion():
    data = pd.DataFrame(
        {
            "person_income": [100_000, 50_000],
            "loan_amnt": [10_000, 5_000],
            "cb_person_cred_hist_length": [5, 2],
            "person_age": [25, 22],
            "person_emp_length": [5, 1],
            "loan_grade": ["A", "B"],
            "loan_int_rate": [7.0, 10.0],
        }
    )
    transformed = FeatureEngineer().fit_transform(data)
    assert set(ENGINEERED_FEATURES).issubset(transformed.columns)
    pipeline = make_pipeline(data, DummyClassifier())
    assert "loan_grade" in pipeline.named_steps["preprocessor"].transformers[1][2]


def test_xgboost_configuration_and_class_weight():
    estimator = make_xgboost(scale_pos_weight=3.0, max_depth=5)
    params = estimator.get_params()
    assert params["scale_pos_weight"] == 3.0
    assert params["max_depth"] == 5
    assert params["eval_metric"] == "logloss"
    assert calculate_scale_pos_weight(pd.Series([0, 0, 0, 1])) == 3.0


def test_search_space_has_expected_xgboost_parameters():
    search_space = xgboost_search_space()
    assert "model__n_estimators" in search_space
    assert "model__learning_rate" in search_space
    assert "model__reg_lambda" in search_space


def test_threshold_metrics_and_oof_coverage():
    result = threshold_metrics(
        pd.Series([0, 0, 1, 1]),
        np.array([0.1, 0.4, 0.6, 0.9]),
        thresholds=(0.5,),
    )
    assert result.loc[0, "Recall"] == 1.0
    assert result.loc[0, "Precision"] == 1.0
    assert tuple(THRESHOLDS) == tuple(round(x / 100, 2) for x in range(10, 71, 5))


def test_out_of_fold_probabilities_cover_every_row():
    data = pd.DataFrame(
        {
            "person_income": [1000 + i * 100 for i in range(20)],
            "loan_amnt": [500 + i * 10 for i in range(20)],
            "cb_person_cred_hist_length": [2 + i % 5 for i in range(20)],
            "person_age": [20 + i % 10 for i in range(20)],
            "person_emp_length": [i % 8 for i in range(20)],
            "loan_grade": ["A", "B"] * 10,
            "loan_int_rate": [7.0 + i % 4 for i in range(20)],
        }
    )
    y = pd.Series([0, 1] * 10)
    probabilities = out_of_fold_probabilities(
        data, y, DummyClassifier(strategy="prior"), cv_splits=5
    )
    assert len(probabilities) == len(data)
    assert not pd.isna(probabilities).any()


def test_calibration_metrics_and_reliability_table():
    y = pd.Series([0, 0, 1, 1, 0, 1])
    predictions = {
        "uncalibrated": np.array([0.1, 0.2, 0.7, 0.8, 0.3, 0.6]),
        "sigmoid": np.array([0.2, 0.3, 0.6, 0.7, 0.4, 0.7]),
        "isotonic": np.array([0.1, 0.2, 0.8, 0.9, 0.2, 0.8]),
    }
    metrics, _ = calibration_metrics(y, predictions, (0.5,))
    assert set(metrics["Model"]) == set(predictions)
    assert {"Brier Score", "Log Loss", "Best F1 Threshold"}.issubset(metrics.columns)
    table = reliability_table(y, predictions["sigmoid"], n_bins=3)
    assert table["number_of_samples"].sum() == len(y)


def test_threshold_costs_and_probability_validation():
    y = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.6, 0.9])
    metrics = cost_threshold_metrics(y, probabilities, (0.3, 0.7))
    scored = add_expected_cost(metrics, false_positive_cost=1, false_negative_cost=5)
    selected = select_lowest_cost_threshold(metrics, 1, 5)
    assert "Expected Cost" in scored
    assert selected["Threshold"] in (0.3, 0.7)
    np.testing.assert_array_equal(validate_probabilities(probabilities), probabilities)
    with pytest.raises(ValueError):
        validate_probabilities(np.array([0.2, 1.2]))


def test_shap_dimensions_names_finiteness_and_single_explanation():
    data = pd.DataFrame(
        {
            "person_income": [1000 + i * 100 for i in range(30)],
            "loan_amnt": [500 + i * 10 for i in range(30)],
            "cb_person_cred_hist_length": [2 + i % 5 for i in range(30)],
            "person_age": [20 + i % 10 for i in range(30)],
            "person_emp_length": [i % 8 for i in range(30)],
            "loan_grade": ["A", "B", "C"] * 10,
            "loan_int_rate": [7.0 + i % 4 for i in range(30)],
        }
    )
    y = pd.Series([0, 1, 0, 1, 1] * 6)
    pipeline = make_pipeline(data, make_xgboost(n_estimators=5, max_depth=2))
    pipeline.fit(data, y)
    shap_values, _, transformed, names = calculate_shap_values(pipeline, data.iloc[:8])
    assert shap_values.shape == transformed.shape
    assert len(names) == transformed.shape[1]
    assert names == get_transformed_feature_names(pipeline)
    assert np.isfinite(shap_values).all()
    importance = create_global_feature_importance(shap_values, names)
    assert importance["mean_absolute_shap"].ge(0).all()
    assert importance["mean_absolute_shap"].is_monotonic_decreasing
    explanation = explain_single_prediction(pipeline, data.iloc[[0]], top_n=2)
    assert 0 <= explanation["predicted_probability"] <= 1
    assert explanation["predicted_class"] in (0, 1)
    assert explanation["positive_contributors"] or explanation["negative_contributors"]


def test_final_candidates_are_predefined_and_include_expected_models():
    candidates = candidate_definitions()
    assert set(candidates) == {
        "Tuned XGBoost - All Valid",
        "Tuned XGBoost - Pre-Decision",
        "Random Forest",
        "Baseline XGBoost",
    }
    assert candidates["Tuned XGBoost - Pre-Decision"][2] == (
        "loan_grade",
        "loan_int_rate",
    )
