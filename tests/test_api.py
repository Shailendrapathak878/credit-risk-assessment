import pytest
from fastapi.testclient import TestClient

from app.main import app

SAMPLE = {
    "person_age": 30,
    "person_income": 60000,
    "person_home_ownership": "RENT",
    "person_emp_length": 5,
    "loan_intent": "PERSONAL",
    "loan_grade": "B",
    "loan_amnt": 10000,
    "loan_int_rate": 11.5,
    "loan_percent_income": 0.17,
    "cb_person_default_on_file": "N",
    "cb_person_cred_hist_length": 7,
}


def test_health_and_openapi():
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["model_loaded"] is True
        assert health.headers["X-Request-ID"]
        assert client.get("/").status_code == 200
        assert client.get("/docs").status_code == 200


def test_predict_accepts_valid_request():
    with TestClient(app) as client:
        response = client.post("/predict", json=SAMPLE)
        body = response.json()
        assert response.status_code == 200
        assert 0 <= body["risk_probability"] <= 1
        assert body["risk_class"] in (0, 1)
        assert body["risk_label"] in ("Low Risk", "High Risk")
        assert body["threshold"] == 0.5
        assert body["model_name"] == "Tuned XGBoost"


@pytest.mark.parametrize(
    "overrides",
    [
        {"person_home_ownership": "MORTGAGE"},
        {"loan_intent": "MEDICAL"},
        {"loan_grade": "F"},
        {"person_income": 120000, "loan_amnt": 5000, "loan_percent_income": 0.04},
    ],
)
def test_synthetic_applicants_return_complete_predictions(overrides):
    with TestClient(app) as client:
        response = client.post("/api/v1/predict", json={**SAMPLE, **overrides})
        body = response.json()
        assert response.status_code == 200
        assert {"risk_probability", "risk_class", "risk_label", "threshold", "model_name"} <= body.keys()


def test_prediction_and_explanation_agree_for_same_synthetic_applicant():
    with TestClient(app) as client:
        prediction = client.post("/api/v1/predict", json=SAMPLE)
        explanation = client.post("/api/v1/explain", json=SAMPLE)
        prediction_body = prediction.json()
        explanation_body = explanation.json()
        assert prediction.status_code == 200
        assert explanation.status_code == 200
        assert explanation_body["risk_probability"] == prediction_body["risk_probability"]
        assert explanation_body["top_positive_factors"]
        assert explanation_body["top_negative_factors"]
        assert explanation_body["model_name"] == prediction_body["model_name"]


def test_invalid_requests_return_422():
    with TestClient(app) as client:
        assert client.post("/predict", json={**SAMPLE, "person_age": 101}).status_code == 422
        assert client.post("/predict", json={**SAMPLE, "person_income": -1}).status_code == 422
        missing = dict(SAMPLE)
        del missing["loan_grade"]
        assert client.post("/predict", json=missing).status_code == 422


def test_explain_returns_shap_contributions():
    with TestClient(app) as client:
        response = client.post("/explain", json=SAMPLE)
        body = response.json()
        assert response.status_code == 200
        assert body["explanation_space"] == "raw_margin_log_odds"
        assert "base_value" in body
        assert body["top_positive_factors"]
        assert body["top_negative_factors"]


def test_model_is_loaded_once_per_application_lifespan():
    with TestClient(app) as client:
        service = app.state.model_service
        assert client.post("/predict", json=SAMPLE).status_code == 200
        assert app.state.model_service is service


def test_versioned_routes_and_model_info():
    with TestClient(app) as client:
        assert client.get("/api/v1/health").status_code == 200
        info = client.get("/api/v1/model-info")
        assert info.status_code == 200
        assert info.json()["model_loaded"] is True
        assert info.json()["explanation_space"] == "raw_margin_log_odds"
        response = client.post("/api/v1/predict", json=SAMPLE)
        assert response.status_code == 200
        assert response.headers["X-Request-ID"]
