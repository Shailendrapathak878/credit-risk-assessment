# API Reference

Start the service from the repository root:

```bash
uvicorn app.main:app --reload
```

The dashboard is available at `http://127.0.0.1:8000/` and interactive Swagger
documentation is available at `/docs`.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Dashboard HTML |
| GET | `/health` | Model and service health |
| GET | `/api/v1/health` | Versioned health response |
| GET | `/api/v1/model-info` | Safe model metadata |
| POST | `/predict` | Applicant risk prediction |
| POST | `/api/v1/predict` | Versioned prediction |
| POST | `/explain` | Prediction with SHAP contributors |
| POST | `/api/v1/explain` | Versioned explanation |

## Request

Both POST routes accept the same JSON payload:

```json
{
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
  "cb_person_cred_hist_length": 7
}
```

Age must be 18-100, positive monetary values are required, and loan percent
of income must be greater than 0 and no more than 1. Invalid payloads return
`422` with Pydantic validation details.

## Prediction response

```json
{
  "risk_probability": 0.03362530097365379,
  "risk_class": 0,
  "risk_label": "Low Risk",
  "threshold": 0.5,
  "model_name": "Tuned XGBoost",
  "model": "tuned_xgboost_all_valid"
}
```

The probability is uncalibrated and the threshold is the frozen evaluation
threshold, not a business-approved lending rule.

## Explanation response

The explanation response includes the prediction fields plus `base_value`,
`explanation_space`, and `top_positive_factors` / `top_negative_factors`.
Each factor contains a readable feature, transformed value, SHAP value, and
direction. SHAP values use raw-margin/log-odds space.

All responses include an `X-Request-ID` header. Request bodies are not logged.
