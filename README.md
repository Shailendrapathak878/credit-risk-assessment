# Credit Risk Assessment

## Overview

This repository demonstrates an end-to-end credit-risk classification system: data audit, leakage-aware preprocessing, feature engineering, model selection, unseen-test evaluation, SHAP explanations, a FastAPI inference service, and a same-origin dashboard. The deployed model artifact is frozen for this phase.

## Problem Statement

Estimate the probability that an applicant belongs to the positive credit-risk class from demographic, employment, loan, and credit-history attributes. The output is a model demonstration, not an approved lending decision.

## Objectives

- Build a reproducible, leakage-aware ML workflow.
- Compare baseline and tuned classifiers on an untouched final test set.
- Serve the selected artifact through a validated API.
- Make individual predictions inspectable with SHAP model contributions.

## Dataset

The supplied dataset has 32,581 rows and 12 columns. `loan_status` is the binary target. Duplicate removal and a stratified `random_state=42` split were defined before final evaluation. The held-out test partition contains 6,484 rows and was not used for tuning, feature engineering, calibration selection, threshold selection, SHAP ranking, or retraining.

## Data Preprocessing

The pipeline imputes missing values, scales numeric features, and one-hot encodes categorical features. The `ColumnTransformer` is fitted only on the development partition; unknown inference categories are ignored safely.

## Feature Engineering

Training-time feature engineering adds target-independent ratios for income to loan, credit-history age to applicant age, and employment stability to age. The API accepts raw applicant fields and runs the serialized pipeline.

## Models Evaluated

The workflow compared Logistic Regression, Random Forest, baseline XGBoost, and tuned XGBoost under all-valid and pre-decision-only feature scenarios. Threshold and calibration analysis were performed on training-only out-of-fold predictions.

## Model Selection

The final ranking policy selected the candidate with the strongest final-test PR-AUC, with ROC-AUC and F1 as supporting comparisons. The selected model is Tuned XGBoost with all valid features. `loan_grade` and `loan_int_rate` require business validation before any real deployment.

## Final Model

- Model: Tuned XGBoost
- Feature scenario: All Valid Features
- Threshold: 0.50 evaluation threshold
- Calibration: Uncalibrated
- Explainability: SHAP TreeExplainer
- Artifact: `models/final_credit_risk_model.joblib`
- Metadata: `models/final_model_metadata.json`
- Runtime model version: `phase_9`
- Random state: 42

## Test Performance

These are the existing Phase 6 final-test values from `reports/final_test_metrics.csv`; they were not recalculated or changed.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier Score | Log Loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.934762 | 0.973359 | 0.721439 | 0.828676 | 0.930137 | 0.882432 | 0.058066 | 0.214068 |
| Baseline XGBoost | 0.936151 | 0.972693 | 0.728491 | 0.833065 | 0.939039 | 0.893531 | 0.055332 | 0.202840 |
| Tuned XGBoost - All Valid | 0.938310 | 0.973929 | 0.737659 | 0.839486 | 0.949201 | 0.907477 | 0.051136 | 0.185310 |
| Tuned XGBoost - Pre-Decision | 0.899136 | 0.901261 | 0.605078 | 0.724051 | 0.916441 | 0.841684 | 0.076109 | 0.257273 |

## Explainability

The explanation endpoint uses SHAP `TreeExplainer` in raw-margin/log-odds space. Positive and negative contributors describe model contribution and are not causal effects or direct probability changes.

## System Architecture

```text
Dataset -> Audit -> Preprocessing -> Feature Engineering -> Baseline Models
    -> XGBoost -> Hyperparameter Tuning -> Threshold Analysis -> Calibration
    -> SHAP -> Final Unseen Test -> Saved Model -> FastAPI -> Dashboard
```

See [docs/architecture.md](docs/architecture.md) for the runtime design.

## API

The service provides `/health`, `/predict`, `/explain`, and versioned equivalents under `/api/v1/`. `/api/v1/model-info` exposes safe metadata only. Swagger is available at `/docs`. See [docs/api.md](docs/api.md) for request and response examples.

## Demo Dashboard

The dashboard at `/` submits raw form values to the backend, displays the returned probability, risk label, threshold, model name, and SHAP contributors. It has no prediction logic of its own and does not hardcode an output.

## Project Structure

```text
app/       FastAPI app, schemas, services, and static dashboard
src/       Audit, preprocessing, feature engineering, training analyses
data/      Supplied dataset
models/    Frozen model artifact and metadata
reports/   Final metrics and explainability reports
tests/     Unit and API integration tests
docs/      Architecture, API, and synthetic demo scenarios
```

## Installation

```bash
python -m pip install -r requirements.txt
```

For local tests and training/report generation, install the additional
development dependencies:

```bash
python -m pip install -r requirements-dev.txt
```

## Running Locally

```bash
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/`. Run the tests with `python -m pytest -q`.

## Docker

```bash
docker build -t credit-risk-assessment .
docker run --rm -p 8000:8000 credit-risk-assessment
```

The image includes the application, source modules, and frozen model artifact; datasets, reports, notebooks, tests, caches, and Git metadata are excluded.

## Vercel

This project uses Vercel's current zero-configuration FastAPI support. Vercel
detects the `FastAPI` instance exported as `app` from `app/main.py`; no
`vercel.json` or separate `api/index.py` adapter is required. The existing
`StaticFiles` mount serves the dashboard and its assets from the same
deployment domain, while the model artifact is loaded from the packaged
`models/` directory at application startup.

Connect the repository to Vercel and deploy with the default Python settings.
No secrets are required. Optional runtime settings are `MODEL_PATH`,
`METADATA_PATH`, `STATIC_PATH`, `MODEL_VERSION`, and
`PREDICTION_THRESHOLD`; the repository defaults are sufficient for the frozen
artifact.

## Example API Request

```bash
curl -X POST http://127.0.0.1:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"person_age":30,"person_income":60000,"person_home_ownership":"RENT","person_emp_length":5,"loan_intent":"PERSONAL","loan_grade":"B","loan_amnt":10000,"loan_int_rate":11.5,"loan_percent_income":0.17,"cb_person_default_on_file":"N","cb_person_cred_hist_length":7}'
```

## How to Demonstrate the Project

1. Start FastAPI and open the dashboard.
2. Enter an applicant profile and click **Assess Credit Risk**.
3. Show the returned probability, Low Risk/High Risk result, threshold, and SHAP contributors.
4. Open `/docs` and show the versioned prediction endpoint.
5. Explain that the API and dashboard use the same frozen model artifact.

Verified synthetic inputs and returned outputs are in [docs/demo_scenarios.md](docs/demo_scenarios.md).

## Limitations

The artifact is development-trained and the 0.50 threshold is an evaluation threshold, not a business-approved policy. The dataset may contain unusual source values. Feature availability, fairness, calibration suitability, privacy, governance, authentication, authorization, HTTPS, rate limiting, and operational monitoring require additional review.

## Future Improvements

Validate feature availability with domain owners, define a cost matrix, monitor drift and calibration, add governance and audit controls, and evaluate fairness before considering production use.

## Disclaimer

This project is for educational and portfolio demonstration only. It must not be used to make real credit, lending, or eligibility decisions.
