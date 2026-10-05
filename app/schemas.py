"""Validated request and response schemas for the inference API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ApplicantRequest(BaseModel):
    person_age: int = Field(..., ge=18, le=100)
    person_income: float = Field(..., gt=0)
    person_home_ownership: str
    person_emp_length: float = Field(..., ge=0)
    loan_intent: str
    loan_grade: str
    loan_amnt: float = Field(..., gt=0)
    loan_int_rate: float = Field(..., gt=0)
    loan_percent_income: float = Field(..., gt=0, le=1)
    cb_person_default_on_file: str
    cb_person_cred_hist_length: float = Field(..., ge=0)

    def to_record(self) -> dict[str, object]:
        return self.model_dump()


class PredictionResponse(BaseModel):
    risk_probability: float
    risk_class: int
    risk_label: str
    threshold: float
    model_name: str
    model: str


class Contribution(BaseModel):
    feature: str
    value: object
    shap_value: float
    direction: str


class ExplanationResponse(PredictionResponse):
    base_value: float
    explanation_space: str
    top_positive_factors: list[Contribution]
    top_negative_factors: list[Contribution]
