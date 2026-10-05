# Synthetic Demo Scenarios

These are demonstration inputs created for Phase 9 API validation. They are
not held-out observations, benchmark results, or credit decisions. Returned
values below were recorded from the frozen artifact through the versioned API.

## Low Risk

```json
{
  "person_age": 35, "person_income": 120000,
  "person_home_ownership": "MORTGAGE", "person_emp_length": 10,
  "loan_intent": "EDUCATION", "loan_grade": "A", "loan_amnt": 5000,
  "loan_int_rate": 7.5, "loan_percent_income": 0.04,
  "cb_person_default_on_file": "N", "cb_person_cred_hist_length": 15
}
```

Returned probability: `0.00856566522270441`; class: `0`; label: `Low Risk`.

## Moderate Risk

The balanced sample uses RENT ownership, PERSONAL intent, and grade B.

Returned probability: `0.03362530097365379`; class: `0`; label: `Low Risk`.

## Higher Risk

The stress sample uses low income, RENT ownership, MEDICAL intent, grade F,
high loan burden, and a prior default flag.

Returned probability: `0.9991820454597473`; class: `1`; label: `High Risk`.

## Validation Error

Using the moderate-risk payload with `person_age: 17` returned HTTP `422` from
both `/api/v1/predict` and `/api/v1/explain`. The applicant is rejected by
schema validation before model inference.
