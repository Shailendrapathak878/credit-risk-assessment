# Final Model Comparison

The 6,484-row held-out test set was used exactly once for final evaluation.
Candidates, preprocessing, features, threshold, and calibration policy were fixed beforehand.

## Tuned XGBoost - All Valid

```text
              precision    recall  f1-score   support

           0       0.93      0.99      0.96      5066
           1       0.97      0.74      0.84      1418

    accuracy                           0.94      6484
   macro avg       0.95      0.87      0.90      6484
weighted avg       0.94      0.94      0.94      6484

```

## Tuned XGBoost - Pre-Decision

```text
              precision    recall  f1-score   support

           0       0.90      0.98      0.94      5066
           1       0.90      0.61      0.72      1418

    accuracy                           0.90      6484
   macro avg       0.90      0.79      0.83      6484
weighted avg       0.90      0.90      0.89      6484

```

## Random Forest

```text
              precision    recall  f1-score   support

           0       0.93      0.99      0.96      5066
           1       0.97      0.72      0.83      1418

    accuracy                           0.93      6484
   macro avg       0.95      0.86      0.89      6484
weighted avg       0.94      0.93      0.93      6484

```

## Baseline XGBoost

```text
              precision    recall  f1-score   support

           0       0.93      0.99      0.96      5066
           1       0.97      0.73      0.83      1418

    accuracy                           0.94      6484
   macro avg       0.95      0.86      0.90      6484
weighted avg       0.94      0.94      0.93      6484

```

## Final metrics

```text
                       model        feature_set  threshold  accuracy  precision   recall       f1  roc_auc   pr_auc  specificity      fpr      fnr  brier_score  log_loss   tn  fp  fn   tp
   Tuned XGBoost - All Valid all_valid_features        0.5  0.938310   0.973929 0.737659 0.839486 0.949201 0.907477     0.994473 0.005527 0.262341     0.051136  0.185310 5038  28 372 1046
Tuned XGBoost - Pre-Decision  pre_decision_only        0.5  0.899136   0.901261 0.605078 0.724051 0.916441 0.841684     0.981445 0.018555 0.394922     0.076109  0.257273 4972  94 560  858
               Random Forest all_valid_features        0.5  0.934762   0.973359 0.721439 0.828676 0.930137 0.882432     0.994473 0.005527 0.278561     0.058066  0.214068 5038  28 395 1023
            Baseline XGBoost all_valid_features        0.5  0.936151   0.972693 0.728491 0.833065 0.939039 0.893531     0.994276 0.005724 0.271509     0.055332  0.202840 5037  29 385 1033
```

Predictive winner by PR-AUC, ROC-AUC, and F1 tie-break: **Tuned XGBoost - All Valid**.
The all-valid winner is not automatically production-safe because grade and interest rate availability requires domain validation.
Calibration: uncalibrated probabilities, as selected using training-only Phase 4 OOF results.
Threshold: 0.50 evaluation threshold, not a business-approved production threshold.

The held-out test set was not used for tuning, feature selection, threshold selection, calibration selection, or retraining.