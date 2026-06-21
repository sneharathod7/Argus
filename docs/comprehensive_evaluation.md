# Comprehensive Evaluation Report: Elite Parking Intelligence System

## 1. Old vs New Pipeline Comparison (Test Set)

| Metric | Old Baseline (Poisson) | New Elite (Two-Stage ZI) | Improvement |
|--------|----------------------|--------------------------|-------------|
| **MAE** | 0.5338 | 0.3237 | 0.2102 |
| **RMSE** | 1.2215 | 1.1623 | 0.0592 |
| **Macro F1** | 0.2628 | 0.2577 | -0.0051 |
| **CRITICAL Recall** | 0.0321 | 0.0000 | -0.0321 |
| **CRITICAL F1** | 0.0569 | 0.0000 | -0.0569 |

## 2. Stage A: Hotspot Occurrence (Activity Detection)
| Metric | Value |
|--------|-------|
| PR-AUC | 0.2645 |
| Precision | 0.1583 |
| Recall | 0.6903 |
| F1 | 0.2575 |
| Brier Score | 0.0421 |

## 3. Operational Metrics: Recall@K and Precision@K

*"If you dispatch K patrol teams to our top-K risk-scored grids each hour, what fraction of true hotspots do you intercept?"*

| K (Patrol Teams) | Recall@K | Precision@K |
|-------------------|----------|-------------|
| 10 | **0.1402** | 0.1402 |
| 20 | **0.1631** | 0.1631 |
| 50 | **0.2249** | 0.2249 |
| 100 | **0.3548** | 0.3548 |

## 4. Temporal Stability (Per-Week Test Performance)

| Week | Rows | MAE | Macro F1 |
|------|------|-----|----------|
| 11 | 5,995 | 0.2405 | 0.2606 |
| 12 | 52,247 | 0.3037 | 0.2605 |
| 13 | 52,641 | 0.3355 | 0.2556 |
| 14 | 52,658 | 0.3457 | 0.2566 |
| 15 | 5,562 | 0.2811 | 0.2555 |

*MAE standard deviation across weeks: 0.0381*
*Observation: Stable performance indicates no significant concept drift.*

## 5. Calibration Analysis (Stage A Probabilities)

| Predicted Probability Bin | Sample Count | Mean Predicted | Mean Actual | Gap |
|---------------------------|-------------|----------------|-------------|-----|
| [0.0, 0.1) | 134,775 | 0.058 | 0.018 | 0.040 |
| [0.1, 0.2) | 34,328 | 0.114 | 0.158 | 0.044 |

*Average calibration gap: 0.0421*
*Brier Score: 0.0421 (closer to 0 = better calibrated)*

## 6. Classification Report (Severity Classes, Test Set)

```text
              precision    recall  f1-score   support

       CLEAR       0.96      0.97      0.96    161232
         LOW       0.06      0.08      0.07      4658
      MEDIUM       0.00      0.00      0.00      1844
    CRITICAL       0.00      0.00      0.00      1369

    accuracy                           0.93    169103
   macro avg       0.25      0.26      0.26    169103
weighted avg       0.91      0.93      0.92    169103

```

## 7. Data Integrity Verification

> **Data Leakage Check:** PASSED
> - Bayesian priors computed strictly on training data (first 70% chronologically)
> - All lag features use backward-looking temporal offsets
> - same_hour_dow_hist_avg uses only training period history
> - Target generated via T+1 forward offset join

> **Chronological Integrity:** PASSED
> - Dataset sorted by `hour` before splitting
> - Train: oldest 70%, Val: next 15%, Test: newest 15%
> - No temporal overlap between splits

## Training Time
10.1 seconds
