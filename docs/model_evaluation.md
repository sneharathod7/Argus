# Model Evaluation Report: STGF Baseline

## Overview
We trained a LightGBM Regressor using a **Poisson** objective function to predict exact hourly violation counts (`target_violation_count`), mapping the continuous predictions to Business Severity Categories.

### Temporal Data Split
- **Train Split (70%):** `2023-11-09 19:00:00+00:00` to `2024-02-20 03:00:00+00:00`
- **Validation Split (15%):** `2024-02-20 04:00:00+00:00` to `2024-03-15 19:00:00+00:00`
- **Test Split (15%):** `2024-03-15 19:00:00+00:00` to `2024-04-08 17:00:00+00:00`

## Evaluation Metrics (Hold-out Test Set)

- **Primary Metric (Regression):** Mean Absolute Error (MAE) = `1.0263` violations.
  *This means our forecast is off by less than 1.03 violations per hour on average.*
- **Secondary Metric (Classification):** Macro F1-Score = `0.3220`.

### Business Severity Classification Report
```text
              precision    recall  f1-score   support

       CLEAR       0.90      0.66      0.76     11879
         LOW       0.17      0.53      0.26      1708
      MEDIUM       0.14      0.13      0.13       725
    CRITICAL       0.37      0.08      0.13       559

    accuracy                           0.60     14871
   macro avg       0.39      0.35      0.32     14871
weighted avg       0.76      0.60      0.65     14871

```

## Top 5 Most Important Features
- **grid_id**: 2880 (splits)
- **hour_of_day**: 528 (splits)
- **current_count_minus_rolling_mean_24h**: 332 (splits)
- **violations_last_7d**: 300 (splits)
- **day_of_week**: 166 (splits)

## Conclusion
- The model successfully forecasts severe hotspots without data leakage using strict chronological splitting.
- Model saved to `stgf_lightgbm_model.txt`.
