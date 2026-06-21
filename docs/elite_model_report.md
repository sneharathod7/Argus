# Elite Model Evaluation Report

## Architecture
Two-Stage Zero-Inflated Model with Ensemble:
- **Stage A**: Hotspot Occurrence (Binary LightGBM, is_unbalance=True, Optuna-tuned)
- **Stage B**: Conditional Intensity (LightGBM Tweedie + CatBoost Tweedie blend)
- **Risk Score**: Composite ranking (no hard thresholds)

## Data Split (Chronological)
| Split | Rows | Time Range |
|-------|------|------------|
| Train (70%) | 789,145 | 2023-11-09 19:00:00+00:00 to 2024-02-23 14:00:00+00:00 |
| Validation (15%) | 169,103 | 2024-02-23 14:00:00+00:00 to 2024-03-17 04:00:00+00:00 |
| Test (15%) | 169,103 | 2024-03-17 04:00:00+00:00 to 2024-04-08 17:00:00+00:00 |

## Bayesian Features (Leakage-Free)
| Feature | Description |
|---------|-------------|
| `bayesian_active_rate` | Smoothed P(violation > 0) per grid |
| `bayesian_critical_rate` | Smoothed P(violation >= 6) per grid |
| `same_hour_dow_hist_avg` | Historical average for (grid, hour, DOW) |
| `is_grid_peak_hour` | Binary: is current hour the grid's historical peak? |
| `grid_avg_response_time` | Historical mean enforcement response time |

## Stage A: Hotspot Occurrence (Test Set)
| Metric | Value |
|--------|-------|
| PR-AUC | 0.2645 |
| Optimal Threshold | 0.10 |
| Precision | 0.1583 |
| Recall | 0.6903 |
| F1 | 0.2575 |

## Stage B: Conditional Intensity
- **LightGBM Tweedie** trained on 33,759 active-only rows
- **Ensemble Blend Weight**: LightGBM=0.75, CatBoost=0.25

## Combined Two-Stage Results (Test Set)
| Metric | Value |
|--------|-------|
| MAE | 0.3237 |
| RMSE | 1.1623 |
| Macro F1 (Severity) | 0.2577 |

## Operational Metrics: Recall@K
*"If you dispatch K patrol teams to our top-K recommended grids, what fraction of true hotspots do you catch?"*

| K | Recall@K |
|---|----------|
| 10 | 0.1402 |
| 20 | 0.1631 |
| 50 | 0.2249 |
| 100 | 0.3548 |

## Top 10 Most Important Features

### Stage A (Occurrence)
| Rank | Feature | Importance (Gain) |
|------|---------|------------------|
| 1 | `same_hour_dow_hist_avg` | 5996031.35 |
| 2 | `hour_cos` | 5717129.89 |
| 3 | `grid_id` | 1643633.79 |
| 4 | `hour_of_day` | 999173.42 |
| 5 | `bayesian_active_rate` | 493983.52 |
| 6 | `rolling_sum_24h` | 418746.87 |
| 7 | `ema_24h` | 343615.49 |
| 8 | `rolling_sum_3h` | 297219.50 |
| 9 | `rolling_std_24h` | 285494.05 |
| 10 | `hotspot_frequency_7d` | 100231.45 |

### Stage B (Intensity)
| Rank | Feature | Importance (Gain) |
|------|---------|------------------|
| 1 | `grid_id` | 157415.32 |
| 2 | `bayesian_critical_rate` | 30150.82 |
| 3 | `rolling_max_168h` | 16716.03 |
| 4 | `rolling_sum_168h` | 14896.12 |
| 5 | `bayesian_active_rate` | 14274.25 |
| 6 | `ema_24h` | 13867.78 |
| 7 | `mean_violations` | 13576.78 |
| 8 | `day_of_month` | 13550.26 |
| 9 | `same_hour_dow_hist_avg` | 13455.78 |
| 10 | `grid_avg_response_time` | 12696.41 |

## Optuna Hyperparameter Search
- Stage A: 0.2340 PR-AUC (50 trials)
- Stage B: 2.3767 MAE (50 trials)

## Models Saved
- `model_stage_a_occurrence.txt` (LightGBM binary)
- `model_stage_b_lgb_intensity.txt` (LightGBM Tweedie)
- `model_stage_b_catboost_intensity.cbm` (CatBoost Tweedie)

## Feature List (72 total)
- `avg_response_time`
- `day_of_month`
- `day_of_week`
- `day_of_week_num`
- `dominant_vehicle_type`
- `dominant_violation_type`
- `grid_id`
- `heavy_vehicle_ratio`
- `hour_of_day`
- `is_dense_grid`
- `is_weekend`
- `month`
- `multi_violation_ratio`
- `num_devices_active`
- `num_offence_codes_total`
- `police_station`
- `repeat_offender_ratio`
- `unique_vehicle_count`
- `violation_count`
- `violation_entropy`
- `week_of_year`
- `weighted_violation_count`
- `violation_count_lag_1`
- `violation_count_lag_2`
- `violation_count_lag_3`
- `violation_count_lag_6`
- `violation_count_lag_12`
- `violation_count_lag_24`
- `violation_count_lag_168`
- `weighted_violation_lag_1`
- `weighted_violation_lag_24`
- `rolling_sum_3h`
- `rolling_mean_3h`
- `rolling_sum_6h`
- `rolling_mean_6h`
- `rolling_sum_12h`
- `rolling_mean_12h`
- `rolling_sum_24h`
- `rolling_mean_24h`
- `rolling_max_24h`
- `rolling_std_24h`
- `rolling_sum_168h`
- `rolling_mean_168h`
- `rolling_max_168h`
- `violations_last_24h`
- `violations_last_7d`
- `weighted_rolling_sum_24h`
- `ema_6h`
- `ema_24h`
- `trend_24h`
- `violation_acceleration`
- `hours_since_last_violation`
- `hour_sin`
- `hour_cos`
- `dow_sin`
- `dow_cos`
- `month_sin`
- `month_cos`
- `is_rush_hour`
- `is_night`
- `is_late_morning`
- `neighbor_violation_sum`
- `neighbor_max_violation`
- `neighbor_active_count`
- `hotspot_frequency_7d`
- `active_streak`
- `bayesian_active_rate`
- `bayesian_critical_rate`
- `mean_violations`
- `same_hour_dow_hist_avg`
- `is_grid_peak_hour`
- `grid_avg_response_time`

## Training Time
289.8 seconds
