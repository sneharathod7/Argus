# Elite Feature Engineering Report

## Summary
- **Total Rows:** 1,127,351
- **Total Features:** 76
- **New Features Added:** 51

## Feature Families
| Family | Count | Features |
|--------|-------|----------|
| Lags | 9 | violation_count_lag_1`, `violation_count_lag_2`, `violation_count_lag_3`, `violation_count_lag_6`, `violation_count_lag_12... |
| Rolling | 14 | rolling_sum_3h`, `rolling_mean_3h`, `rolling_sum_6h`, `rolling_mean_6h`, `rolling_sum_12h... |
| EMA | 2 | ema_6h`, `ema_24h |
| Fourier | 7 | hours_since_last_violation`, `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos... |
| Calendar | 4 | is_rush_hour`, `is_night`, `is_late_morning`, `is_weekend |
| Spatial | 3 | neighbor_violation_sum`, `neighbor_max_violation`, `neighbor_active_count |
| Burstiness | 3 | trend_24h`, `violation_acceleration`, `hours_since_last_violation |
| Persistence | 2 | hotspot_frequency_7d`, `active_streak |
| Targets | 7 | target_violation_count_1h`, `target_violation_count_2h`, `target_violation_count_3h`, `target_is_active_1h`, `target_is_active_2h... |

## Target Distribution (T+1 Horizon)
| Severity | Count | Percentage |
|----------|-------|------------|
| CLEAR | 1,078,218 | 95.64% |
| LOW | 30,071 | 2.67% |
| MEDIUM | 11,261 | 1.00% |
| CRITICAL | 7,801 | 0.69% |

## Multi-Horizon Targets
- `target_violation_count_1h`: Exact count at T+1
- `target_violation_count_2h`: Exact count at T+2
- `target_violation_count_3h`: Exact count at T+3
- `target_is_active_1h/2h/3h`: Binary (any violation?)
- `target_severity_1h`: Categorical severity class

## Key Design Decisions
- **Lag-168** captures same-time-last-week recurrence
- **EMA** provides decay-weighted recent trends
- **Spatial neighbors** act as lightweight graph propagation
- **Hours since last violation** encodes burstiness for sparse grids
- **Violation acceleration** detects emerging hotspots
