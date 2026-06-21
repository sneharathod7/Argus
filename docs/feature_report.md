# Temporal Feature Engineering Report

## Dataset Summary
- **Processed Rows:** 99,140 (Maintained exact dimensionality via precision offset joins)
- **Features Generated:**
  - **Lags:** `lag_1`, `lag_2`, `lag_3`, `lag_6`, `lag_12`, `lag_24`
  - **Rolling Sums:** `6h`, `12h`, `24h`
  - **Rolling Means:** `3h`, `6h`, `12h`, `24h` (Adjusted for true hourly window normalization)
  - **Rolling Max:** `24h`
  - **Persistence:** `violations_last_24h`, `violations_last_7d`
  - **Trend:** `current_count_minus_rolling_mean_24h`

## Target Distribution (T+1 Forecasting)
The target tracks the exact violation count occurring in the identical spatial grid in the subsequent hour.

| Severity Class | Definition | Volume | Percentage |
|----------------|------------|--------|------------|
| **CLEAR** | 0 violations | 79,354 | 80.04% |
| **LOW** | 1-2 violations | 11,173 | 11.27% |
| **MEDIUM** | 3-5 violations | 4,821 | 4.86% |
| **CRITICAL** | 6+ violations | 3,792 | 3.82% |

### Engineering Strategy Notes
- **Zero Data Leakage:** Targets were generated strictly by offset joining `T+1` backwards to `T`. Lags were joined by offsetting `T` forward to `T+Lag`. No future rows bleed into current features.
- **Sparsity Handling:** Missing rows for lag features were robustly interpreted as 0 violations, preserving mathematical integrity without bloating the dataset into 20 million empty static rows.
