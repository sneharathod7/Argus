"""
Phase 3: Elite Feature Engineering
====================================
Builds the full feature set (~55+ features) from the dense panel.

Feature families:
  1. Exact lag features (1, 2, 3, 6, 12, 24, 168 hours)
  2. Rolling statistics (sum, mean, max, std)
  3. Exponential moving averages
  4. Fourier cyclical encoding (hour, DOW, month)
  5. Calendar enrichment (rush hour, night, etc.)
  6. Burstiness / inter-event features
  7. Spatial neighbor features
  8. Historical recurrence / Bayesian priors (computed at train time)
  9. Multi-horizon targets (T+1, T+2, T+3)

Input:  grid_hourly_panel.parquet (from Phase 2)
Output: elite_forecasting_dataset.parquet
"""

import pandas as pd
import numpy as np
import pygeohash as pgh
import os
import time
import warnings

warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────
# Geohash neighbor computation
# ─────────────────────────────────────────────────────────
def get_geohash_neighbors(geohash_str):
    """Get the 8 adjacent geohash cells."""
    try:
        return pgh.neighbors(geohash_str)
    except Exception:
        return {}


def main():
    start_time = time.time()
    print("=" * 70)
    print("PHASE 3: ELITE FEATURE ENGINEERING")
    print("=" * 70)

    # ─────────────────────────────────────────────────────────
    # Load panel
    # ─────────────────────────────────────────────────────────
    print("\n[1/10] Loading grid_hourly_panel.parquet...")
    df = pd.read_parquet("grid_hourly_panel.parquet")
    df['hour'] = pd.to_datetime(df['hour'])
    df = df.sort_values(['grid_id', 'hour']).reset_index(drop=True)
    print(f"  Loaded {len(df):,} rows, {df['grid_id'].nunique():,} grids")

    initial_cols = len(df.columns)

    # ─────────────────────────────────────────────────────────
    # Feature 1: Exact Lag Features
    # ─────────────────────────────────────────────────────────
    print("\n[2/10] Generating exact lag features...")
    lags = [1, 2, 3, 6, 12, 24, 168]

    for lag in lags:
        col_name = f'violation_count_lag_{lag}'
        # Create a shifted copy to join
        lag_df = df[['grid_id', 'hour', 'violation_count']].copy()
        lag_df['hour'] = lag_df['hour'] + pd.Timedelta(hours=lag)
        lag_df.rename(columns={'violation_count': col_name}, inplace=True)
        df = df.merge(lag_df, on=['grid_id', 'hour'], how='left')
        df[col_name] = df[col_name].fillna(0)

    # Also create weighted violation lags for key horizons
    for lag in [1, 24]:
        col_name = f'weighted_violation_lag_{lag}'
        lag_df = df[['grid_id', 'hour', 'weighted_violation_count']].copy()
        lag_df['hour'] = lag_df['hour'] + pd.Timedelta(hours=lag)
        lag_df.rename(columns={'weighted_violation_count': col_name}, inplace=True)
        df = df.merge(lag_df, on=['grid_id', 'hour'], how='left')
        df[col_name] = df[col_name].fillna(0)

    print(f"  Added {len(lags) + 2} lag features")

    # ─────────────────────────────────────────────────────────
    # Feature 2: Rolling Statistics
    # ─────────────────────────────────────────────────────────
    print("\n[3/10] Generating rolling statistics...")

    # For dense grids, rolling windows on sorted data are correct
    # Group by grid_id, set hour as index, compute rolling
    df_indexed = df.set_index('hour')

    windows = [3, 6, 12, 24, 168]
    for w in windows:
        w_str = f'{w}h'
        print(f"  Rolling window {w_str}...", end=' ')

        # Rolling SUM
        roll = df_indexed.groupby('grid_id')['violation_count'].rolling(
            w_str, min_periods=1
        ).sum().reset_index()
        roll.rename(columns={'violation_count': f'rolling_sum_{w_str}'}, inplace=True)
        df = df.merge(roll, on=['grid_id', 'hour'], how='left')

        # Rolling MEAN (derived from sum)
        df[f'rolling_mean_{w_str}'] = df[f'rolling_sum_{w_str}'] / w

        # Rolling MAX (for select windows)
        if w in [24, 168]:
            roll_max = df_indexed.groupby('grid_id')['violation_count'].rolling(
                w_str, min_periods=1
            ).max().reset_index()
            roll_max.rename(columns={'violation_count': f'rolling_max_{w_str}'}, inplace=True)
            df = df.merge(roll_max, on=['grid_id', 'hour'], how='left')

        # Rolling STD (for select windows — captures volatility)
        if w in [24]:
            roll_std = df_indexed.groupby('grid_id')['violation_count'].rolling(
                w_str, min_periods=2
            ).std().reset_index()
            roll_std.rename(columns={'violation_count': f'rolling_std_{w_str}'}, inplace=True)
            df = df.merge(roll_std, on=['grid_id', 'hour'], how='left')
            df[f'rolling_std_{w_str}'] = df[f'rolling_std_{w_str}'].fillna(0)

        print("done")

    # Aliases
    df['violations_last_24h'] = df['rolling_sum_24h']
    df['violations_last_7d'] = df['rolling_sum_168h']

    # Weighted violation rolling
    roll_w = df_indexed.groupby('grid_id')['weighted_violation_count'].rolling(
        '24h', min_periods=1
    ).sum().reset_index()
    roll_w.rename(columns={'weighted_violation_count': 'weighted_rolling_sum_24h'}, inplace=True)
    df = df.merge(roll_w, on=['grid_id', 'hour'], how='left')

    # ─────────────────────────────────────────────────────────
    # Feature 3: Exponential Moving Average
    # ─────────────────────────────────────────────────────────
    print("\n[4/10] Generating EMA features...")

    for span in [6, 24]:
        ema_col = f'ema_{span}h'
        df[ema_col] = df.groupby('grid_id')['violation_count'].transform(
            lambda x: x.ewm(span=span, adjust=False).mean()
        )
    print("  Added EMA-6h, EMA-24h")

    # ─────────────────────────────────────────────────────────
    # Feature 4: Trend and Burstiness
    # ─────────────────────────────────────────────────────────
    print("\n[5/10] Generating trend and burstiness features...")

    # Trend: current - rolling mean
    df['trend_24h'] = df['violation_count'] - df['rolling_mean_24h']

    # Acceleration: change in rolling_sum_3h over 3 hours
    accel_lag = df[['grid_id', 'hour', 'rolling_sum_3h']].copy()
    accel_lag['hour'] = accel_lag['hour'] + pd.Timedelta(hours=3)
    accel_lag.rename(columns={'rolling_sum_3h': 'rolling_sum_3h_prev'}, inplace=True)
    df = df.merge(accel_lag, on=['grid_id', 'hour'], how='left')
    df['rolling_sum_3h_prev'] = df['rolling_sum_3h_prev'].fillna(0)
    df['violation_acceleration'] = (df['rolling_sum_3h'] - df['rolling_sum_3h_prev']) / 3.0

    # Hours since last violation (for dense grids)
    # Compute consecutive-zero run length
    df['has_violation'] = (df['violation_count'] > 0).astype(int)
    df['hours_since_last_violation'] = df.groupby('grid_id')['has_violation'].transform(
        lambda x: x.groupby((x != 0).cumsum()).cumcount()
    )
    # Cap at 168 (1 week)
    df['hours_since_last_violation'] = df['hours_since_last_violation'].clip(upper=168)

    print("  Added trend, acceleration, hours_since_last_violation")

    # ─────────────────────────────────────────────────────────
    # Feature 5: Fourier Cyclical Encoding
    # ─────────────────────────────────────────────────────────
    print("\n[6/10] Generating Fourier cyclical features...")
    df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week_num'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week_num'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    print("  Added 6 Fourier features")

    # ─────────────────────────────────────────────────────────
    # Feature 6: Calendar Enrichment
    # ─────────────────────────────────────────────────────────
    print("\n[7/10] Generating calendar features...")
    df['is_rush_hour'] = df['hour_of_day'].isin([8, 9, 10, 17, 18, 19, 20]).astype(int)
    df['is_night'] = df['hour_of_day'].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
    df['is_late_morning'] = df['hour_of_day'].isin([10, 11, 12]).astype(int)
    print("  Added rush_hour, night, late_morning flags")

    # ─────────────────────────────────────────────────────────
    # Feature 7: Spatial Neighbor Features
    # ─────────────────────────────────────────────────────────
    print("\n[8/10] Generating spatial neighbor features...")

    # Precompute neighbors for all grids
    all_grids = df['grid_id'].unique()
    grid_neighbor_map = {}
    for g in all_grids:
        neighbors = get_geohash_neighbors(g)
        if isinstance(neighbors, dict):
            grid_neighbor_map[g] = list(neighbors.values())
        elif isinstance(neighbors, (list, tuple)):
            grid_neighbor_map[g] = list(neighbors)
        else:
            grid_neighbor_map[g] = []

    # Create a lookup: (grid_id, hour) -> violation_count for fast neighbor aggregation
    print("  Building grid-hour violation lookup...")
    grid_hour_violations = df.set_index(['grid_id', 'hour'])['violation_count'].to_dict()

    # For efficiency, compute neighbor features in vectorized batches
    print("  Computing neighbor aggregates (this may take a moment)...")

    def compute_neighbor_features_batch(df_chunk, grid_hour_violations, grid_neighbor_map):
        """Compute neighbor features for a batch of rows."""
        neighbor_sum_list = []
        neighbor_max_list = []
        neighbor_active_list = []

        for _, row in df_chunk.iterrows():
            grid = row['grid_id']
            hour = row['hour']
            neighbors = grid_neighbor_map.get(grid, [])

            vals = []
            for n in neighbors:
                key = (n, hour)
                v = grid_hour_violations.get(key, 0)
                vals.append(v)

            if vals:
                neighbor_sum_list.append(sum(vals))
                neighbor_max_list.append(max(vals))
                neighbor_active_list.append(sum(1 for v in vals if v > 0))
            else:
                neighbor_sum_list.append(0)
                neighbor_max_list.append(0)
                neighbor_active_list.append(0)

        return neighbor_sum_list, neighbor_max_list, neighbor_active_list

    # Process in chunks to show progress
    CHUNK_SIZE = 50000
    n_chunks = (len(df) + CHUNK_SIZE - 1) // CHUNK_SIZE
    all_sums, all_maxes, all_active = [], [], []

    for i in range(n_chunks):
        start = i * CHUNK_SIZE
        end = min((i + 1) * CHUNK_SIZE, len(df))
        chunk = df.iloc[start:end]
        s, m, a = compute_neighbor_features_batch(chunk, grid_hour_violations, grid_neighbor_map)
        all_sums.extend(s)
        all_maxes.extend(m)
        all_active.extend(a)
        print(f"  Chunk {i+1}/{n_chunks} done ({end:,} / {len(df):,} rows)", end='\r')

    df['neighbor_violation_sum'] = all_sums
    df['neighbor_max_violation'] = all_maxes
    df['neighbor_active_count'] = all_active
    print(f"\n  Added 3 spatial neighbor features")

    # ─────────────────────────────────────────────────────────
    # Feature 8: Hotspot Frequency / Persistence Features
    # ─────────────────────────────────────────────────────────
    print("\n[9/10] Generating hotspot frequency and persistence features...")

    # Hotspot frequency: number of active hours in last 7 days (rolling count of non-zero)
    df['hotspot_frequency_7d'] = df.groupby('grid_id')['has_violation'].transform(
        lambda x: x.rolling(168, min_periods=1).sum()
    )

    # Active streak: consecutive hours with violations
    # Compute using cumsum trick
    df['active_streak'] = df.groupby('grid_id')['has_violation'].transform(
        lambda x: x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1)
    )

    print("  Added hotspot_frequency_7d, active_streak")

    # ─────────────────────────────────────────────────────────
    # Feature 9: Multi-Horizon Targets
    # ─────────────────────────────────────────────────────────
    print("\n[10/10] Generating multi-horizon targets...")

    for horizon in [1, 2, 3]:
        target_col = f'target_violation_count_{horizon}h'
        target_df = df[['grid_id', 'hour', 'violation_count']].copy()
        target_df['hour'] = target_df['hour'] - pd.Timedelta(hours=horizon)
        target_df.rename(columns={'violation_count': target_col}, inplace=True)
        df = df.merge(target_df, on=['grid_id', 'hour'], how='left')
        df[target_col] = df[target_col].fillna(0)

    # Binary targets: will there be any violation?
    for horizon in [1, 2, 3]:
        df[f'target_is_active_{horizon}h'] = (df[f'target_violation_count_{horizon}h'] > 0).astype(int)

    # Primary severity target (for backwards compatibility)
    def get_severity(x):
        if pd.isna(x) or x == 0:
            return 'CLEAR'
        if 1 <= x <= 2:
            return 'LOW'
        if 3 <= x <= 5:
            return 'MEDIUM'
        return 'CRITICAL'

    df['target_severity_1h'] = df['target_violation_count_1h'].apply(get_severity)

    print("  Added T+1, T+2, T+3 targets (count + binary + severity)")

    # ─────────────────────────────────────────────────────────
    # Clean up temporary columns
    # ─────────────────────────────────────────────────────────
    drop_cols = ['has_violation', 'rolling_sum_3h_prev']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # ─────────────────────────────────────────────────────────
    # Save
    # ─────────────────────────────────────────────────────────
    output_path = 'elite_forecasting_dataset.parquet'
    df.to_parquet(output_path, index=False)

    elapsed = time.time() - start_time
    final_cols = len(df.columns)

    print("\n" + "=" * 70)
    print("PHASE 3 COMPLETE")
    print("=" * 70)
    print(f"  Total rows:     {len(df):,}")
    print(f"  Columns:        {initial_cols} -> {final_cols} (+{final_cols - initial_cols} features)")
    print(f"  Output:         {output_path}")
    print(f"  Output size:    {os.path.getsize(output_path) / 1e6:.1f} MB")
    print(f"  Time elapsed:   {elapsed:.1f}s")

    # Feature summary
    feature_families = {
        'Lags': [c for c in df.columns if 'lag_' in c],
        'Rolling': [c for c in df.columns if 'rolling_' in c],
        'EMA': [c for c in df.columns if 'ema_' in c],
        'Fourier': [c for c in df.columns if '_sin' in c or '_cos' in c],
        'Calendar': ['is_rush_hour', 'is_night', 'is_late_morning', 'is_weekend'],
        'Spatial': [c for c in df.columns if 'neighbor_' in c],
        'Burstiness': ['trend_24h', 'violation_acceleration', 'hours_since_last_violation'],
        'Persistence': ['hotspot_frequency_7d', 'active_streak'],
        'Targets': [c for c in df.columns if 'target_' in c],
    }

    print("\n  Feature Summary:")
    for family, cols in feature_families.items():
        existing = [c for c in cols if c in df.columns]
        print(f"    {family}: {len(existing)} features")

    # Target distribution
    print("\n  Target Distribution (T+1):")
    for sev in ['CLEAR', 'LOW', 'MEDIUM', 'CRITICAL']:
        count = (df['target_severity_1h'] == sev).sum()
        pct = count / len(df) * 100
        print(f"    {sev}: {count:,} ({pct:.2f}%)")

    # Save feature report
    report = f"""# Elite Feature Engineering Report

## Summary
- **Total Rows:** {len(df):,}
- **Total Features:** {final_cols}
- **New Features Added:** {final_cols - initial_cols}

## Feature Families
| Family | Count | Features |
|--------|-------|----------|
"""
    for family, cols in feature_families.items():
        existing = [c for c in cols if c in df.columns]
        report += f"| {family} | {len(existing)} | {'`, `'.join(existing[:5])}{'...' if len(existing) > 5 else ''} |\n"

    report += f"""
## Target Distribution (T+1 Horizon)
| Severity | Count | Percentage |
|----------|-------|------------|
| CLEAR | {(df['target_severity_1h']=='CLEAR').sum():,} | {(df['target_severity_1h']=='CLEAR').mean()*100:.2f}% |
| LOW | {(df['target_severity_1h']=='LOW').sum():,} | {(df['target_severity_1h']=='LOW').mean()*100:.2f}% |
| MEDIUM | {(df['target_severity_1h']=='MEDIUM').sum():,} | {(df['target_severity_1h']=='MEDIUM').mean()*100:.2f}% |
| CRITICAL | {(df['target_severity_1h']=='CRITICAL').sum():,} | {(df['target_severity_1h']=='CRITICAL').mean()*100:.2f}% |

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
"""
    with open('feature_engineering_report.md', 'w') as f:
        f.write(report)
    print("\n  Report saved to feature_engineering_report.md")


if __name__ == '__main__':
    main()
