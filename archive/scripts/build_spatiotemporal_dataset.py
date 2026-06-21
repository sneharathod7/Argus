"""
Phase 2: Dense Spatiotemporal Panel Construction
=================================================
Builds a complete Grid × Hour panel with:
  - Top-300 most active grids: full hourly materialization (with explicit zeros)
  - Remaining grids: event-driven rows (sparse, as before)
  - Enhanced aggregation features per grid-hour

Input:  cleaned_dataset.parquet (from Phase 1)
Output: grid_hourly_panel.parquet
"""

import pandas as pd
import numpy as np
import pygeohash as pgh
import json
import ast
import os
import time
from collections import Counter


def parse_json_column(series):
    """Safely parse JSON-encoded list columns."""
    def safe_parse(val):
        if pd.isna(val):
            return []
        try:
            return json.loads(val)
        except Exception:
            try:
                return ast.literal_eval(val)
            except Exception:
                return [val]
    return series.apply(safe_parse)


def shannon_entropy(counts_dict):
    """Compute Shannon entropy from a frequency dict."""
    total = sum(counts_dict.values())
    if total == 0:
        return 0.0
    probs = [c / total for c in counts_dict.values()]
    return -sum(p * np.log2(p + 1e-12) for p in probs if p > 0)


def aggregate_grid_hour(g, all_vehicle_numbers_seen_per_grid=None, grid_id=None):
    """
    Compute rich features for a single grid-hour group.
    """
    parsed_viols = g['_parsed_violations']
    all_viol_items = [item for sublist in parsed_viols for item in sublist]
    viol_counter = Counter(all_viol_items)

    vehicle_counter = Counter(g['vehicle_type'].values)
    total_vehicles = len(g)

    # Heavy vehicles: goods, truck, tractor, construction, bus, lorry, tanker
    heavy_keywords = ['goods', 'truck', 'tractor', 'construction', 'bus', 'lorry', 'tanker', 'maxi']
    heavy_count = sum(
        cnt for vtype, cnt in vehicle_counter.items()
        if any(kw in str(vtype).lower() for kw in heavy_keywords)
    )

    # Repeat offender detection
    repeat_offender_ratio = 0.0
    if all_vehicle_numbers_seen_per_grid is not None and grid_id is not None:
        seen_before = all_vehicle_numbers_seen_per_grid.get(grid_id, set())
        current_vehicles = set(g['vehicle_number'].values)
        repeats = len(current_vehicles & seen_before)
        repeat_offender_ratio = repeats / max(len(current_vehicles), 1)

    res = {
        # Core counts
        'violation_count': len(g),
        'weighted_violation_count': g['weighted_violation_score'].sum(),
        'unique_vehicle_count': g['vehicle_number'].nunique(),

        # Vehicle composition
        'dominant_vehicle_type': g['vehicle_type'].mode().iloc[0] if not g['vehicle_type'].mode().empty else 'Unknown',
        'heavy_vehicle_ratio': heavy_count / max(total_vehicles, 1),

        # Violation composition
        'dominant_violation_type': Counter(all_viol_items).most_common(1)[0][0] if all_viol_items else 'Unknown',
        'num_offence_codes_total': g['num_offence_codes'].sum(),
        'multi_violation_ratio': (g['num_offence_codes'] > 1).mean(),
        'violation_entropy': shannon_entropy(viol_counter),

        # Monitoring
        'num_devices_active': g['device_id'].nunique(),

        # Repeat offenders
        'repeat_offender_ratio': repeat_offender_ratio,

        # Response time (historical aggregate — safe to use as grid-level feature)
        'avg_response_time': g['response_time_minutes'].mean() if 'response_time_minutes' in g.columns else np.nan,

        # Temporal
        'is_weekend': g['is_weekend'].iloc[0],
        'hour_of_day': g['hour_of_day'].iloc[0],
        'day_of_week': g['day_of_week'].iloc[0],
        'day_of_week_num': g['day_of_week_num'].iloc[0],
        'month': g['month'].iloc[0],
        'day_of_month': g['day_of_month'].iloc[0],
        'week_of_year': g['week_of_year'].iloc[0],

        # Spatial
        'police_station': g['police_station'].mode().iloc[0] if not g['police_station'].mode().empty else 'Unknown',
    }
    return pd.Series(res)


def main():
    start_time = time.time()
    print("=" * 70)
    print("PHASE 2: DENSE SPATIOTEMPORAL PANEL CONSTRUCTION")
    print("=" * 70)

    # ─────────────────────────────────────────────────────────
    # Load cleaned data
    # ─────────────────────────────────────────────────────────
    print("\n[1/6] Loading cleaned dataset...")
    df = pd.read_csv("dataset1.csv")  # Start from raw to get all columns
    # Apply same preprocessing inline (we need the parsed violations)
    df_clean = pd.read_parquet("cleaned_dataset.parquet")
    # Use the cleaned IDs to filter raw data
    clean_ids = set(df_clean['id'].values)
    df = df[df['id'].isin(clean_ids)].copy()
    del df_clean  # free memory

    print(f"  Loaded {len(df):,} cleaned records")

    # Re-derive needed columns
    df['created_datetime'] = pd.to_datetime(df['created_datetime'], errors='coerce')
    df['modified_datetime'] = pd.to_datetime(df['modified_datetime'], errors='coerce')
    df['time_hour'] = df['created_datetime'].dt.floor('h')
    df['hour_of_day'] = df['created_datetime'].dt.hour
    df['day_of_week'] = df['created_datetime'].dt.day_name()
    df['day_of_week_num'] = df['created_datetime'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin(['Saturday', 'Sunday'])
    df['month'] = df['created_datetime'].dt.month
    df['day_of_month'] = df['created_datetime'].dt.day
    df['week_of_year'] = df['created_datetime'].dt.isocalendar().week.astype(int)

    # Geohash
    print("  Computing geohash-7...")
    df['geohash_7'] = df.apply(
        lambda row: pgh.encode(row['latitude'], row['longitude'], precision=7),
        axis=1
    )

    # Parse violations
    df['_parsed_violations'] = parse_json_column(df['violation_type'])
    df['vehicle_type'] = df['vehicle_type'].fillna('Unknown')
    df['police_station'] = df['police_station'].fillna('Unknown')

    # Parse offence codes for weighted score
    df['parsed_offence_codes'] = parse_json_column(df['offence_code'])
    df['num_offence_codes'] = df['_parsed_violations'].apply(len).clip(lower=1)
    df['weighted_violation_score'] = df['num_offence_codes']

    # Response time
    df['response_time_minutes'] = (
        (df['modified_datetime'] - df['created_datetime']).dt.total_seconds() / 60.0
    ).clip(lower=0, upper=43200)

    # ─────────────────────────────────────────────────────────
    # Identify top-300 grids
    # ─────────────────────────────────────────────────────────
    print("\n[2/6] Identifying top-300 active grids...")
    grid_total_violations = df.groupby('geohash_7').size().sort_values(ascending=False)
    total_violations = grid_total_violations.sum()

    TOP_N = 300
    top_grids = grid_total_violations.head(TOP_N).index.tolist()
    top_grid_violations = grid_total_violations.head(TOP_N).sum()
    coverage = top_grid_violations / total_violations * 100

    print(f"  Total unique grids: {len(grid_total_violations):,}")
    print(f"  Top-{TOP_N} grids cover {top_grid_violations:,} / {total_violations:,} violations ({coverage:.1f}%)")

    sparse_grids = grid_total_violations.index[TOP_N:].tolist()
    print(f"  Sparse grids (event-driven): {len(sparse_grids):,}")

    # ─────────────────────────────────────────────────────────
    # Build repeat-offender lookup (per grid, which vehicles have been seen)
    # ─────────────────────────────────────────────────────────
    print("\n[3/6] Building repeat-offender index...")
    # Build cumulative set of vehicles seen per grid before each hour
    # For efficiency, we pre-compute the full set per grid (used as a static feature)
    grid_vehicle_sets = df.groupby('geohash_7')['vehicle_number'].apply(set).to_dict()

    # ─────────────────────────────────────────────────────────
    # Aggregate: Event-driven rows for ALL grids first
    # ─────────────────────────────────────────────────────────
    print("\n[4/6] Aggregating grid-hour features (all grids)...")
    grouped = df.groupby(['geohash_7', 'time_hour'])

    agg_results = []
    total_groups = len(grouped)
    for i, ((grid_id, hour), g) in enumerate(grouped):
        if i % 5000 == 0:
            print(f"  Processing group {i:,} / {total_groups:,}...", end='\r')
        res = aggregate_grid_hour(g, grid_vehicle_sets, grid_id)
        res['grid_id'] = grid_id
        res['hour'] = hour
        agg_results.append(res)

    print(f"\n  Aggregated {len(agg_results):,} grid-hour events")
    events_df = pd.DataFrame(agg_results)

    # ─────────────────────────────────────────────────────────
    # Materialize dense panel for top-300 grids
    # ─────────────────────────────────────────────────────────
    print("\n[5/6] Materializing dense panel for top-300 grids...")

    min_hour = df['time_hour'].min()
    max_hour = df['time_hour'].max()
    all_hours = pd.date_range(start=min_hour, end=max_hour, freq='h')
    print(f"  Time range: {min_hour} -> {max_hour} ({len(all_hours):,} hours)")

    # Create the dense index
    dense_index = pd.MultiIndex.from_product(
        [top_grids, all_hours],
        names=['grid_id', 'hour']
    )
    dense_df = pd.DataFrame(index=dense_index).reset_index()
    print(f"  Dense panel size: {len(dense_df):,} rows ({TOP_N} grids × {len(all_hours)} hours)")

    # Separate top-grid events and sparse-grid events
    top_events = events_df[events_df['grid_id'].isin(set(top_grids))].copy()
    sparse_events = events_df[~events_df['grid_id'].isin(set(top_grids))].copy()

    # Merge dense panel with actual events
    dense_panel = dense_df.merge(top_events, on=['grid_id', 'hour'], how='left')

    # Fill missing hours with zeros/defaults
    zero_fill_cols = [
        'violation_count', 'weighted_violation_count', 'unique_vehicle_count',
        'num_offence_codes_total', 'num_devices_active', 'violation_entropy',
        'heavy_vehicle_ratio', 'multi_violation_ratio', 'repeat_offender_ratio',
    ]
    for col in zero_fill_cols:
        if col in dense_panel.columns:
            dense_panel[col] = dense_panel[col].fillna(0)

    # Fill temporal features from the hour itself
    dense_panel['hour_of_day'] = dense_panel['hour'].dt.hour
    dense_panel['day_of_week'] = dense_panel['hour'].dt.day_name()
    dense_panel['day_of_week_num'] = dense_panel['hour'].dt.dayofweek
    dense_panel['is_weekend'] = dense_panel['day_of_week'].isin(['Saturday', 'Sunday'])
    dense_panel['month'] = dense_panel['hour'].dt.month
    dense_panel['day_of_month'] = dense_panel['hour'].dt.day
    dense_panel['week_of_year'] = dense_panel['hour'].dt.isocalendar().week.astype(int)

    # Fill categorical defaults
    dense_panel['dominant_vehicle_type'] = dense_panel['dominant_vehicle_type'].fillna('None')
    dense_panel['dominant_violation_type'] = dense_panel['dominant_violation_type'].fillna('None')
    dense_panel['police_station'] = dense_panel['police_station'].fillna('Unknown')
    dense_panel['avg_response_time'] = dense_panel['avg_response_time'].fillna(0)

    # Mark which rows are dense vs event-driven
    dense_panel['is_dense_grid'] = True

    # Sparse grids — add temporal features and mark
    sparse_events['is_dense_grid'] = False
    sparse_events['hour_of_day'] = sparse_events['hour'].dt.hour
    sparse_events['day_of_week'] = sparse_events['hour'].dt.day_name()
    sparse_events['day_of_week_num'] = sparse_events['hour'].dt.dayofweek
    sparse_events['is_weekend'] = sparse_events['day_of_week'].isin(['Saturday', 'Sunday'])
    sparse_events['month'] = sparse_events['hour'].dt.month
    sparse_events['day_of_month'] = sparse_events['hour'].dt.day
    sparse_events['week_of_year'] = sparse_events['hour'].dt.isocalendar().week.astype(int)

    # Combine
    print("\n[6/6] Combining dense + sparse panels...")
    # Align columns
    common_cols = sorted(set(dense_panel.columns) & set(sparse_events.columns))
    final_df = pd.concat([dense_panel[common_cols], sparse_events[common_cols]], ignore_index=True)
    final_df = final_df.sort_values(['grid_id', 'hour']).reset_index(drop=True)

    # Add grid centroid coordinates (decode geohash)
    print("  Adding grid centroids...")
    unique_grids = final_df['grid_id'].unique()
    centroids = {}
    for gh in unique_grids:
        try:
            lat, lng = pgh.decode(gh)
            centroids[gh] = (lat, lng)
        except Exception:
            centroids[gh] = (12.9716, 77.5946)

    final_df['grid_lat'] = final_df['grid_id'].map(lambda x: centroids[x][0])
    final_df['grid_lng'] = final_df['grid_id'].map(lambda x: centroids[x][1])

    # Save
    output_path = 'grid_hourly_panel.parquet'
    final_df.to_parquet(output_path, index=False)

    # ─────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    dense_rows = final_df[final_df['is_dense_grid'] == True].shape[0]
    sparse_rows = final_df[final_df['is_dense_grid'] == False].shape[0]
    nonzero_rows = (final_df['violation_count'] > 0).sum()
    zero_rows = (final_df['violation_count'] == 0).sum()

    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETE")
    print("=" * 70)
    print(f"  Total rows:         {len(final_df):,}")
    print(f"  Dense panel rows:   {dense_rows:,} (top-{TOP_N} grids, full timeline)")
    print(f"  Sparse event rows:  {sparse_rows:,} (remaining grids, events only)")
    print(f"  Non-zero hours:     {nonzero_rows:,}")
    print(f"  Zero-fill hours:    {zero_rows:,}")
    print(f"  Unique grids:       {final_df['grid_id'].nunique():,}")
    print(f"  Columns:            {len(final_df.columns)}")
    print(f"  Output:             {output_path}")
    print(f"  Output size:        {os.path.getsize(output_path) / 1e6:.1f} MB")
    print(f"  Time elapsed:       {elapsed:.1f}s")

    # Save report
    report = f"""# Dense Spatiotemporal Panel Report

## Panel Structure
- **Total Rows:** {len(final_df):,}
- **Dense Panel (Top-{TOP_N} Grids):** {dense_rows:,} rows (complete hourly timeline)
- **Sparse Events (Remaining Grids):** {sparse_rows:,} rows (event-driven)
- **Coverage:** Top-{TOP_N} grids cover {coverage:.1f}% of all violations
- **Time Range:** {min_hour} -> {max_hour} ({len(all_hours):,} hours)

## Sparsity Analysis (Dense Panel Only)
- **Non-zero hours:** {nonzero_rows:,}
- **Zero-fill hours:** {zero_rows:,}
- **Fill rate in dense panel:** {nonzero_rows/dense_rows*100:.2f}%

## New Features per Grid-Hour
| Feature | Description |
|---------|-------------|
| `violation_count` | Raw event count |
| `weighted_violation_count` | Sum of offence codes (severity-weighted) |
| `unique_vehicle_count` | Distinct vehicles |
| `heavy_vehicle_ratio` | Fraction of trucks/buses/goods vehicles |
| `multi_violation_ratio` | Fraction of events with 2+ offence codes |
| `violation_entropy` | Shannon entropy of violation types |
| `num_devices_active` | Number of monitoring devices active |
| `repeat_offender_ratio` | Fraction of vehicles previously seen in this grid |
| `avg_response_time` | Mean enforcement response time (minutes) |
| `grid_lat`, `grid_lng` | Grid centroid coordinates |
| `is_dense_grid` | Whether this grid has full hourly materialization |
"""
    with open('panel_report.md', 'w') as f:
        f.write(report)
    print("  Report saved to panel_report.md")


if __name__ == '__main__':
    main()
