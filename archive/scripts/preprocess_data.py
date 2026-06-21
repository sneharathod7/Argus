"""
Phase 1: Data Quality & Preprocessing Pipeline
================================================
Produces a clean dataset for downstream modeling.

Steps:
  1. Drop dead columns (100% null)
  2. Filter validation_status (keep 'approved' + NULL, drop rejected/duplicate/processing)
  3. Near-duplicate deduplication (vehicle_number, geohash_7, 5-min bucket)
  4. GPS IQR outlier removal within Bengaluru bbox
  5. Compute weighted_violation_score (number of offence codes per event)
  6. Compute enforcement response time (modified - created, per record)
  7. Save cleaned parquet

Output: cleaned_dataset.parquet
"""

import pandas as pd
import numpy as np
import pygeohash as pgh
import json
import ast
import os
import time


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


def main():
    start_time = time.time()
    print("=" * 70)
    print("PHASE 1: DATA QUALITY & PREPROCESSING")
    print("=" * 70)

    # ─────────────────────────────────────────────────────────
    # Load raw data
    # ─────────────────────────────────────────────────────────
    print("\n[1/7] Loading raw dataset...")
    df = pd.read_csv("dataset1.csv")
    initial_count = len(df)
    print(f"  Loaded {initial_count:,} records with {len(df.columns)} columns.")

    # ─────────────────────────────────────────────────────────
    # Step 1: Drop dead columns (100% null or useless)
    # ─────────────────────────────────────────────────────────
    print("\n[2/7] Dropping dead columns...")
    dead_cols = ['description', 'closed_datetime', 'action_taken_timestamp']
    existing_dead = [c for c in dead_cols if c in df.columns]
    df.drop(columns=existing_dead, inplace=True)
    print(f"  Dropped {len(existing_dead)} columns: {existing_dead}")

    # ─────────────────────────────────────────────────────────
    # Step 2: Filter validation_status
    # ─────────────────────────────────────────────────────────
    print("\n[3/7] Filtering validation_status...")
    if 'validation_status' in df.columns:
        status_dist_before = df['validation_status'].value_counts(dropna=False)
        print(f"  Before filtering:")
        for status, count in status_dist_before.items():
            label = status if pd.notna(status) else "NULL/NaN"
            print(f"    {label}: {count:,}")

        # Keep: 'approved' and NULL (unprocessed, not rejected)
        # Drop: 'rejected', 'duplicate', 'processing', 'created1'
        bad_statuses = ['rejected', 'duplicate', 'processing', 'created1']
        mask_keep = ~df['validation_status'].isin(bad_statuses)
        removed_count = (~mask_keep).sum()
        df = df[mask_keep].reset_index(drop=True)
        print(f"  Removed {removed_count:,} records with status in {bad_statuses}")
        print(f"  Remaining: {len(df):,} records")
    else:
        print("  Column not found, skipping.")

    # ─────────────────────────────────────────────────────────
    # Step 3: Parse datetimes and compute geohash for dedup
    # ─────────────────────────────────────────────────────────
    print("\n[4/7] Near-duplicate deduplication...")
    # Need lat/lng for geohash — drop rows missing coordinates first
    df = df.dropna(subset=['latitude', 'longitude']).reset_index(drop=True)
    print(f"  After dropping missing GPS: {len(df):,} records")

    # Parse datetime
    df['created_datetime'] = pd.to_datetime(df['created_datetime'], errors='coerce')
    df = df.dropna(subset=['created_datetime']).reset_index(drop=True)

    # Compute geohash-7 for spatial bucketing
    print("  Computing geohash-7 for deduplication...")
    df['geohash_7'] = df.apply(
        lambda row: pgh.encode(row['latitude'], row['longitude'], precision=7),
        axis=1
    )

    # Create 5-minute time buckets
    df['time_bucket_5min'] = df['created_datetime'].dt.floor('5min')

    # Dedup: same vehicle, same geohash-7 cell, within same 5-minute bucket
    before_dedup = len(df)
    df = df.drop_duplicates(
        subset=['vehicle_number', 'geohash_7', 'time_bucket_5min'],
        keep='first'
    ).reset_index(drop=True)
    dedup_removed = before_dedup - len(df)
    print(f"  Removed {dedup_removed:,} near-duplicate events ({dedup_removed/before_dedup*100:.2f}%)")
    print(f"  Remaining: {len(df):,} records")

    # Drop the temporary bucket column
    df.drop(columns=['time_bucket_5min'], inplace=True)

    # ─────────────────────────────────────────────────────────
    # Step 4: GPS IQR outlier removal within Bengaluru bbox
    # ─────────────────────────────────────────────────────────
    print("\n[5/7] GPS outlier removal...")
    # First: broad Bengaluru bbox filter
    bbox_mask = (
        (df['latitude'] > 12.0) & (df['latitude'] < 14.0) &
        (df['longitude'] > 77.0) & (df['longitude'] < 78.5)
    )
    removed_bbox = (~bbox_mask).sum()
    df = df[bbox_mask].reset_index(drop=True)
    print(f"  Removed {removed_bbox:,} records outside Bengaluru bbox")

    # Then: IQR-based outlier removal within the bbox
    for coord in ['latitude', 'longitude']:
        q1 = df[coord].quantile(0.005)
        q3 = df[coord].quantile(0.995)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        before = len(df)
        df = df[(df[coord] >= lower) & (df[coord] <= upper)].reset_index(drop=True)
        removed_iqr = before - len(df)
        if removed_iqr > 0:
            print(f"  Removed {removed_iqr:,} {coord} outliers (range: [{lower:.4f}, {upper:.4f}])")

    print(f"  Remaining: {len(df):,} records")

    # ─────────────────────────────────────────────────────────
    # Step 5: Compute weighted violation score
    # ─────────────────────────────────────────────────────────
    print("\n[6/7] Computing weighted violation scores...")
    df['parsed_violations'] = parse_json_column(df['violation_type'])
    df['num_offence_codes'] = df['parsed_violations'].apply(len)

    # Also parse offence_code for verification
    df['parsed_offence_codes'] = parse_json_column(df['offence_code'])
    df['weighted_violation_score'] = df['parsed_offence_codes'].apply(len)

    # Use the max of both as the weight (in case one is more complete)
    df['weighted_violation_score'] = df[['num_offence_codes', 'weighted_violation_score']].max(axis=1)
    # Minimum weight is 1 (at least one violation)
    df['weighted_violation_score'] = df['weighted_violation_score'].clip(lower=1)

    print(f"  Weighted violation score distribution:")
    print(f"    Mean:   {df['weighted_violation_score'].mean():.2f}")
    print(f"    Median: {df['weighted_violation_score'].median():.0f}")
    print(f"    Max:    {df['weighted_violation_score'].max():.0f}")

    # ─────────────────────────────────────────────────────────
    # Step 6: Compute enforcement response time
    # ─────────────────────────────────────────────────────────
    print("\n[7/7] Computing enforcement response time...")
    df['modified_datetime'] = pd.to_datetime(df['modified_datetime'], errors='coerce')

    # Response time = modified - created (in minutes)
    # This is a historical feature: how fast was this violation processed?
    df['response_time_minutes'] = (
        (df['modified_datetime'] - df['created_datetime']).dt.total_seconds() / 60.0
    )

    # Clip extreme values (some records may have data entry errors)
    # Cap at 30 days (43200 minutes) — anything beyond is likely an error
    df['response_time_minutes'] = df['response_time_minutes'].clip(lower=0, upper=43200)

    valid_response = df['response_time_minutes'].notna().sum()
    print(f"  Valid response times: {valid_response:,} / {len(df):,}")
    print(f"  Mean response time: {df['response_time_minutes'].mean():.1f} minutes")
    print(f"  Median response time: {df['response_time_minutes'].median():.1f} minutes")

    # ─────────────────────────────────────────────────────────
    # Enrich with additional temporal features for downstream
    # ─────────────────────────────────────────────────────────
    df['time_hour'] = df['created_datetime'].dt.floor('h')
    df['hour_of_day'] = df['created_datetime'].dt.hour
    df['day_of_week'] = df['created_datetime'].dt.day_name()
    df['day_of_week_num'] = df['created_datetime'].dt.dayofweek  # 0=Monday
    df['is_weekend'] = df['day_of_week'].isin(['Saturday', 'Sunday'])
    df['month'] = df['created_datetime'].dt.month
    df['day_of_month'] = df['created_datetime'].dt.day
    df['week_of_year'] = df['created_datetime'].dt.isocalendar().week.astype(int)

    # ─────────────────────────────────────────────────────────
    # Save cleaned dataset
    # ─────────────────────────────────────────────────────────
    output_path = 'cleaned_dataset.parquet'
    # Select columns to keep (drop intermediate parsing columns for storage efficiency)
    keep_cols = [
        'id', 'latitude', 'longitude', 'location', 'vehicle_number', 'vehicle_type',
        'violation_type', 'offence_code', 'created_datetime', 'modified_datetime',
        'device_id', 'created_by_id', 'center_code', 'police_station',
        'data_sent_to_scita', 'junction_name',
        'data_sent_to_scita_timestamp', 'updated_vehicle_number',
        'updated_vehicle_type', 'validation_status', 'validation_timestamp',
        # Computed fields
        'geohash_7', 'weighted_violation_score', 'response_time_minutes',
        'time_hour', 'hour_of_day', 'day_of_week', 'day_of_week_num',
        'is_weekend', 'month', 'day_of_month', 'week_of_year',
        'num_offence_codes'
    ]
    # Only keep columns that exist
    keep_cols = [c for c in keep_cols if c in df.columns]
    df[keep_cols].to_parquet(output_path, index=False)

    # ─────────────────────────────────────────────────────────
    # Summary Report
    # ─────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    final_count = len(df)

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)
    print(f"  Initial records:    {initial_count:,}")
    print(f"  Final records:      {final_count:,}")
    print(f"  Records removed:    {initial_count - final_count:,} ({(initial_count - final_count)/initial_count*100:.2f}%)")
    print(f"  Columns saved:      {len(keep_cols)}")
    print(f"  Output:             {output_path}")
    print(f"  Output size:        {os.path.getsize(output_path) / 1e6:.1f} MB")
    print(f"  Time elapsed:       {elapsed:.1f}s")
    print(f"\n  Breakdown of removals:")
    print(f"    Validation status filter:  {removed_count:,}")
    print(f"    Near-duplicate dedup:      {dedup_removed:,}")
    print(f"    GPS outliers:              {removed_bbox + (before - len(df)):,}")

    # Save preprocessing report
    report = f"""# Data Preprocessing Report

## Summary
- **Initial Records:** {initial_count:,}
- **Final Records:** {final_count:,}
- **Records Removed:** {initial_count - final_count:,} ({(initial_count - final_count)/initial_count*100:.2f}%)

## Removal Breakdown
| Step | Records Removed | Reason |
|------|----------------|--------|
| Validation Status | {removed_count:,} | Removed rejected/duplicate/processing/created1 |
| Near-Duplicate Dedup | {dedup_removed:,} | Same vehicle, same grid, within 5 minutes |
| GPS Outliers | {removed_bbox:,}+ | Outside Bengaluru bbox or IQR outliers |

## New Features Added
| Feature | Description |
|---------|-------------|
| `geohash_7` | Geohash precision-7 spatial grid ID |
| `weighted_violation_score` | Number of offence codes per event (severity proxy) |
| `response_time_minutes` | Modified - Created datetime (enforcement speed) |
| `num_offence_codes` | Count of individual violation types per event |

## Data Quality Assessment
- **Weighted Violation Score:** mean={df['weighted_violation_score'].mean():.2f}, max={df['weighted_violation_score'].max():.0f}
- **Response Time:** mean={df['response_time_minutes'].mean():.1f} min, median={df['response_time_minutes'].median():.1f} min
- **Time Range:** {df['created_datetime'].min()} to {df['created_datetime'].max()}
- **Unique Grids (Geohash-7):** {df['geohash_7'].nunique():,}
- **Unique Vehicles:** {df['vehicle_number'].nunique():,}
"""
    with open('preprocessing_report.md', 'w') as f:
        f.write(report)
    print("  Report saved to preprocessing_report.md")


if __name__ == '__main__':
    main()
