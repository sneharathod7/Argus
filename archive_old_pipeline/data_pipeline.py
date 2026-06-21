import pandas as pd
import numpy as np

def transform_traffic_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms the smart traffic log dataset using vectorized operations.
    Patched explicitly for UTC timezone string formatting and null lifespans.
    """
    df = df.copy()

    # 1. Datetime Configuration & Safe Timezone Dropping
    datetime_cols = ['created_datetime', 'validation_timestamp']
    for col in datetime_cols:
        # Parse all strings to UTC, then drop timezone metadata to make them naive
        df[col] = pd.to_datetime(df[col], errors='coerce', utc=True).dt.tz_localize(None)

    # Clean-up: Drop entries missing the baseline timeline target
    df = df.dropna(subset=['validation_timestamp']).reset_index(drop=True)

    # 2. Sequential Sorting
    df = df.sort_values('validation_timestamp', ascending=True).reset_index(drop=True)

    # 3. Lifespan Metric: Calculated from Validation process delay since closed_datetime is NULL
    df['violation_duration_mins'] = (df['validation_timestamp'] - df['created_datetime']).dt.total_seconds() / 60.0
    
    # Fill extreme anomalies or negative calculations with a robust median fallback
    df['violation_duration_mins'] = df['violation_duration_mins'].fillna(30.0)
    df['violation_duration_mins'] = df['violation_duration_mins'].clip(lower=1.0, upper=1440.0)

    # 4. Comprehensive Passenger Car Unit (PCU) Feature Mapping
    # Expanded to map strings found in your raw dataset sample
    eff_veh_type = df['updated_vehicle_type'].fillna(df['vehicle_type']).astype(str).str.strip().str.upper()
    pcu_mapping = {
        'BUS': 4.0, 'TRUCK': 4.0, 'HEAVY VEHICLE': 4.0, 'TANKER': 4.0,
        'MAXI-CAB': 2.5, 'TEMPO': 2.5, 'VAN': 2.5,
        'CAR': 1.0, 'SUV': 1.0, 'AUTO': 1.0, 'AUTO-RICKSHAW': 1.0, 
        'PASSENGER AUTO': 1.0, 'GOODS AUTO': 1.2,
        'TWO-WHEELER': 0.5, 'MOTORCYCLE': 0.5, 'MOTOR CYCLE': 0.5, 'SCOOTER': 0.5, 'MOPED': 0.5
    }
    df['pcu_weight'] = eff_veh_type.map(pcu_mapping).fillna(1.0).astype(np.float32)

    # 5. Cyclical Time Configurations
    hour = df['validation_timestamp'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24.0).astype(np.float32)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24.0).astype(np.float32)
    
    df['day_of_week'] = df['validation_timestamp'].dt.dayofweek.fillna(-1).astype(np.int8)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(np.int8)

    # 6. Spatio-Temporal Rolling Calculators
    df['active_concurrent_violations'] = 0
    df['rolling_pcu_load'] = 0.0

    # Ensure empty string representations are treated as genuine categorical blocks
    df['junction_name'] = df['junction_name'].fillna('No Junction').astype(str).str.strip()

    # Track indexes to calculate clean rolling features across the dataframe
    df['orig_idx'] = df.index
    sorted_df = df.sort_values(['junction_name', 'validation_timestamp'])
    sorted_df = sorted_df.set_index('validation_timestamp')
    
    rolling_grp = sorted_df.groupby('junction_name', sort=False)
    
    # Calculate rolling counts and weights within a 45-minute window
    df.loc[sorted_df['orig_idx'], 'active_concurrent_violations'] = (
        rolling_grp['id'].rolling('45min').count().to_numpy().astype(np.int32)
    )
    df.loc[sorted_df['orig_idx'], 'rolling_pcu_load'] = (
        rolling_grp['pcu_weight'].rolling('45min').sum().to_numpy().astype(np.float32)
    )

    return df.drop(columns=['orig_idx'])

def compute_disruption_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the Traffic Disruption Index (TDI) bound between 0.0 to 100.0.
    Uses max parameters instead of strict quantiles to scale metrics smoothly.
    """
    df = df.copy()

    # Spatial Scaling Component
    max_pcu = df['rolling_pcu_load'].max()
    spatial_density = (df['rolling_pcu_load'] / max_pcu).clip(lower=0.0, upper=1.0) if max_pcu > 0 else pd.Series(0.0, index=df.index)

    # Temporal Scaling Component
    max_dur = df['violation_duration_mins'].max()
    temporal_obstruction = (df['violation_duration_mins'] / max_dur).clip(lower=0.0, upper=1.0) if max_dur > 0 else pd.Series(0.0, index=df.index)

    # Network Peak Multiplier Component
    mins_past_midnight = df['validation_timestamp'].dt.hour * 60 + df['validation_timestamp'].dt.minute
    peak_mask = ((mins_past_midnight >= 480) & (mins_past_midnight <= 690)) | ((mins_past_midnight >= 1020) & (mins_past_midnight <= 1260))
    daytime_mask = (mins_past_midnight >= 691) & (mins_past_midnight <= 1019)

    demand_mult = pd.Series(0.2, index=df.index, dtype=np.float32)
    demand_mult.loc[daytime_mask] = 0.6
    demand_mult.loc[peak_mask] = 1.0

    # Combine metrics into final engineered target score
    base_tdi = (spatial_density * 0.4) + (temporal_obstruction * 0.3) + (demand_mult * 0.3)
    df['calculated_tdi'] = (base_tdi * 100.0).astype(np.float32)

    return df