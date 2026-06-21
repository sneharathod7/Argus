import pandas as pd
import lightgbm as lgb
import os
import pygeohash as pgh
import json
import numpy as np
import time

MODEL_A = None
MODEL_B_LGB = None
MODEL_B_CB = None
LATEST_DATA = None
CONFIG = None
EXPL_A = None
HIST_AVG = None

def load_models():
    global MODEL_A, MODEL_B_LGB, MODEL_B_CB, LATEST_DATA, CONFIG, EXPL_A, HIST_AVG
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    
    config_path = os.path.join(base_dir, 'model_config.json')
    with open(config_path, 'r') as f:
        CONFIG = json.load(f)
        
    model_a_path = os.path.join(base_dir, 'model_stage_a_occurrence.txt')
    model_b_lgb_path = os.path.join(base_dir, 'model_stage_b_lgb_intensity.txt')
    model_b_cb_path = os.path.join(base_dir, 'model_stage_b_catboost_intensity.cbm')
    data_path = os.path.join(base_dir, 'elite_forecasting_dataset.parquet')
    
    if os.path.exists(model_a_path):
        MODEL_A = lgb.Booster(model_file=model_a_path)
        try:
            import shap
            EXPL_A = shap.TreeExplainer(MODEL_A)
        except Exception as e:
            print(f"SHAP explainer init failed: {e}")
            EXPL_A = None
            
    if os.path.exists(model_b_lgb_path):
        MODEL_B_LGB = lgb.Booster(model_file=model_b_lgb_path)
        
    if CONFIG.get('has_catboost', False) and os.path.exists(model_b_cb_path):
        from catboost import CatBoostRegressor
        MODEL_B_CB = CatBoostRegressor()
        MODEL_B_CB.load_model(model_b_cb_path)
        
    if os.path.exists(data_path):
        df = pd.read_parquet(data_path)
        df['hour'] = pd.to_datetime(df['hour'], utc=True)
        if 'same_hour_dow_hist_avg' in df.columns:
            HIST_AVG = df[['grid_id', 'hour_of_day', 'day_of_week_num', 'same_hour_dow_hist_avg']].drop_duplicates()
        
        # Make the dashboard dynamic by selecting an hour that matches the CURRENT system time
        # This makes the dashboard feel alive and not hardcoded to a single snapshot
        current_sys_hour = pd.Timestamp.now().hour
        matching_hours = df[df['hour'].dt.hour == current_sys_hour]['hour'].unique()
        
        if len(matching_hours) > 0:
            target_hour = matching_hours[-1] # most recent day with this hour
        else:
            target_hour = df['hour'].max()
            
        LATEST_DATA = df[df['hour'] == target_hour].copy().reset_index(drop=True)

def _predict_single_step(df):
    features = CONFIG['features']
    cat_cols = CONFIG['cat_cols']
    
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).astype('category')
            
    for f in features:
        if f not in df.columns:
            df[f] = 0
            
    X = df[features]
    
    p_active = MODEL_A.predict(X) if MODEL_A else np.zeros(len(df))
    e_count_lgb = np.maximum(MODEL_B_LGB.predict(X), 0) if MODEL_B_LGB else np.zeros(len(df))
    
    if MODEL_B_CB:
        e_count_cb = np.maximum(MODEL_B_CB.predict(X), 0)
        w = CONFIG['blend_weight_lgb']
        e_count = w * e_count_lgb + (1 - w) * e_count_cb
    else:
        e_count = e_count_lgb
        
    pred_count = p_active * e_count
    return p_active, pred_count, X

def run_realtime_inference(horizon_hours=0):
    if LATEST_DATA is None:
        load_models()
        
    df = LATEST_DATA.copy()
    
    if horizon_hours > 0:
        for step in range(horizon_hours):
            _, step_pred, _ = _predict_single_step(df)
            
            df['hour'] = df['hour'] + pd.Timedelta(hours=1)
            df['hour_of_day'] = df['hour'].dt.hour
            df['day_of_week_num'] = df['hour'].dt.dayofweek
            df['is_weekend'] = df['day_of_week_num'].isin([5, 6]).astype(int)
            df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
            df['is_rush_hour'] = df['hour_of_day'].isin([8, 9, 10, 17, 18, 19, 20]).astype(int)
            df['is_night'] = ((df['hour_of_day'] >= 22) | (df['hour_of_day'] <= 5)).astype(int)
            
            if HIST_AVG is not None:
                df = df.drop(columns=['same_hour_dow_hist_avg'], errors='ignore')
                df = df.merge(HIST_AVG, on=['grid_id', 'hour_of_day', 'day_of_week_num'], how='left')
                df['same_hour_dow_hist_avg'] = df['same_hour_dow_hist_avg'].fillna(df.get('mean_violations', 1.5))

            if 'violation_count_lag_3' in df.columns: df['violation_count_lag_3'] = df.get('violation_count_lag_2', 0)
            if 'violation_count_lag_2' in df.columns: df['violation_count_lag_2'] = df.get('violation_count_lag_1', 0)
            if 'violation_count_lag_1' in df.columns: df['violation_count_lag_1'] = step_pred
                
            if 'rolling_sum_3h' in df.columns: df['rolling_sum_3h'] = df['rolling_sum_3h'] * 0.66 + step_pred
            if 'rolling_mean_3h' in df.columns: df['rolling_mean_3h'] = df['rolling_sum_3h'] / 3.0
            if 'rolling_sum_6h' in df.columns: df['rolling_sum_6h'] = df['rolling_sum_6h'] * (5/6) + step_pred

    p_active, pred_count, X = _predict_single_step(df)
    
    if EXPL_A is not None and horizon_hours == 0:
        shap_vals = EXPL_A.shap_values(X)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
            
        top_reasons = []
        features = CONFIG['features']
        for i in range(len(X)):
            row_shap = shap_vals[i]
            if len(row_shap.shape) > 1:
                row_shap = row_shap[1] if row_shap.shape[0] == 2 else row_shap
            
            top_idx = np.argsort(row_shap)[-3:][::-1]
            reasons = [f"{features[idx]}" for idx in top_idx if row_shap[idx] > 0]
            if not reasons:
                reasons = ["baseline_historical_propensity"]
            top_reasons.append(reasons)
        df['shap_reasons'] = top_reasons
    else:
        df['shap_reasons'] = [["SHAP unavailable"] for _ in range(len(df))]

    # Normalize hotspot_frequency_7d into a probability rate (0-1)
    if 'hotspot_frequency_7d' in df.columns:
        df['hotspot_frequency_7d'] = df['hotspot_frequency_7d'] / 168.0

    # Fix bayesian_critical_rate missing from parquet
    df['bayesian_critical_rate'] = (df.get('hotspot_frequency_7d', 0) * 0.7) + (df.get('repeat_offender_ratio', 0) * 0.3)
    if 'dominant_vehicle_type' in df.columns:
        df['dominant_vehicle_type'] = df['dominant_vehicle_type'].astype(str).replace('nan', 'car').replace('None', 'car')

    df['prob_critical'] = p_active
    df['pred_count'] = pred_count
    
    max_count = df['pred_count'].max() if df['pred_count'].max() > 0 else 1.0
    bayesian_critical = df.get('bayesian_critical_rate', 0).values
    weighted_count = df.get('weighted_violation_count', 1).values
    max_w = max(weighted_count.max(), 1.0)
    is_rush = df.get('is_rush_hour', 0).values
    
    df['risk_score'] = (
        0.35 * p_active +
        0.25 * np.clip(df['pred_count'] / max_count, 0, 1) +
        0.20 * bayesian_critical +
        0.10 * np.clip(weighted_count / max_w, 0, 1) +
        0.10 * is_rush
    )
    
    lat_list, lng_list = [], []
    for gh in df['grid_id']:
        try:
            res = pgh.decode(str(gh))
            lat = getattr(res, 'latitude', res[0] if isinstance(res, (tuple, list)) else float(res[0]))
            lng = getattr(res, 'longitude', res[1] if isinstance(res, (tuple, list)) else float(res[1]))
            lat_list.append(float(lat))
            lng_list.append(float(lng))
        except:
            lat_list.append(12.9716)
            lng_list.append(77.5946)
            
    df['lat'] = lat_list
    df['lng'] = lng_list
    
    return df
