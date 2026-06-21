import os
import pandas as pd
import json

LATEST_DATA = pd.DataFrame()

def load_models():
    pass

def run_realtime_inference(horizon_hours=0):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    cache_path = os.path.join(base_dir, 'artifacts', 'forecast_cache.parquet')
    
    if not os.path.exists(cache_path):
        # Fallback empty df with required columns
        return pd.DataFrame(columns=[
            'grid_id', 'lat', 'lng', 'pred_count', 'prob_critical', 'risk_score',
            'bayesian_critical_rate', 'hotspot_frequency_7d', 'dominant_vehicle_type',
            'shap_reasons', 'forecast_trend', 'area_name', 'mean_violations', 'is_rush_hour'
        ])
        
    df = pd.read_parquet(cache_path)
    df = df[df['horizon_hours'] == horizon_hours].copy()
    
    if 'shap_reasons_json' in df.columns:
        df['shap_reasons'] = [json.loads(x) if pd.notnull(x) and x != 'nan' else [] for x in df['shap_reasons_json'].astype(str)]
        
    if 'forecast_trend_json' in df.columns:
        df['forecast_trend'] = [json.loads(x) if pd.notnull(x) and x != 'nan' else [] for x in df['forecast_trend_json'].astype(str)]
        
    return df
