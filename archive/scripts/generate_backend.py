import os

def create_files():
    base_dir = os.path.join("dashboard", "backend")
    os.makedirs(os.path.join(base_dir, "api"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "core"), exist_ok=True)

    # models.py
    models_code = """from pydantic import BaseModel
from typing import List

class GridRealtimeForecast(BaseModel):
    grid_id: str
    lat: float
    lng: float
    pred_count: float
    prob_critical: float
    severity_score: float
    urgency_score: float
    tfdi_score: float
    recommended_action: str
    confidence_level: str
    decision_reason: List[str]

class InfrastructureRecommendation(BaseModel):
    grid_id: str
    police_station: str
    historical_critical_rate: float
    dominant_offender: str
    proposed_infrastructure: str
    expected_impact: str
    priority_level: str
"""
    with open(os.path.join(base_dir, "core", "models.py"), "w") as f:
        f.write(models_code)

    # engine.py
    engine_code = """def get_vehicle_weight(vehicle_type: str) -> float:
    v = str(vehicle_type).lower()
    if 'goods' in v or 'truck' in v or 'tractor' in v or 'construction' in v:
        return 3.0
    if 'bus' in v or 'maxi' in v or 'cab' in v:
        return 2.0
    if 'car' in v or 'suv' in v:
        return 1.0
    return 0.5

def get_temporal_multiplier(hour: int) -> float:
    if 8 <= hour <= 11 or 17 <= hour <= 20:
        return 1.5
    if hour >= 22 or hour <= 6:
        return 0.5
    return 1.0

def get_road_capacity(grid_id: str) -> float:
    return 1.5

def compute_severity(prob_critical: float, pred_count: float, max_pred_count: float = 25.0) -> float:
    count_norm = min(pred_count / max_pred_count, 1.0)
    return (0.6 * prob_critical) + (0.4 * count_norm)

def compute_ous(prob_critical: float, pred_count: float, hist_rate: float, vehicle_weight: float, max_pred_count: float = 25.0) -> float:
    count_norm = min(pred_count / max_pred_count, 1.0)
    impact_norm = min(vehicle_weight / 3.0, 1.0)
    return (0.4 * count_norm) + (0.3 * prob_critical) + (0.2 * hist_rate) + (0.1 * impact_norm)

def compute_tfdi(pred_count: float, vehicle_weight: float, hour: int, road_capacity: float) -> float:
    temporal = get_temporal_multiplier(hour)
    return (pred_count * vehicle_weight * temporal) / road_capacity

def get_recommended_action(severity: float) -> str:
    if severity < 0.25:
        return "Monitoring Only"
    if severity < 0.50:
        return "Patrol Deployment"
    if severity < 0.75:
        return "Immediate Enforcement"
    return "Emergency Escalation"

def get_explainability(prob_critical: float, hist_rate: float, tfdi: float) -> tuple[str, list[str]]:
    reasons = []
    confidence = "LOW"
    
    if prob_critical > 0.60:
        reasons.append("critical_probability_above_0.60")
        confidence = "HIGH"
    elif prob_critical > 0.30:
        reasons.append("moderate_critical_probability")
        confidence = "MEDIUM"
        
    if hist_rate > 0.20:
        reasons.append("historical_hotspot_rate_above_0.20")
        confidence = "HIGH"
        
    if tfdi > 15.0:
        reasons.append("severe_congestion_impact")
        
    if not reasons:
        reasons.append("baseline_regression_forecast")
        
    return confidence, reasons

def generate_infrastructure_recommendation(grid_id: str, dominant_vehicle: str, hist_rate: float) -> str:
    v = str(dominant_vehicle).lower()
    if 'car' in v:
        return "Construct dedicated off-street parking or implement surge-pricing."
    elif 'maxi' in v or 'bus' in v or 'auto' in v:
        return "Construct designated passenger Pickup/Drop Zones."
    elif 'goods' in v or 'truck' in v:
        return "Designate specific Loading/Unloading Bays and restrict entry to off-peak hours."
    else:
        return "Implement physical infrastructure (bollards, narrowed lane widths)."
"""
    with open(os.path.join(base_dir, "core", "engine.py"), "w") as f:
        f.write(engine_code)

    # inference.py
    inference_code = """import pandas as pd
import lightgbm as lgb
import os
import pygeohash as pgh

REG_MODEL = None
CLF_MODEL = None
LATEST_DATA = None

def load_models():
    global REG_MODEL, CLF_MODEL, LATEST_DATA
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    reg_path = os.path.join(base_dir, 'stgf_lightgbm_model.txt')
    clf_path = os.path.join(base_dir, 'hotspot_classifier_model.txt')
    data_path = os.path.join(base_dir, 'forecasting_dataset.parquet')
    
    if os.path.exists(reg_path):
        REG_MODEL = lgb.Booster(model_file=reg_path)
    if os.path.exists(clf_path):
        CLF_MODEL = lgb.Booster(model_file=clf_path)
    if os.path.exists(data_path):
        df = pd.read_parquet(data_path)
        df['hour'] = pd.to_datetime(df['hour'], utc=True)
        LATEST_DATA = df.sort_values('hour').groupby('grid_id').tail(1).reset_index(drop=True)

def run_realtime_inference():
    if LATEST_DATA is None:
        load_models()
        
    df = LATEST_DATA.copy()
    cat_cols = ['grid_id', 'dominant_vehicle_type', 'dominant_violation_type', 'day_of_week']
    for col in cat_cols:
        df[col] = df[col].astype(str).astype('category')
        
    base_features = [
        'violation_count_lag_1', 'violation_count_lag_2', 'violation_count_lag_3',
        'violation_count_lag_6', 'violation_count_lag_12', 'violation_count_lag_24',
        'rolling_mean_3h', 'rolling_mean_6h', 'rolling_mean_12h', 'rolling_mean_24h',
        'rolling_sum_6h', 'rolling_sum_12h', 'rolling_sum_24h', 'rolling_max_24h',
        'violations_last_24h', 'violations_last_7d',
        'current_count_minus_rolling_mean_24h',
        'is_weekend', 'hour_of_day'
    ] + cat_cols
    
    clf_features = base_features + ['critical_rate_per_grid']
    
    if 'critical_rate_per_grid' not in df.columns:
        df['critical_rate_per_grid'] = 0.05
        
    if REG_MODEL: df['pred_count'] = REG_MODEL.predict(df[base_features])
    else: df['pred_count'] = 0
    
    if CLF_MODEL: df['prob_critical'] = CLF_MODEL.predict(df[clf_features])
    else: df['prob_critical'] = 0
    
    def get_lat_lng(gh):
        try:
            return pgh.decode(gh)
        except:
            return (12.9716, 77.5946)
            
    coords = df['grid_id'].apply(get_lat_lng)
    df['lat'] = [c[0] for c in coords]
    df['lng'] = [c[1] for c in coords]
    
    return df
"""
    with open(os.path.join(base_dir, "core", "inference.py"), "w") as f:
        f.write(inference_code)

    # routes.py
    routes_code = """from fastapi import APIRouter
from typing import List
from .core.models import GridRealtimeForecast, InfrastructureRecommendation
from .core.inference import run_realtime_inference, LATEST_DATA
from .core.engine import (get_vehicle_weight, get_road_capacity, compute_severity, 
                          compute_ous, compute_tfdi, get_recommended_action, 
                          get_explainability, generate_infrastructure_recommendation)

router = APIRouter(prefix="/api/v1")

@router.get("/forecast/realtime", response_model=List[GridRealtimeForecast])
def get_realtime_forecast():
    df = run_realtime_inference()
    
    results = []
    for _, row in df.iterrows():
        vw = get_vehicle_weight(row['dominant_vehicle_type'])
        rc = get_road_capacity(row['grid_id'])
        
        sev = compute_severity(row['prob_critical'], row['pred_count'])
        ous = compute_ous(row['prob_critical'], row['pred_count'], row['critical_rate_per_grid'], vw)
        tfdi = compute_tfdi(row['pred_count'], vw, int(row['hour_of_day']), rc)
        action = get_recommended_action(sev)
        conf, reasons = get_explainability(row['prob_critical'], row['critical_rate_per_grid'], tfdi)
        
        results.append(GridRealtimeForecast(
            grid_id=row['grid_id'],
            lat=row['lat'],
            lng=row['lng'],
            pred_count=round(row['pred_count'], 2),
            prob_critical=round(row['prob_critical'], 4),
            severity_score=round(sev, 4),
            urgency_score=round(ous, 4),
            tfdi_score=round(tfdi, 2),
            recommended_action=action,
            confidence_level=conf,
            decision_reason=reasons
        ))
        
    results.sort(key=lambda x: x.urgency_score, reverse=True)
    return results[:100] # Return top 100 for dashboard performance

@router.get("/infrastructure/recommendations", response_model=List[InfrastructureRecommendation])
def get_infra_recommendations():
    if LATEST_DATA is None:
        run_realtime_inference()
        
    from .core.inference import LATEST_DATA
    df = LATEST_DATA[LATEST_DATA['critical_rate_per_grid'] > 0.15].copy()
    
    results = []
    for _, row in df.iterrows():
        rec = generate_infrastructure_recommendation(row['grid_id'], row['dominant_vehicle_type'], row['critical_rate_per_grid'])
        results.append(InfrastructureRecommendation(
            grid_id=row['grid_id'],
            police_station="Local Jurisdiction", 
            historical_critical_rate=round(row['critical_rate_per_grid'], 4),
            dominant_offender=str(row['dominant_vehicle_type']),
            proposed_infrastructure=rec,
            expected_impact="High Reduction in Disruption",
            priority_level="High" if row['critical_rate_per_grid'] > 0.3 else "Medium"
        ))
        
    results.sort(key=lambda x: x.historical_critical_rate, reverse=True)
    return results
"""
    with open(os.path.join(base_dir, "api", "routes.py"), "w") as f:
        f.write(routes_code)

    # main.py
    main_code = """from fastapi import FastAPI
from .api.routes import router

app = FastAPI(title="Traffic Command Center API")
app.include_router(router)

@app.on_event("startup")
def startup_event():
    from .core.inference import load_models
    print("Loading ML models into memory...")
    load_models()
    print("Backend ready.")
"""
    with open(os.path.join(base_dir, "main.py"), "w") as f:
        f.write(main_code)
        
    # Create empty init files
    open(os.path.join(base_dir, "__init__.py"), 'a').close()
    open(os.path.join(base_dir, "api", "__init__.py"), 'a').close()
    open(os.path.join(base_dir, "core", "__init__.py"), 'a').close()

if __name__ == "__main__":
    create_files()
    print("FastAPI Backend Structure created successfully.")
