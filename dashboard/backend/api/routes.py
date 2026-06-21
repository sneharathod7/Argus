from fastapi import APIRouter, Query
from typing import List, Optional
from pydantic import BaseModel
from core.models import GridRealtimeForecast, InfrastructureRecommendation, AnalyticsPerformance
from core.inference import run_realtime_inference, LATEST_DATA
from core.engine import (get_vehicle_weight, get_road_capacity, compute_severity, 
                          compute_ous, compute_tfdi, get_recommended_action, 
                          get_explainability, generate_infrastructure_recommendation)
from core.dispatch import generate_patrol_routes

router = APIRouter(prefix="/api/v1")

@router.get("/forecast/simulate", response_model=List[GridRealtimeForecast])
def get_simulated_forecast(horizon_hours: int = Query(0, ge=0, le=24)):
    df = run_realtime_inference(horizon_hours=horizon_hours)
    
    results = []
    for _, row in df.iterrows():
        vw = get_vehicle_weight(row.get('dominant_vehicle_type', 'car'))
        rc = get_road_capacity(row.get('mean_violations', 1.5))
        
        sev = compute_severity(row['prob_critical'], row['pred_count'])
        ous = compute_ous(row.get('risk_score', 0.0))
        tfdi = compute_tfdi(row['pred_count'], vw, int(row.get('hour_of_day', 12)), rc)
        action = get_recommended_action(ous)
        conf, reasons = get_explainability(row['prob_critical'], row.get('shap_reasons', []))
        
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
            decision_reason=reasons,
            area_name=row.get('area_name', 'Bengaluru Locality'),
            forecast_trend=row.get('forecast_trend', [])
        ))
        
    results.sort(key=lambda x: x.urgency_score, reverse=True)
    return results[:200]

@router.get("/forecast/escalation")
def get_escalation_forecast(grid_id: str):
    """Returns the forecasted trajectory for a specific grid (Now, +1, +2, +3)"""
    trajectory = []
    for h in range(4):
        df = run_realtime_inference(horizon_hours=h)
        grid_row = df[df['grid_id'] == grid_id]
        if not grid_row.empty:
            trajectory.append({
                'horizon': h,
                'pred_count': round(grid_row.iloc[0]['pred_count'], 2),
                'risk_score': round(grid_row.iloc[0]['risk_score'], 4)
            })
    return trajectory

@router.get("/dispatch/routes")
def get_dispatch_routes(num_patrols: int = Query(1, ge=1, le=50), top_k: int = Query(20, ge=5, le=100), horizon_hours: int = 0):
    df = run_realtime_inference(horizon_hours=horizon_hours)
    # Get top K grids by risk score
    top_grids = df.sort_values(by='risk_score', ascending=False).head(top_k)
    
    # Generate clustered routes
    routes = generate_patrol_routes(top_grids, num_patrols)
    return routes

@router.get("/infrastructure/recommendations", response_model=List[InfrastructureRecommendation])
def get_infra_recommendations():
    df = run_realtime_inference(horizon_hours=0)
    
    # Filter for persistent problem grids
    # We use empirical history from Bayesian priors and 7d frequency
    df_infra = df[(df['bayesian_critical_rate'] > 0.15) | (df['hotspot_frequency_7d'] > 0.4)].copy()
    
    results = []
    for _, row in df_infra.iterrows():
        rec, impact = generate_infrastructure_recommendation(
            row['grid_id'], 
            row.get('dominant_vehicle_type', 'car'), 
            row.get('bayesian_critical_rate', 0.0),
            row.get('hotspot_frequency_7d', 0.0)
        )
        results.append(InfrastructureRecommendation(
            grid_id=row['grid_id'],
            area_name=row.get('area_name', 'Bengaluru Locality'),
            police_station=row.get('police_station', 'Local Jurisdiction'), 
            historical_critical_rate=round(row.get('bayesian_critical_rate', 0.0), 4),
            dominant_offender=str(row.get('dominant_vehicle_type', 'unknown')),
            proposed_infrastructure=rec,
            expected_impact=impact,
            priority_level="High" if row.get('bayesian_critical_rate', 0) > 0.3 or row.get('hotspot_frequency_7d', 0) > 0.6 else "Medium"
        ))
        
    # Sort by priority
    results.sort(key=lambda x: (x.priority_level == "Medium", -x.historical_critical_rate))
    return results[:50]

@router.get("/analytics/performance", response_model=AnalyticsPerformance)
def get_analytics_performance():
    return AnalyticsPerformance(
        regression_mae=0.3237,
        regression_rmse=1.1623,
        classification_recall=0.6903,
        classification_f1=0.2575
    )
