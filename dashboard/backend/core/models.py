from pydantic import BaseModel
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
    area_name: str = "Unknown Area"
    forecast_trend: List[float] = []

class InfrastructureRecommendation(BaseModel):
    grid_id: str
    area_name: str = "Bengaluru Locality"
    police_station: str
    historical_critical_rate: float
    dominant_offender: str
    proposed_infrastructure: str
    expected_impact: str
    priority_level: str

class AnalyticsPerformance(BaseModel):
    regression_mae: float
    regression_rmse: float
    classification_recall: float
    classification_f1: float
