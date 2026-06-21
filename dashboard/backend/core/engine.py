def get_vehicle_weight(vehicle_type: str) -> float:
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

def get_road_capacity(mean_violations: float) -> float:
    return max(min(mean_violations, 10.0), 1.0)

def compute_severity(prob_critical: float, pred_count: float, max_pred_count: float = 25.0) -> float:
    count_norm = min(pred_count / max_pred_count, 1.0)
    return (0.6 * prob_critical) + (0.4 * count_norm)

def compute_ous(risk_score: float) -> float:
    return min(max(risk_score, 0.0), 1.0)

def compute_tfdi(pred_count: float, vehicle_weight: float, hour: int, road_capacity: float) -> float:
    temporal = get_temporal_multiplier(hour)
    return (pred_count * vehicle_weight * temporal) / road_capacity

def get_recommended_action(ous: float) -> str:
    if ous < 0.25:
        return "Monitoring Only"
    if ous < 0.45:
        return "Patrol Deployment"
    if ous < 0.70:
        return "Immediate Enforcement"
    return "Emergency Escalation"

def get_explainability(prob_critical: float, shap_features: list[str]) -> tuple[str, list[str]]:
    confidence = "LOW"
    if prob_critical > 0.60:
        confidence = "HIGH"
    elif prob_critical > 0.20:
        confidence = "MEDIUM"
        
    narratives = []
    
    # Map raw SHAP features to human-readable narratives
    feature_map = {
        "same_hour_dow_hist_avg": "Strong recurring congestion pattern for this specific time and day.",
        "hour_cos": "Natural cyclical traffic peak for this time of day.",
        "hour_sin": "Cyclical transition into higher volume hours.",
        "grid_id": "Grid possesses inherent structural or zoning risk factors.",
        "hour_of_day": "Absolute time-of-day traffic surge.",
        "bayesian_active_rate": "Historically, this grid is almost always a hotspot.",
        "rolling_sum_24h": "Sustained high violation volume over the past 24 hours.",
        "ema_24h": "Long-term trend indicates escalating congestion.",
        "rolling_sum_3h": "Sudden recent surge in violations within the last 3 hours.",
        "rolling_std_24h": "High volatility and burstiness in traffic behavior.",
        "hotspot_frequency_7d": "Repeated hotspot activity observed throughout the last week.",
        "bayesian_critical_rate": "Grid has a history of reaching CRITICAL severity levels.",
        "rolling_max_168h": "Weekly peak volume suggests recurring systemic capacity failures.",
        "mean_violations": "High absolute baseline violation volume.",
        "grid_avg_response_time": "Poor historical enforcement response times encouraging repeat offenses.",
        "is_rush_hour": "Peak rush-hour traffic multiplier is currently in effect.",
        "neighbor_violation_sum": "Spillover congestion bleeding over from neighboring grids.",
        "baseline_historical_propensity": "General historical propensity matching current conditions."
    }
    
    for f in shap_features:
        if f in feature_map:
            narratives.append(feature_map[f])
        else:
            narratives.append(f"Elevated risk driven by '{f}'.")
            
    # Add an overriding narrative if none matched
    if not narratives:
        narratives.append("High baseline probability of hotspot occurrence.")
        
    return confidence, narratives

def generate_infrastructure_recommendation(grid_id: str, dominant_vehicle: str, hist_rate: float, freq_7d: float = 0.0) -> tuple[str, str]:
    """Returns (Recommendation, Expected Reduction) based on evidence."""
    v = str(dominant_vehicle).lower()
    
    if freq_7d > 0.6 or hist_rate > 0.4:
        # Extremely persistent
        if 'car' in v or 'suv' in v:
            return "Convert to Permanent Paid/Permit Parking Zone", "45% Reduction"
        elif 'truck' in v or 'goods' in v:
            return "Designate off-peak loading bays and install bollards", "60% Reduction"
        else:
            return "Install permanent physical barricades", "55% Reduction"
            
    if hist_rate > 0.25:
        # Recurring critical
        if 'bus' in v or 'maxi' in v or 'auto' in v:
            return "Construct designated transit Pickup/Drop Zones", "35% Reduction"
        else:
            return "Install Automated Camera Enforcement (ANPR)", "40% Reduction"
            
    # Moderate / Occasional
    return "Increase targeted patrol frequency during peak hours", "20% Reduction"
