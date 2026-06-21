import os
import requests
import pandas as pd

BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000/api/v1")

def get_simulated_forecast(horizon_hours: int = 0):
    try:
        response = requests.get(f"{BASE_URL}/forecast/simulate", params={"horizon_hours": horizon_hours})
        if response.status_code == 200:
            return pd.DataFrame(response.json())
        return pd.DataFrame()
    except Exception as e:
        print("API Error:", e)
        return pd.DataFrame()

def get_escalation_forecast(grid_id: str):
    try:
        response = requests.get(f"{BASE_URL}/forecast/escalation", params={"grid_id": grid_id})
        if response.status_code == 200:
            return response.json()
        return []
    except Exception as e:
        print("API Error:", e)
        return []

def get_infra_recommendations():
    try:
        response = requests.get(f"{BASE_URL}/infrastructure/recommendations")
        if response.status_code == 200:
            return pd.DataFrame(response.json())
        return pd.DataFrame()
    except:
        return pd.DataFrame()
        
def get_dispatch_routes(num_patrols: int = 1, top_k: int = 20, horizon_hours: int = 0):
    try:
        response = requests.get(f"{BASE_URL}/dispatch/routes", params={
            "num_patrols": num_patrols,
            "top_k": top_k,
            "horizon_hours": horizon_hours
        })
        if response.status_code == 200:
            return response.json()
        return []
    except:
        return []
