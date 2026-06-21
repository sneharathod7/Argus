import os
import sys
import pandas as pd
import numpy as np
import pygeohash as pgh
import time
import json

sys.path.append(os.path.join(os.path.dirname(__file__), 'dashboard', 'backend'))
from core.ml_engine import run_realtime_inference, load_models

def main():
    print("Loading models and data...")
    load_models()
    
    print("Precomputing forecasts for horizons 0 to 12...")
    all_forecasts = []
    
    for h in range(13):
        print(f"  Computing horizon {h}h...")
        df_h = run_realtime_inference(horizon_hours=h)
        df_h['horizon_hours'] = h
        # Ensure shap_reasons is a string for parquet compatibility if it's a list
        if 'shap_reasons' in df_h.columns:
            df_h['shap_reasons_json'] = df_h['shap_reasons'].apply(json.dumps)
            df_h = df_h.drop(columns=['shap_reasons'])
        all_forecasts.append(df_h)
        
    final_df = pd.concat(all_forecasts, ignore_index=True)
    
    print("Computing forecast trends...")
    trends = final_df[final_df['horizon_hours'].isin([0, 1, 2, 3])].pivot(index='grid_id', columns='horizon_hours', values='pred_count')
    
    print("Reverse geocoding unique grids...")
    unique_grids = final_df[['grid_id', 'lat', 'lng']].drop_duplicates().reset_index(drop=True)
    
    try:
        from geopy.geocoders import Photon
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "geopy"])
        from geopy.geocoders import Photon
        
    geolocator = Photon(user_agent="flipkart_gridlock_v3_cache")
    area_names = []
    wards = []
    
    print(f"Total grids to geocode: {len(unique_grids)}")
    for i, row in unique_grids.iterrows():
        if i % 10 == 0:
            print(f"  Geocoded {i}/{len(unique_grids)}...")
        try:
            # Skip invalid coordinates
            if row['lat'] == 0.0 or row['lng'] == 0.0:
                area, ward = "Bengaluru Locality", "Bengaluru"
            else:
                location = geolocator.reverse(f"{row['lat']}, {row['lng']}", timeout=5)
                if location and location.raw.get('properties'):
                    props = location.raw['properties']
                    area = props.get('district') or props.get('locality') or props.get('neighbourhood') or props.get('city') or "Bengaluru Locality"
                    ward = props.get('county') or props.get('state') or "Bengaluru"
                else:
                    area, ward = "Bengaluru Locality", "Bengaluru"
        except Exception as e:
            print(f"    Geocoding failed for {row['lat']},{row['lng']}: {e}")
            area, ward = "Bengaluru Locality", "Bengaluru"
            
        area_names.append(area)
        wards.append(ward)
        time.sleep(0.1)
        
    unique_grids['area_name'] = area_names
    unique_grids['ward'] = wards
    
    final_df = final_df.merge(unique_grids[['grid_id', 'area_name', 'ward']], on='grid_id', how='left')
    
    trend_dict = trends.apply(lambda r: [float(r.get(0, 0)), float(r.get(1, 0)), float(r.get(2, 0)), float(r.get(3, 0))], axis=1).to_dict()
    final_df['forecast_trend_json'] = final_df['grid_id'].map(lambda g: json.dumps(trend_dict.get(g, [0,0,0,0])))

    os.makedirs('artifacts', exist_ok=True)
    cache_path = os.path.join('artifacts', 'forecast_cache.parquet')
    final_df.to_parquet(cache_path)
    
    print(f"Cache generated successfully at {cache_path}! Total rows: {len(final_df)}")

if __name__ == "__main__":
    main()
