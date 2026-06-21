"""
Phase 7: Patrol Route Optimization (Differentiator 1)
======================================================
This script takes the top 10 highest-risk grids from our elite models 
and calculates the most efficient patrol route using the TSP
(Traveling Salesperson Problem) approximation.

Generates a map visualization of the route.
"""

import os
import sys
import numpy as np
import pandas as pd
from math import radians, cos, sin, asin, sqrt
import plotly.graph_objects as go

# Add backend to path so we can use inference
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_dir, 'dashboard', 'backend'))

from core.inference import run_realtime_inference

def haversine(lon1, lat1, lon2, lat2):
    """Calculate the great circle distance in kilometers between two points on the earth."""
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r

def solve_tsp_nearest_neighbor(points):
    """
    Greedy nearest neighbor TSP algorithm.
    points: list of tuples (id, lat, lng, score, index)
    Returns the ordered list of indices.
    """
    n = len(points)
    if n == 0:
        return []
        
    unvisited = set(range(n))
    
    # Start at the highest risk point (index 0 if pre-sorted)
    current = 0
    route = [current]
    unvisited.remove(current)
    
    while unvisited:
        nearest = None
        min_dist = float('inf')
        
        for neighbor in unvisited:
            dist = haversine(
                points[current][2], points[current][1], 
                points[neighbor][2], points[neighbor][1]
            )
            if dist < min_dist:
                min_dist = dist
                nearest = neighbor
                
        route.append(nearest)
        current = nearest
        unvisited.remove(nearest)
        
    # Return to start to complete the loop (optional for patrol, but standard for TSP)
    # route.append(route[0]) 
    return route

def main():
    print("=" * 70)
    print("PHASE 7: PATROL ROUTE OPTIMIZATION")
    print("=" * 70)
    
    print("[1/4] Running Elite Inference for current hour...")
    try:
        df = run_realtime_inference()
    except Exception as e:
        print(f"Error running inference: {e}")
        return
        
    if df.empty:
        print("No inference data available.")
        return
        
    # Sort by risk score (urgency) and take top 10
    top_grids = df.sort_values(by='risk_score', ascending=False).head(10).reset_index(drop=True)
    print(f"  Selected top {len(top_grids)} highest-risk grids for patrol route.")
    
    print("[2/4] Formatting spatial points...")
    points = []
    for i, row in top_grids.iterrows():
        points.append((row['grid_id'], row['lat'], row['lng'], row['risk_score'], i))
        
    print("[3/4] Solving TSP (Nearest Neighbor)...")
    route_indices = solve_tsp_nearest_neighbor(points)
    
    ordered_points = [points[i] for i in route_indices]
    
    # Calculate total distance
    total_dist = 0
    for i in range(len(ordered_points) - 1):
        total_dist += haversine(
            ordered_points[i][2], ordered_points[i][1],
            ordered_points[i+1][2], ordered_points[i+1][1]
        )
        
    print(f"  Route established. Total length: {total_dist:.2f} km")
    
    print("[4/4] Generating Route Visualization...")
    
    fig = go.Figure()
    
    # Draw path
    lats = [p[1] for p in ordered_points]
    lngs = [p[2] for p in ordered_points]
    texts = [f"Stop {i+1}: Grid {p[0]}<br>Risk: {p[3]:.2f}" for i, p in enumerate(ordered_points)]
    
    fig.add_trace(go.Scattermapbox(
        mode="markers+lines",
        lon=lngs,
        lat=lats,
        marker={'size': 14, 'color': 'red'},
        line={'width': 4, 'color': 'blue'},
        text=texts,
        hoverinfo='text'
    ))
    
    # Highlight start point
    fig.add_trace(go.Scattermapbox(
        mode="markers",
        lon=[ordered_points[0][2]],
        lat=[ordered_points[0][1]],
        marker={'size': 20, 'color': 'gold', 'symbol': 'star'},
        text=["START: Highest Risk Target"],
        hoverinfo='text'
    ))
    
    center_lat = sum(lats) / len(lats)
    center_lng = sum(lngs) / len(lngs)
    
    fig.update_layout(
        margin={'l':0, 't':0, 'b':0, 'r':0},
        mapbox={
            'center': {'lon': center_lng, 'lat': center_lat},
            'style': "carto-positron",
            'zoom': 11
        },
        title=f"Optimized Patrol Route (Total Dist: {total_dist:.2f} km)"
    )
    
    output_path = os.path.join(base_dir, 'patrol_route_map.html')
    fig.write_html(output_path)
    
    print("======================================================================")
    print(f"COMPLETE. Map saved to {output_path}")
    
    # Generate route itinerary text
    print("\nRecommended Itinerary:")
    for i, p in enumerate(ordered_points):
        print(f"  {i+1}. Grid {p[0]} (Risk: {p[3]:.3f})")

if __name__ == "__main__":
    main()
