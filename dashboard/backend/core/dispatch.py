import numpy as np
from math import radians, cos, sin, asin, sqrt
from sklearn.cluster import KMeans

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 
    return c * r

def solve_tsp_nearest_neighbor(points):
    """
    points: list of dicts with 'lat', 'lng', 'grid_id', etc.
    """
    n = len(points)
    if n == 0: return [], 0.0
    if n == 1: return [points[0]], 0.0
        
    unvisited = set(range(n))
    current = 0
    route = [points[current]]
    unvisited.remove(current)
    total_dist = 0.0
    
    while unvisited:
        nearest = None
        min_dist = float('inf')
        
        for neighbor in unvisited:
            dist = haversine(
                points[current]['lng'], points[current]['lat'], 
                points[neighbor]['lng'], points[neighbor]['lat']
            )
            if dist < min_dist:
                min_dist = dist
                nearest = neighbor
                
        total_dist += min_dist
        route.append(points[nearest])
        current = nearest
        unvisited.remove(nearest)
        
    return route, total_dist

def generate_patrol_routes(top_grids_df, num_patrols: int):
    """
    Given a dataframe of top grids (sorted by risk), cluster them into num_patrols 
    spatial zones and generate a TSP route for each.
    """
    points = top_grids_df.to_dict('records')
    n_points = len(points)
    
    if n_points == 0:
        return []
        
    num_patrols = min(num_patrols, n_points)
    
    # Cluster spatially
    coords = np.array([[p['lat'], p['lng']] for p in points])
    
    if num_patrols > 1:
        kmeans = KMeans(n_clusters=num_patrols, random_state=42, n_init='auto')
        clusters = kmeans.fit_predict(coords)
    else:
        clusters = np.zeros(n_points, dtype=int)
        
    routes = []
    for c in range(num_patrols):
        cluster_points = [points[i] for i in range(n_points) if clusters[i] == c]
        if not cluster_points:
            continue
            
        # Ensure we start with the highest risk point in the cluster
        cluster_points.sort(key=lambda x: x.get('risk_score', 0), reverse=True)
        
        route, dist = solve_tsp_nearest_neighbor(cluster_points)
        
        # Estimate completion time: average 20km/h speed in city + 10 mins per stop
        speed_kmh = 20.0
        travel_time_min = (dist / speed_kmh) * 60
        stop_time_min = len(route) * 10
        total_time_min = travel_time_min + stop_time_min
        
        routes.append({
            'patrol_id': f"Patrol-{c+1}",
            'num_stops': len(route),
            'distance_km': round(dist, 2),
            'estimated_mins': int(total_time_min),
            'itinerary': route
        })
        
    return routes
