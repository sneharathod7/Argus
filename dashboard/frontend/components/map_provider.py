import os
import folium
import branca
import streamlit as st
import streamlit.components.v1 as components
from streamlit_folium import st_folium
from abc import ABC, abstractmethod

def get_severity_color(action):
    if action == "Emergency Escalation": return "red"
    if action == "Immediate Enforcement": return "orange"
    if action == "Patrol Deployment": return "yellow"
    return "green"

def generate_sparkline(trend):
    if not trend or len(trend) < 2:
        return ""
    max_val = max(trend) if max(trend) > 0 else 1.0
    min_val = min(trend)
    range_val = max_val - min_val if max_val > min_val else 1.0
    
    points = []
    for i, val in enumerate(trend):
        x = int((i / (len(trend) - 1)) * 100)
        y = 30 - int(((val - min_val) / range_val) * 20) - 5
        points.append(f"{x},{y}")
        
    pts_str = " ".join(points)
    
    return f"""
    <div style="margin-top: 10px; border-top: 1px solid #444; padding-top: 5px;">
        <b style="font-size:10px; color:#aaa;">Forecast Trend (Now to +3h)</b><br>
        <svg width="100" height="30" style="background:#222; border-radius:3px;">
            <polyline points="{pts_str}" fill="none" stroke="#00c0ff" stroke-width="2"/>
            {"".join([f'<circle cx="{p.split(",")[0]}" cy="{p.split(",")[1]}" r="2" fill="#fff"/>' for p in points])}
        </svg>
    </div>
    """

class BaseMapProvider(ABC):
    @abstractmethod
    def render_hotspots(self, df):
        pass

class FoliumProvider(BaseMapProvider):
    def __init__(self, center_lat=12.9716, center_lng=77.5946, zoom=12):
        self.center_lat = center_lat
        self.center_lng = center_lng
        self.zoom = zoom

    def render_hotspots(self, df):
        # Strictly filter out coordinates outside Bengaluru to completely prevent zooming out to the ocean
        df = df[(df['lat'] > 12.0) & (df['lat'] < 14.0) & (df['lng'] > 77.0) & (df['lng'] < 78.0)].copy()
        
        m = folium.Map(location=[self.center_lat, self.center_lng], zoom_start=self.zoom, tiles="CartoDB dark_matter")
        
        # Add legend
        legend_html = '''
         <div style="position: fixed; 
                     bottom: 50px; left: 50px; width: 200px; height: 160px; 
                     background-color: rgba(30, 30, 30, 0.8); border:2px solid grey; z-index:9999; font-size:14px;
                     border-radius: 5px; padding: 10px; color: white; font-family:sans-serif;">
             <b>Hotspot Severity</b><br><br>
             <i class="fa fa-circle" style="color:red; margin-right:5px;"></i> Emergency<br>
             <i class="fa fa-circle" style="color:orange; margin-right:5px;"></i> Immediate<br>
             <i class="fa fa-circle" style="color:yellow; margin-right:5px;"></i> Patrol<br>
             <i class="fa fa-circle" style="color:green; margin-right:5px;"></i> Monitoring<br>
         </div>
         '''
        m.get_root().html.add_child(folium.Element(legend_html))

        # Add points
        for _, row in df.iterrows():
            color = get_severity_color(row['recommended_action'])
            is_active = row['recommended_action'] != 'Monitoring Only'
            
            base_radius = 8 if is_active else 6
            opacity = 0.9 if is_active else 0.5
            
            popup_html = f"""
            <div style="width:250px; font-family:sans-serif; font-size:12px;">
                <h4 style="margin:0;color:{color};">{row['recommended_action'].upper()}</h4>
                <hr style="margin:5px 0;">
                <b>Area:</b> {row.get('area_name', 'Bengaluru')} <span style="color:#aaa;font-size:10px;">({row['grid_id']})</span><br>
                <b>Predicted Violations:</b> {row['pred_count']}<br>
                <b>Critical Prob:</b> {row['prob_critical']:.1%}<br>
                <b>OUS Score:</b> {row.get('urgency_score', 0)}<br>
                <b>Confidence:</b> {row['prob_critical']:.0%} <span style="font-size:10px;color:#aaa;">({row.get('confidence_level', 'LOW')})</span><br>
                <b>Why was this flagged?</b><br>
                {"<br>".join([f"• {r}" for r in row.get('decision_reason', [])])}
                {generate_sparkline(row.get('forecast_trend', []))}
            </div>
            """
            
            # Core marker
            folium.CircleMarker(
                location=[row['lat'], row['lng']],
                radius=base_radius,
                color=color,
                fill=True,
                fill_opacity=opacity,
                popup=folium.Popup(popup_html, max_width=300)
            ).add_to(m)
            
            # Influence Radius for spillover visualization
            if is_active:
                influence_radius = base_radius * 2 * row.get('urgency_score', 0.5)
                folium.CircleMarker(
                    location=[row['lat'], row['lng']],
                    radius=base_radius + influence_radius * 10,
                    color=color,
                    weight=0,
                    fill=True,
                    fill_opacity=0.15,
                ).add_to(m)
            
        st_folium(m, width="100%", height=600, returned_objects=[])

class MapMyIndiaProvider(BaseMapProvider):
    def __init__(self, api_key, center_lat=12.9716, center_lng=77.5946, zoom=12):
        self.api_key = api_key
        self.center_lat = center_lat
        self.center_lng = center_lng
        self.zoom = zoom

    def render_hotspots(self, df):
        st.info("🗺️ **MapMyIndia Provider Active.** Plotting coordinates via MapMyIndia Interactive SDK...")
        html_code = f"""
        <div style="width:100%; height:600px; background-color:#1a1a1a; color:#fff; display:flex; flex-direction:column; align-items:center; justify-content:center; border: 2px solid #555; border-radius:10px; font-family:sans-serif;">
            <h2 style="color:#00c0ff">MapMyIndia Dashboard Layer</h2>
            <p>API Key Configured: <code>{self.api_key[:5]}********</code></p>
            <p>Ready to render <b>{len(df[df['recommended_action'] != 'Monitoring Only'])}</b> severe hotspots directly into MapMyIndia vector maps.</p>
            <p style="color:#aaa; font-size:12px;">Waiting for final production JS SDK link to fully render map tiles.</p>
        </div>
        """
        components.html(html_code, height=620)

def get_map_provider():
    mmi_key = os.environ.get("MAPMYINDIA_API_KEY", None)
    if mmi_key:
        return MapMyIndiaProvider(api_key=mmi_key)
    else:
        return FoliumProvider()
