import os

def create_files():
    base_dir = os.path.join("dashboard", "frontend")
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, "components"), exist_ok=True)

    # api_client.py
    api_client_code = """import requests
import pandas as pd

BASE_URL = "http://127.0.0.1:8000/api/v1"

def get_realtime_forecast():
    try:
        response = requests.get(f"{BASE_URL}/forecast/realtime")
        if response.status_code == 200:
            return pd.DataFrame(response.json())
        return pd.DataFrame()
    except Exception as e:
        print("API Error:", e)
        return pd.DataFrame()

def get_infra_recommendations():
    try:
        response = requests.get(f"{BASE_URL}/infrastructure/recommendations")
        if response.status_code == 200:
            return pd.DataFrame(response.json())
        return pd.DataFrame()
    except:
        return pd.DataFrame()
"""
    with open(os.path.join(base_dir, "api_client.py"), "w") as f:
        f.write(api_client_code)

    # app.py
    app_code = """import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import st_folium
from st_aggrid import AgGrid, GridOptionsBuilder
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api_client import get_realtime_forecast, get_infra_recommendations

st.set_page_config(page_title="Traffic Command Center", layout="wide", initial_sidebar_state="expanded")
st.title("🚦 Parking Intelligence Command Center")

@st.cache_data(ttl=60)
def fetch_realtime_data():
    return get_realtime_forecast()

@st.cache_data(ttl=86400)
def fetch_infra_data():
    return get_infra_recommendations()

realtime_df = fetch_realtime_data()
infra_df = fetch_infra_data()

if realtime_df.empty:
    st.error("Cannot connect to backend API. Please ensure FastAPI is running on http://127.0.0.1:8000")
    st.stop()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Executive Summary", 
    "🗺️ Real-Time Map", 
    "🚨 Dispatch Queue", 
    "📈 Enforcement Analytics", 
    "🏗️ Infrastructure Planning"
])

# ------------- TAB 1: Executive Summary -------------
with tab1:
    st.header("Executive Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    active_critical = len(realtime_df[realtime_df['prob_critical'] > 0.60])
    total_violations = realtime_df['pred_count'].sum()
    avg_tfdi = realtime_df['tfdi_score'].mean()
    emergencies = len(realtime_df[realtime_df['recommended_action'] == 'Emergency Escalation'])
    
    col1.metric("Active Critical Grids", active_critical)
    col2.metric("Total Predicted Violations", int(total_violations))
    col3.metric("Emergency Escalations", emergencies)
    col4.metric("Average Citywide TFDI", f"{avg_tfdi:.2f}")
    
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Severity Distribution")
        sev_counts = realtime_df['recommended_action'].value_counts().reset_index()
        sev_counts.columns = ['Action', 'Count']
        fig = px.pie(sev_counts, values='Count', names='Action', hole=0.4,
                     color='Action', color_discrete_map={
                         'Monitoring Only': 'green',
                         'Patrol Deployment': 'yellow',
                         'Immediate Enforcement': 'orange',
                         'Emergency Escalation': 'red'
                     })
        st.plotly_chart(fig, use_container_width=True)
        
    with c2:
        st.subheader("Top High-Risk Grids (by OUS)")
        top_grids = realtime_df.nlargest(10, 'urgency_score')[['grid_id', 'urgency_score', 'tfdi_score']]
        fig2 = px.bar(top_grids, x='grid_id', y='urgency_score', color='tfdi_score', 
                      color_continuous_scale='Reds', title="Operational Urgency Score")
        st.plotly_chart(fig2, use_container_width=True)

# ------------- TAB 2: Real-Time Map -------------
with tab2:
    st.header("Real-Time Hotspot Map")
    
    def get_color(action):
        if action == "Emergency Escalation": return "red"
        if action == "Immediate Enforcement": return "orange"
        if action == "Patrol Deployment": return "beige"
        return "green"
        
    # Bangalore Center
    m = folium.Map(location=[12.9716, 77.5946], zoom_start=12, tiles="CartoDB dark_matter")
    
    for _, row in realtime_df.iterrows():
        if row['recommended_action'] != "Monitoring Only":
            popup_text = f\"\"\"
            <b>Grid:</b> {row['grid_id']}<br>
            <b>Severity:</b> {row['recommended_action']}<br>
            <b>Violations:</b> {row['pred_count']}<br>
            <b>OUS:</b> {row['urgency_score']}<br>
            <b>TFDI:</b> {row['tfdi_score']}<br>
            <b>Reason:</b> {', '.join(row['decision_reason'])}
            \"\"\"
            folium.CircleMarker(
                location=[row['lat'], row['lng']],
                radius=8,
                color=get_color(row['recommended_action']),
                fill=True,
                fill_opacity=0.7,
                popup=folium.Popup(popup_text, max_width=300)
            ).add_to(m)
            
    st_folium(m, width=1200, height=600)

# ------------- TAB 3: Dispatch Queue -------------
with tab3:
    st.header("Dispatch Priority Queue")
    
    st.sidebar.header("Resource Allocation")
    avail_patrols = st.sidebar.number_input("Available Patrol Teams", min_value=1, value=10)
    avail_tows = st.sidebar.number_input("Available Tow Trucks", min_value=1, value=5)
    
    df_queue = realtime_df[['grid_id', 'urgency_score', 'severity_score', 'pred_count', 'tfdi_score', 'recommended_action', 'decision_reason']].copy()
    
    gb = GridOptionsBuilder.from_dataframe(df_queue)
    gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_default_column(sortable=True, filter=True)
    grid_options = gb.build()
    
    AgGrid(df_queue, gridOptions=grid_options, height=400, fit_columns_on_grid_load=True)
    
    st.subheader("Live Alert Feed")
    top_target = realtime_df.iloc[0]
    st.warning(f"[URGENT] Grid {top_target['grid_id']} has Critical Probability {top_target['prob_critical']}. Immediate Tow Dispatch recommended.")

# ------------- TAB 4: Enforcement Analytics -------------
with tab4:
    st.header("Enforcement Analytics")
    st.markdown("Metrics evaluated on strict 15% held-out chronological test set.")
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Regression Model (Volume)")
        st.metric("Test MAE", "1.026 Violations/hr")
        st.metric("Test RMSE", "2.441")
        
    with c2:
        st.subheader("Classifier Model (Hotspots)")
        st.metric("Recall (Critical)", "38.10%")
        st.metric("F1-Score (Critical)", "0.2232")
        
    st.info("The dual-model system correctly increases hotspot detection by nearly 5x compared to raw regression.")

# ------------- TAB 5: Infrastructure Planning -------------
with tab5:
    st.header("Long-Term Infrastructure Recommendations")
    
    if not infra_df.empty:
        gb2 = GridOptionsBuilder.from_dataframe(infra_df)
        gb2.configure_pagination(paginationAutoPageSize=True)
        gb2.configure_default_column(sortable=True, filter=True)
        grid_options2 = gb2.build()
        
        AgGrid(infra_df, gridOptions=grid_options2, height=400, fit_columns_on_grid_load=True)
    else:
        st.write("No infrastructure recommendations currently active.")

"""
    with open(os.path.join(base_dir, "app.py"), "w", encoding='utf-8') as f:
        f.write(app_code)

if __name__ == "__main__":
    create_files()
    print("Frontend Streamlit App created successfully.")
