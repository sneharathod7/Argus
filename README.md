# Traffic Operations Command Center V3 🚦

Welcome to the **Traffic Operations Command Center**, an elite, machine-learning-driven spatiotemporal forecasting and resource allocation dashboard built for the Bengaluru Traffic Police.

This platform leverages cutting-edge gradient boosting models (LightGBM & CatBoost) combined with geospatial analytics to predict traffic violation hotspots, model emergency escalations, and mathematically optimize patrol deployments.

---

## 🏗️ Architecture

The system operates on a dual-stack architecture:

1. **FastAPI Backend (`dashboard/backend/`)**:
   - Houses the core Machine Learning Inference Engine (`ml_engine.py`).
   - Serves high-performance REST APIs to power real-time dashboards.
   - Executes VRP (Vehicle Routing Problem) dispatch optimization and Bayesian infrastructure recommendations.
2. **Streamlit Frontend (`dashboard/frontend/`)**:
   - A hyper-modern, Cyberpunk/Glassmorphism UI built on top of Streamlit.
   - Displays real-time interactive mapping (Folium) and dynamic Plotly data visualizations.
   - Consumes the FastAPI backend to visualize +0h to +12h forecast horizons.

## 🧠 Machine Learning Engine

The core predictive power comes from a **Two-Stage Hurdle Model** architecture:

- **Stage A (Occurrence Model):** A LightGBM binary classifier that estimates the probability ($P_{active}$) of a specific geohash grid becoming a critical violation hotspot in a given hour.
- **Stage B (Intensity Model):** A blended ensemble (LightGBM + CatBoost Regressors) that predicts the raw volume/intensity of violations, conditionally activated if Stage A predicts a hotspot.

### Real-Time Inference Loop
The `run_realtime_inference()` pipeline combines the Stage A/B outputs with real-time lagged features, Bayesian prior risk multipliers (`hotspot_frequency_7d`, `repeat_offender_ratio`), and cyclical temporal features (Rush Hour, Time of Day) to output an aggregated `risk_score` per grid. 

SHAP (SHapley Additive exPlanations) is utilized in real-time to provide human-readable justification for *why* a grid was flagged (e.g., "Driven by: cyclical morning peak").

## 📂 Project Structure

```text
d:\flipkart_round2\flipkart_round2\
├── dashboard/
│   ├── backend/             # FastAPI Server & ML Engine
│   │   ├── api/             # REST Endpoints (routes.py)
│   │   ├── core/            # ML Inference (ml_engine.py) & VRP Dispatch (dispatch.py)
│   │   └── main.py          # Uvicorn entry point
│   ├── frontend/            # Streamlit Dashboard UI
│   │   ├── components/      # Folium Map Provider
│   │   ├── api_client.py    # Bridge to backend
│   │   └── app.py           # Main Dashboard UI & Styling
├── model_stage_a_*.txt      # Serialized ML Models (LightGBM/CatBoost)
├── elite_forecasting*.parquet # Spatiotemporal Feature Store
├── generate_v3_cache.py     # Background worker to precompute hourly cache
├── train_elite_model.py     # ML Training Pipeline
├── archive/                 # Deprecated scripts, raw datasets, and early models
└── docs/                    # Archived EDA and model evaluation reports
```

## 🚀 How to Run

### 1. Precompute Forecast Cache
To ensure the dashboard operates at blazing speeds, precompute the multi-hour forecast cache:
```bash
python generate_v3_cache.py
```
*(This writes a pre-calculated state to `artifacts/forecast_cache.parquet`)*

### 2. Start the Backend API
In a new terminal window, boot up the FastAPI server:
```bash
cd dashboard/backend
uvicorn main:app --reload --port 8000
```

### 3. Launch the Command Center UI
In a separate terminal, launch the Streamlit frontend:
```bash
cd dashboard/frontend
streamlit run app.py
```
Navigate to `http://localhost:8501` to view the dashboard!

---

## 🌟 Key Features

- **Command Map:** Visualizes the entire city's risk landscape using dynamically sized, heat-mapped radius circles overlaying a dark matter cartographic theme.
- **TSP Dispatch Routing:** Solves the Traveling Salesperson Problem on the fly to cluster high-risk grids and assign optimal routing itineraries to available patrol vehicles.
- **Enforcement Simulator:** An empirical ROI curve demonstrating the law of diminishing returns for deploying additional resources against the city's current risk topology.
- **Infrastructure Strategy:** Bayesian models flag historically chronic problem zones, automatically recommending permanent infrastructure changes (e.g., "Install ANPR Cameras") for zones heavily frequented by multi-axle or chronic offender vehicles.
