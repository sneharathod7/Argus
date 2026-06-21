# Traffic Operations Command Center (Argus) 🚦

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg?style=flat&logo=FastAPI)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg?style=flat&logo=Streamlit)](https://streamlit.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0%2B-FF8000.svg)](https://github.com/microsoft/LightGBM)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2%2B-F50057.svg)](https://catboost.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, machine-learning-driven spatiotemporal forecasting and patrol resource optimization platform designed for the **Bengaluru Traffic Police**. 

Argus transforms raw historical violation logs (1.1M+ records across Bengaluru) into actionable, real-time intelligence. By forecasting where and when traffic violations are likely to occur, it empowers law enforcement to switch from **reactive response** to **proactive prevention**.

---

## 📈 Key Performance Metrics

Our ML pipeline achieves massive improvements over baseline models, validated using rigorous, leakage-free chronological splitting (5-week rolling test window):

| Metric | Model Value | Baseline Value | Operational Impact |
|:---|:---|:---|:---|
| **Hotspot Occurrence PR-AUC** | **0.2645** | 0.0140 (Random) | **18× lift** in finding critical violation grids |
| **Violation Intensity MAE** | **0.3237** | 0.5338 (Historical) | **39.3% reduction** in forecasting error |
| **Recall@100 (Patrol Coverage)**| **35.48%** | 2.10% (Random) | **1-in-3 major violations** intercepted with just 100 patrol teams city-wide |
| **Temporal Stability ($\sigma$)** | **0.038** | — | High calibration stability across test periods, zero concept drift |

---

## 🏗️ Architecture

The application employs a decoupled, production-grade architecture:

```
[Spatiotemporal Features Store] ──> [Two-Stage Hurdle Model]
                                           │
  ┌────────────────────────────────────────┴────────────────────────────────────────┐
  ▼ (Stage A: Occurrence Classifier)                                                 ▼ (Stage B: Intensity Regressor)
LightGBM + Focal Loss (Is grid active?)                                Blended LightGBM + CatBoost (Tweedie)
  │                                                                                 │
  └────────────────────────────────────────┬────────────────────────────────────────┘
                                           ▼
                                 [Combined Risk Score]
                                           │
                                  [FastAPI Backend] 
                 (Serves Live ML Inference, VRP Dispatch & Recommendations)
                                           │
                                  [Streamlit Frontend]
                       (Interactive Cyberpunk Folium Command Map)
```

1. **FastAPI Backend (`dashboard/backend/`)**:
   - Houses the `ml_engine.py` responsible for serving +0h to +12h real-time predictions.
   - Executes VRP (Vehicle Routing Problem) TSP route optimization for active patrol units.
   - Computes Bayesian-smoothed historical prior analytics for infrastructure planning.
2. **Streamlit Frontend (`dashboard/frontend/`)**:
   - A modern, dark-matter styled command dashboard using Glassmorphism themes.
   - Integrates Leaflet/Folium maps for responsive geospatial rendering.
   - Features interactive sliders to simulate resource allocation and visualize SHAP-based feature attribution.

---

## 🛠️ Feature Engineering (72 Features)

Our predictive models extract signal from a complex spatiotemporal feature store:
- **Temporal Lags:** Multi-horizon lag variables (1h to 168h) capturing hour-of-day and day-of-week recurrence.
- **Rolling Statistics:** Windows (3h, 6h, 12h, 24h) computing mean and standard deviation of historical violation rates.
- **Geospatial Spillover:** Geohash neighborhood statistics reflecting spatial propagation of traffic behavior.
- **Bayesian Priors:** Leakage-free, Bayesian-smoothed offender and junction frequency profiles.
- **Temporal Encodings:** Sin/Cos transformations mapping continuous cyclical nature of time (rush hours, diurnal cycles).

---

## 📂 Project Structure

```text
Argus/
├── dashboard/
│   ├── backend/             # FastAPI Backend Server
│   │   ├── api/             # REST Endpoints (routes.py)
│   │   ├── core/            # ML Engine, Inference Loop, & TSP Dispatch (dispatch.py)
│   │   └── main.py          # Backend entry point
│   ├── frontend/            # Streamlit Frontend Client
│   │   ├── components/      # Folium Map Integration component
│   │   ├── api_client.py    # Backend connection bridge
│   │   └── app.py           # Streamlit Dashboard UI
│   │
├── docs/                    # Detailed ML Evaluation Reports & EDA Summaries
├── artifacts/               # Caches & serialized configurations
├── train_elite_model.py     # End-to-end training, Optuna HPO, and evaluation pipeline
├── generate_v3_cache.py     # Background worker to precompute hourly predictions cache
├── model_config.json        # Hyperparameter and model configuration parameters
├── requirements.txt         # Project package dependencies
└── README.md                # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Pip (Python Package Manager)

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/sneharathod7/Argus.git
cd Argus
pip install -r requirements.txt
```

### 2. Precompute Forecast Cache
To ensure instant dashboard load times, run the cache precomputation worker:
```bash
python generate_v3_cache.py
```
*(This writes a pre-calculated spatiotemporal prediction matrix to `artifacts/forecast_cache.parquet` in 2-5 minutes).*

### 3. Run the Backend API
Start the FastAPI server:
```bash
cd dashboard/backend
uvicorn main:app --reload --port 8000
```
Ensure you see `"Application startup complete."` in the terminal.

### 4. Run the Dashboard UI
In a separate terminal, launch the Streamlit server:
```bash
cd dashboard/frontend
streamlit run app.py
```
Navigate to `http://localhost:8501` in your browser.

---

## 🌟 Interactive Features & Capabilities

- **🗺️ Live Command Map:** Geospatial monitoring of Bengaluru with color-coded risk markers. Clicking on markers shows real-time SHAP explainability detailing *why* the AI flagged that location.
- **🚏 TSP Patrol Routing:** Solves the Traveling Salesperson Problem on-the-fly to calculate optimal patrol paths, minimizing travel distances between active hotspots.
- **📊 Resource Simulator:** Interactive slider dynamically plots the ROI curve of patrol effectiveness (Recall@K) to optimize squad deployments.
- **🏗️ Infrastructure Strategy:** Flags chronologically persistent violation zones, applying Bayesian evidence criteria to recommend structural fixes (like ANPR speed cameras or paid parking zones).
