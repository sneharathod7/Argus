from fastapi import FastAPI
from api.routes import router

app = FastAPI(title="Traffic Command Center API")
app.include_router(router)

@app.on_event("startup")
def startup_event():
    from core.inference import load_models
    print("Loading ML models into memory...")
    load_models()
    print("Backend ready.")
