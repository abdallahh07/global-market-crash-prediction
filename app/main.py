from fastapi import FastAPI
from app.api import api_router

app = FastAPI(
    title="Global Market Crash Prediction API",
    description="Predicts the probability of a market crash in the next 63 trading days using institutional-grade macro indicators.",
    version="1.0.0"
)

app.include_router(api_router)

@app.get("/")
def root():
    return {
        "message": "Global Market Crash Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict"
    }