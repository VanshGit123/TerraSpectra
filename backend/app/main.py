from __future__ import annotations
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
from .data import make_synthetic_cube
from .inference import predict_tiled, load_model

app = FastAPI(title="TerraSpectra API", version="1.0.0",
              description="Hyperspectral crop stress mapping demo API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False,
                   allow_methods=["*"], allow_headers=["*"])

CUBE, LABELS = make_synthetic_cube()
PROBABILITY = None
MODEL_SOURCE = "not-run"
FARM = {"id": "farm-demo-01", "name": "TerraSpectra Demo Farm",
        "area_acres": 1000, "center": {"latitude": 29.9457, "longitude": 78.1642},
        "crs": "EPSG:4326", "grid_size": [CUBE.shape[0], CUBE.shape[1]]}

class PredictRequest(BaseModel):
    tile_size: int = Field(default=32, ge=8, le=128)
    overlap: int = Field(default=8, ge=0, le=64)


def current_layer():
    global PROBABILITY, MODEL_SOURCE
    if PROBABILITY is None:
        PROBABILITY, MODEL_SOURCE = predict_tiled(CUBE, tile_size=32, overlap=8)
    return PROBABILITY


@app.get("/")
def root():
    return {"name": "TerraSpectra API", "docs": "/docs", "health": "/health"}

@app.get("/health")
def health():
    return {"status": "ok", "device": "cuda" if __import__('torch').cuda.is_available() else "cpu",
            "trained_model_loaded": load_model() is not None}

@app.get("/api/farms")
def farms():
    return {"farms": [FARM]}

@app.get("/api/metrics")
def metrics():
    risk = current_layer()
    high = risk >= 0.70
    medium = (risk >= 0.40) & (risk < 0.70)
    # Each pixel represents equal area within the illustrative 1,000-acre demo farm.
    pixel_acres = FARM["area_acres"] / risk.size
    return {"farm": FARM["name"], "total_acres": FARM["area_acres"],
            "high_risk_acres": round(float(high.sum() * pixel_acres), 1),
            "medium_risk_acres": round(float(medium.sum() * pixel_acres), 1),
            "low_risk_acres": round(float((~(high | medium)).sum() * pixel_acres), 1),
            "mean_stress_probability": round(float(risk.mean()), 4),
            "high_risk_pixels": int(high.sum()), "model_source": MODEL_SOURCE}

@app.get("/api/layers")
def layers():
    risk = current_layer()
    h, w = risk.shape
    lat0, lon0 = FARM["center"]["latitude"], FARM["center"]["longitude"]
    features = []
    # Sparse cell polygons keep the response lightweight for the demo UI.
    step = 3
    for y in range(0, h, step):
        for x in range(0, w, step):
            value = float(risk[y:y+step, x:x+step].mean())
            dlat, dlon = 0.00022, 0.00022
            west = lon0 + (x - w / 2) * dlon
            east = lon0 + (min(x + step, w) - w / 2) * dlon
            north = lat0 + (h / 2 - y) * dlat
            south = lat0 + (h / 2 - min(y + step, h)) * dlat
            features.append({"type": "Feature", "properties": {"risk": round(value, 4),
                             "risk_level": "high" if value >= .70 else "medium" if value >= .40 else "low"},
                             "geometry": {"type": "Polygon", "coordinates": [[[west,south],[east,south],[east,north],[west,north],[west,south]]]}})
    return {"type": "FeatureCollection", "features": features, "model_source": MODEL_SOURCE}

@app.post("/api/predict")
def predict(request: PredictRequest):
    global PROBABILITY, MODEL_SOURCE
    try:
        PROBABILITY, MODEL_SOURCE = predict_tiled(CUBE, request.tile_size, request.overlap)
        return {"message": "Inference completed", "shape": list(PROBABILITY.shape),
                "model_source": MODEL_SOURCE, "metrics": metrics()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
