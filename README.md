# TerraSpectra — Hyperspectral Crop Disease Forecasting

An internship-ready MVP for precision agriculture. It demonstrates a hyperspectral data pipeline, a PyTorch hybrid 3D-CNN + Transformer segmentation model, tiled inference through FastAPI, and a React/Deck.gl GIS dashboard.

> **Scientific limitation:** the included data generator is synthetic. The model demonstrates a workflow for detecting *simulated spectral stress*, not a validated crop-disease diagnosis or a reliable three-week forecast. Real forecasting requires labeled multi-date hyperspectral imagery, field-verified disease labels, geographic calibration, and independent validation.

## Architecture

```text
Synthetic cube / GeoTIFF
        │
        ▼
Raster loader + normalization + PCA
        │
        ▼
PyTorch Hybrid 3D-CNN + Transformer
        │
        ▼
Tiled inference + stress probability raster
        │
        ▼
FastAPI (/api/predict, /api/metrics, /api/layers)
        │
        ▼
React dashboard + Deck.gl map + analytics
```

## Repository layout

- `backend/app/data.py` — reproducible synthetic hyperspectral cube and labels
- `backend/app/raster.py` — optional GeoTIFF read/write helpers (Rasterio)
- `backend/app/model.py` — hybrid 3D-CNN + Transformer segmentation network
- `backend/app/inference.py` — tiled prediction and risk metrics
- `backend/app/main.py` — FastAPI application
- `backend/scripts/train.py` — training and model checkpoint creation
- `frontend/` — React + Vite dashboard; Deck.gl map activates when a Mapbox token is supplied

## 1. Backend setup (Python 3.10+)

```bash
cd backend
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
```

Train the demo model (optional; the API can run with deterministic spectral-index fallback until a checkpoint exists):

```bash
python scripts/train.py --epochs 8
```

Start the API from the `backend` directory:

```bash
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for interactive API documentation and `http://127.0.0.1:8000/health` for health status.

## 2. Frontend setup (Node.js 18+)

```bash
cd frontend
npm install
```

Create `frontend/.env` from `.env.example`. Set `VITE_MAPBOX_TOKEN` to a Mapbox public token to enable the geographic basemap. Without it, the dashboard uses a local grid heatmap so the app remains usable.

```bash
npm run dev
```

Open the local URL printed by Vite. The API defaults to `http://127.0.0.1:8000`.

## API endpoints

- `GET /health` — service and model status
- `GET /api/farms` — demo farm metadata
- `GET /api/layers` — current demo risk layer as GeoJSON features
- `GET /api/metrics` — acreage and risk summary
- `POST /api/predict` — run tiled inference on the demo cube

## Data and model notes

The synthetic cube contains 32 bands by default (the model supports arbitrary band counts at runtime). Simulated stressed areas have altered spectral signatures and binary pixel labels. The model learns a pixel-level stress segmentation task. The 3D convolution extracts local spatial-spectral features; a Transformer encoder models relationships between spatial tokens; a convolutional head returns a stress probability for each pixel.

For real imagery, install compatible GDAL/Rasterio builds and pass a multi-band GeoTIFF through `read_hyperspectral_geotiff`. Ensure bands are calibrated, cloud/shadow masked, co-registered, and accompanied by CRS/transform metadata. The current API intentionally serves the reproducible demo cube; production upload, authentication, large-file storage, and GPU deployment are future hardening tasks.

## Evaluation checklist for internship review

1. Record train/validation loss and pixel-level precision, recall, F1, and IoU.
2. Split by field/date rather than randomly splitting neighboring pixels to avoid spatial leakage.
3. Compare against NDVI/NDRE or a 2D-CNN baseline.
4. Validate on field-surveyed disease labels and report uncertainty/calibration.
5. Benchmark tile size, overlap, throughput, CPU/GPU memory, and seam artifacts.

## Troubleshooting

- **API import error:** run `uvicorn app.main:app --reload` while your terminal is inside `backend`.
- **No trained checkpoint:** API falls back to a spectral stress heuristic; run the training script for model inference.
- **Map is blank:** set `VITE_MAPBOX_TOKEN`; without it, the grid fallback should still render.
- **Rasterio install fails on Windows:** use a compatible wheel/conda environment; the synthetic demo does not require Rasterio to run.

Run backend tests from the `backend` directory:

```bash
pytest -q
```
