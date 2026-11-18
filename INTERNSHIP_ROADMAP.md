# TerraSpectra — 4-week internship execution plan

This plan follows the milestones in the supplied project brief. Keep a short daily engineering log with the problem, implementation, evidence (screenshots/tests), and next task.

## Week 1 — Geospatial data pipeline and map scaffold

**Backend / ML**
- Day 1: repository setup, environment, synthetic hyperspectral cube and metadata conventions.
- Day 2: inspect GeoTIFF/HDF5 shapes, nodata values, CRS and transform; document band ordering.
- Day 3: normalization, invalid-pixel masking and PCA experiments for 200+ bands.
- Day 4: tile extraction, overlap strategy and memory profiling.
- Day 5: data-quality report and reproducible sample cube.

**Frontend / GIS**
- Day 1: Vite + React shell and dashboard layout.
- Day 2: Deck.gl map layer and Mapbox basemap configuration.
- Day 3: risk polygon styling and map legend.
- Day 4: API integration and loading/error states.
- Day 5: responsive layout and first demo review.

## Week 2 — Hybrid model

- Implement and explain the 3D convolutional spectral-spatial feature extractor.
- Add Transformer encoder tokens and pixel-level segmentation head.
- Train on synthetic patches; track loss, precision, recall, F1 and IoU.
- Compare against a spectral-index baseline and a 2D-CNN baseline.
- Frontend: render backend GeoJSON risk cells and model confidence summaries.

## Mid-project review

Demonstrate that data loading, preprocessing, patching, model forward pass and API inference work end-to-end. Include memory and latency measurements. Explicitly label synthetic results as simulated.

## Week 3 — Model refinement and analytics

- Test PCA component counts and normalization strategies.
- Add spatially aware train/validation splits when real field data becomes available.
- Evaluate false positives/false negatives and calibration.
- Add acreage-by-risk analytics and historical observations from dated input data.
- Validate map CRS alignment using known control points.

## Week 4 — API, tiling and polish

- Add upload/storage and validation for real raster inputs.
- Implement tiled inference with overlap averaging and GeoTIFF output.
- Add API limits, background jobs, logging and clear failure responses.
- Profile CPU/GPU throughput and memory; document tile-size tradeoffs.
- Finalize README, architecture diagram, demo script and limitations.

## Final review deliverables

1. Reproducible backend and frontend setup.
2. Trained model checkpoint and training log (synthetic demo only until real labels are available).
3. API documentation and sample inference response.
4. GIS risk map and acreage summary.
5. Evaluation report with dataset provenance, split strategy, metrics and known limitations.

## Suggested daily Git commit style

Use small, truthful commits tied to completed work, e.g. `feat(data): add hyperspectral cube normalization`, `feat(model): add transformer segmentation head`, `test(api): cover health and metrics routes`, `docs: record week two evaluation`. Do not manufacture commit dates or claim tests that were not run.
