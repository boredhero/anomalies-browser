# Magic Eyes — Development Guide

## Project Overview
LiDAR-based terrain anomaly detection platform for caves, mines, sinkholes, and terrain anomalies. Regions: Western PA, Eastern PA, West Virginia, Eastern Ohio, Upstate NY.

## License
GPL-3.0-or-later

## Architecture

### Processing Pipeline (NATIVE — no Python compute)
- **PDAL** (C++): point cloud → ground-classified DEM
- **GDAL** (C): hillshade, slope, TPI, roughness via `gdaldem`
- **WhiteboxTools** (Rust): fill_depressions, SVF, LRM, curvature
- **Rasterio** (Python): only for fill_difference subtraction (trivial)
- All derivatives run in **parallel via ProcessPoolExecutor** (8 workers)
- Results **permanently cached** on SSD — never recomputed unless `force=True`

### Detection passes consume pre-computed rasters ONLY
- 11 passes (8 classical + 3 ML), registered via `@register_pass`
- Passes NEVER compute derivatives internally
- `required_derivatives` declares what each pass needs
- Passes run in parallel via ThreadPoolExecutor
- Results fused by `ResultFuser` (DBSCAN + weighted confidence)
- TOML configs in `configs/passes/`

### Storage
- **PostGIS**: detections, ground truth, tiles, jobs — permanent across deploys
- **SSD /data**: LiDAR tiles, DEMs, derivatives — permanent
- Users can re-run tiles from UI; results overwrite old for that tile

## Deployment

```
Internet → anomalies.martinospizza.dev / holefinder.martinospizza.dev
  → .69 nginx (TLS, HSTS, security headers, certbot auto-renew)
  → .111:9747 Docker containers:
    - magic-eyes-api (FastAPI + built frontend)
    - magic-eyes-db (PostGIS 16)
    - magic-eyes-redis (Redis 7)
    - magic-eyes-worker (Celery: ingest/process/detect)
    - magic-eyes-gpu-worker (Celery: gpu queue, /dev/kfd + /dev/dri)
    - magic-eyes-autoheal
```

- All containers: `restart: unless-stopped`, Docker enabled on boot
- DB/Redis bound to 127.0.0.1 (Docker network only)
- API on port 9747 (non-obvious)
- Data on 1TB Samsung SSD mounted at /data (ext4, fstab)
- Logs: Docker volume `/app/logs/magic_eyes.log`

### CI/CD
- `develop`: push triggers tests (cancel-in-progress)
- `main`: merge triggers build → Docker image → push GHCR → deploy to .111 via .69 jumpbox
- nginx config NOT overwritten by CI (certbot manages it)
- Secrets: DEPLOY_HOST, DEPLOY_USER, DEPLOY_SSH_KEY, GHCR_TOKEN, POSTGRES_PASSWORD

### GPU
- RX 6900 XT on .111: ROCm 7.2.0, PyTorch 2.5.1+rocm6.2, 17.2GB VRAM
- `HSA_OVERRIDE_GFX_VERSION=10.3.0`
- R9 Fury on .69: NO ROCm support (GCN3, too old)

## Running Tests
```bash
# Requires GDAL + WhiteboxTools (installed on .111 and in Docker)
# Tests use the SAME native pipeline as production — no numpy fallbacks
uv run pytest tests/unit/ -v

# Run on .111 for speed (8 cores, 64GB):
ssh noah@192.168.1.111 'cd ~/anomalies-browser && git pull && uv run pytest tests/unit/ -v'
```

## Data Sources (all free, no API keys)
- USGS 3DEP: Planetary Computer STAC → COPC from Azure (signed URLs)
- PASDA, WV/NY/OH state GIS portals

## Conventions
- `uv` for Python, `pnpm` for frontend
- Top-level imports preferred (no inline imports unless circular)
- No Co-Authored-By in commits
- Push to develop only — Noah merges to main
- Detection passes are pure raster consumers, never compute derivatives
- All coordinates: WGS84 (EPSG:4326) for storage, UTM for processing
- Version info in `info.yml`, served at `/api/info`
