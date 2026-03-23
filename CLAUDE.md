# Magic Eyes — Development Guide

## Project Overview
LiDAR-based terrain anomaly detection platform for automated discovery of cave entrances, mine portals, sinkholes, and other geomorphological features. Target regions: Western PA, Eastern PA, West Virginia, Eastern Ohio, Upstate NY.

## License
GNU General Public License v3.0 (GPL-3.0-or-later). See LICENSE file.

## Architecture
- **Backend**: Python 3.12, FastAPI, SQLAlchemy + GeoAlchemy2, Celery + Redis
- **Frontend**: React + TypeScript, deck.gl + MapLibre, Zustand, TanStack Query, Tailwind CSS
- **Database**: PostGIS 16 on 192.168.1.111 (Docker)
- **Task Queue**: Celery with Redis broker, 4 queues (ingest, process, detect, gpu)
- **Package Manager**: uv (not pip/pipenv), pnpm for frontend
- **Detection**: 11 plugin-based passes (8 classical + 3 ML) with `@register_pass` decorator

## Key Patterns
- **Detection passes** implement `DetectionPass` ABC in `src/magic_eyes/detection/base.py`
- Register via `@register_pass` decorator from `detection/registry.py`
- Each pass returns `list[Candidate]`, fused by `ResultFuser` (DBSCAN + weighted scoring)
- Pass configs are TOML files in `configs/passes/`
- ML passes gracefully return empty if no trained model exists

## Running Tests
```bash
uv run pytest tests/unit/ -v          # unit tests (fast, no external deps)
uv run pytest tests/integration/      # integration tests (needs PostGIS/Redis)
uv run pytest tests/validation/       # validation against known sites (needs data)
```

## Deployment
- **CI/CD**: GitHub Actions — push to `main` triggers test → build → deploy
- **Gateway (192.168.1.69)**: runs API container (Docker) + nginx reverse proxy
- **Compute (192.168.1.111)**: runs PostGIS + Redis (Docker), Celery workers (native for GPU), RX 6900 XT for ML
- **Domain**: anomalies.martinospizza.dev → .69 nginx → localhost:8000 API container
- **Docker image**: ghcr.io/boredhero/anomalies-browser:latest
- `.69` can SSH to `.111` as jumpbox for compute node management
- All Docker services have `restart: unless-stopped` and persist data in volumes
- Logs go to `/app/logs/magic_eyes.log` (Docker volume)

## Branching
- `develop`: active development, push freely
- `main`: production deploys, merge from develop

## Infrastructure
- **Compute (192.168.1.111)**: Ryzen 7 5800X3D, 64GB RAM, RX 6900 XT (16GB VRAM, ROCm)
- **Gateway (192.168.1.69)**: i7-6700K, 32GB RAM, R9 Fury (no ROCm — too old)
- Docker compose for DB/Redis at `~/magic-eyes-docker/docker-compose.yml` on .111
- Docker compose for API at `~/anomalies-browser/docker-compose.yml` on .69

## Data Sources (all free, no API keys needed)
- USGS 3DEP: `s3://usgs-lidar-public/` (COPC/EPT, no auth)
- PASDA: pasda.psu.edu (PA spatial data)
- OpenTopography, WV/NY/OH state portals

## Dependencies
- `pdal` requires system PDAL library — only installed on remote worker, not local dev
- `richdem` removed (C++ build issues on Python 3.12) — terrain derivatives implemented in numpy/scipy
- GPU deps (`torch`, `ultralytics`) are in `[project.optional-dependencies] gpu`

## Conventions
- Use `uv` for all package management (not pip, not pipenv)
- Commits should be descriptive, reference what changed and why
- Detection passes are self-contained modules in `detection/passes/`
- All coordinates in WGS84 (EPSG:4326) for storage, UTM for processing
- Mobile-friendly UI required — use responsive Tailwind patterns
- Version info in `info.yml` — displayed on website via `/api/info`
