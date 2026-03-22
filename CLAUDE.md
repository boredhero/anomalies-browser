# Magic Eyes — Development Guide

## Project Overview
LiDAR-based terrain anomaly detection platform for automated discovery of cave entrances, mine portals, sinkholes, and other geomorphological features. Target regions: Western PA, Eastern PA, West Virginia, Eastern Ohio, Upstate NY.

## Architecture
- **Backend**: Python 3.12, FastAPI, SQLAlchemy + GeoAlchemy2, Celery + Redis
- **Frontend**: React + TypeScript, deck.gl + MapLibre, Zustand, TanStack Query, Tailwind CSS
- **Database**: PostGIS 16 on 192.168.1.111 (Docker)
- **Task Queue**: Celery with Redis broker, 4 queues (ingest, process, detect, gpu)
- **Package Manager**: uv (not pip/pipenv)
- **Detection**: Plugin-based pass system with `@register_pass` decorator

## Key Patterns
- **Detection passes** implement `DetectionPass` ABC in `src/magic_eyes/detection/base.py`
- Register via `@register_pass` decorator from `detection/registry.py`
- Each pass returns `list[Candidate]`, fused by `ResultFuser` (DBSCAN + weighted scoring)
- Pass configs are TOML files in `configs/passes/`

## Running Tests
```bash
uv run pytest tests/unit/ -v          # unit tests (fast, no external deps)
uv run pytest tests/integration/      # integration tests (needs PostGIS/Redis)
uv run pytest tests/validation/       # validation against known sites (needs data)
```

## Infrastructure
- **Remote (192.168.1.111)**: All computation — PostGIS, Redis, Celery workers, FastAPI
- **Local**: Development only (8GB RAM laptop)
- Docker compose at `~/magic-eyes-docker/docker-compose.yml` on 192.168.1.111
- Start services: `ssh noah@192.168.1.111 "cd ~/magic-eyes-docker && docker compose up -d"`

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
