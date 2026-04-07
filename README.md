# Hole Finder

A modular LiDAR analysis platform that automatically detects cave entrances, mine portals, sinkholes, and other terrain anomalies using a hybrid classical + machine learning approach.

**Live at:** [holefinder.martinospizza.dev](https://holefinder.martinospizza.dev) | [anomalies.martinospizza.dev](https://anomalies.martinospizza.dev)

## What It Does

Processes free, publicly available LiDAR elevation data (USGS 3DEP, PASDA, state GIS portals) to find underground features that are invisible to the naked eye but leave subtle signatures in terrain data:

- **Cave entrances** detected via Local Relief Models, point density voids, and multi-return analysis
- **Mine portals** found through fill-difference analysis and collapse pit morphometry
- **Sinkholes** identified with multi-scale TPI, sky-view factor, and curvature analysis
- **Other anomalies** via a plugin system that makes adding new detection passes trivial

### Consumer Experience

"Find a Hole Near Me" — enter your zip code or share your location, and the system automatically downloads and processes nearby LiDAR terrain data in under 5 minutes. An animated loading screen shows real-time download progress (MB downloaded) and processing stages. When done, you're taken on a Tinder-style guided tour of the most interesting finds, swiping through detection cards while the map flies to each one.

### Advanced Playground

Full-featured interface at `/playground` with "Search this area" — scans exactly what's visible in the viewport. Sidebar filtering by feature type and confidence with zoom-adaptive visibility (zoomed in = more detections shown). Job management, validation workflows, comments, 3D terrain, and heatmap overlays.

## Architecture

### Data Pipeline
```
                          ┌─────────────────────────────────────────────┐
                          │            USGS 3DEP / PASDA / State GIS   │
                          │              (free, no API keys)            │
                          └────────────────────┬────────────────────────┘
                                               │ COPC/LAZ tiles
                                               ▼
                     ┌─────────────────────────────────────────────────────┐
                     │                   PDAL (C++)                        │
                     │          SMRF ground classify → IDW DEM             │
                     │              + filled DEM (WBT Rust)                │
                     └────────────────────┬────────────────────────────────┘
                                          │ GeoTIFF DEM
                    ┌─────────────────────┼─────────────────────┐
                    ▼                     ▼                     ▼
           ┌──────────────┐    ┌──────────────────┐   ┌─────────────────┐
           │   GDAL (C)   │    │ WhiteboxTools    │   │   Rasterio      │
           │  hillshade   │    │  (Rust)          │   │   (Python)      │
           │  slope       │    │  SVF             │   │  fill_diff =    │
           │  TPI         │    │  LRM x3          │   │  filled - DEM   │
           │  roughness   │    │  curvature x2    │   │                 │
           └──────┬───────┘    └────────┬─────────┘   └───────┬─────────┘
                  │                     │                     │
                  └─────────────────────┼─────────────────────┘
                         ALL IN PARALLEL (ProcessPoolExecutor)
                                        │
                                        ▼
                              11 derivative GeoTIFFs
                              (cached permanently on SSD)
```

### Detection Engine
```
         11 derivative rasters (read-only)
                     │
    ┌────────────────┼────────────────────────────────┐
    ▼                ▼                ▼                ▼
┌────────┐   ┌────────────┐   ┌──────────┐   ┌────────────────┐
│fill_diff│   │    LRM     │   │curvature │   │  SVF / TPI /   │
│  pass   │   │   pass     │   │  pass    │   │ point_density  │
│         │   │(cave gold  │   │          │   │ multi_return   │
│         │   │ standard)  │   │          │   │ morpho_filter  │
└────┬────┘   └─────┬──────┘   └────┬─────┘   └──────┬─────────┘
     │              │               │                 │
     └──────────────┼───────────────┼─────────────────┘
           ALL IN PARALLEL (ThreadPoolExecutor)
                    │
                    ▼
          ┌─────────────────┐
          │  Result Fuser   │
          │  DBSCAN (10m)   │
          │  + weighted     │
          │  confidence     │
          │  scoring        │
          └────────┬────────┘
                   │
                   ▼
          PostGIS detections
          (permanent, WGS84)
```

### Deployment
```
    Internet
       │
       ▼
  holefinder.martinospizza.dev
  anomalies.martinospizza.dev
       │
       ▼
┌──────────────────────┐
│  .69 (gateway)       │
│  nginx reverse proxy │
│  TLS (certbot)       │
│  HSTS + sec headers  │
└──────────┬───────────┘
           │ LAN :9747
           ▼
┌──────────────────────────────────────────────────────────┐
│  .111 (compute)                                          │
│  Ryzen 7 5800X3D · 64GB · RX 6900 XT 17GB              │
│                                                          │
│  ┌─────────────┐ ┌───────────┐ ┌──────────────────────┐ │
│  │ hole-finder  │ │ PostGIS   │ │ Redis                │ │
│  │ -api         │ │ 16        │ │ 7                    │ │
│  │ (FastAPI +   │ │ 127.0.0.1 │ │ 127.0.0.1            │ │
│  │  frontend)   │ │ :5432     │ │ :6379                │ │
│  └─────────────┘ └───────────┘ └──────────────────────┘ │
│  ┌─────────────┐ ┌─────────────┐ ┌──────────────────┐   │
│  │ celery      │ │ celery      │ │ autoheal         │   │
│  │ -worker     │ │ -gpu-worker │ │                  │   │
│  │ (4 conc)    │ │ (ROCm GPU) │ │                  │   │
│  └─────────────┘ └─────────────┘ └──────────────────┘   │
│                                                          │
│  /data (1TB SSD) ─── raw tiles, DEMs, derivatives        │
│  All containers: restart:unless-stopped                   │
└──────────────────────────────────────────────────────────┘
```

### CI/CD
```
  develop branch                    main branch
       │                                │
  push triggers                    merge triggers
       │                                │
       ▼                                ▼
  ┌──────────┐                   ┌─────────────────┐
  │  Test    │                   │ Build & Deploy  │
  │  pytest  │                   │                 │
  │  + build │                   │ 1. pnpm build   │
  │          │                   │ 2. Docker image │
  │ cancel-  │                   │ 3. Push GHCR    │
  │ in-prog  │                   │ 4. SSH .69→.111 │
  └──────────┘                   │ 5. docker up    │
                                 │ 6. health check │
                                 └─────────────────┘
```

### Detection Passes

| Pass | Method | Best For |
|------|--------|----------|
| Fill-Difference | Priority-flood sink subtraction | Sinkholes (93% recall) |
| Local Relief Model | Multi-scale trend surface removal | Cave entrances (80% confirmed) |
| Curvature | Zevenbergen & Thorne profile/plan | Concavities |
| Sky-View Factor | Horizon angle sampling | Enclosed features |
| TPI | Multi-scale topographic position | Depressions |
| Point Density | Z-score void detection | Cave/mine openings |
| Multi-Return | Anomalous return patterns | Sub-surface openings |
| Morphometric Filter | Depth/area/circularity/k-param | False positive filtering |
| Random Forest | 10-feature classifier (sklearn) | Sinkhole classification |
| U-Net | 5-channel semantic segmentation | Pixel-level detection |
| YOLOv8 | Hillshade object detection | Cave/mine bounding boxes |

### Target Regions

- Western Pennsylvania (Allegheny Plateau karst, bituminous coal belt)
- Eastern Pennsylvania (Great Valley karst, anthracite coal region)
- West Virginia (Greenbrier County karst, extensive coal mining)
- Eastern Ohio (coal mine regions, Lockport Formation karst)
- Upstate New York (Niagara Escarpment, Lockport dolomite)
- Western North Carolina (Blue Ridge karst, Spruce Pine mica mining, Piedmont gold mines)
- Western Maryland (Hagerstown Valley karst, western MD coal mining)
- Western Massachusetts (Berkshire County marble belt, pyrite and mica mines)
- South Louisiana (salt dome collapse sinkholes, Bayou Corne area)
- North Louisiana (limestone karst)
- Northern California (Modoc Plateau lava tubes, Lava Beds National Monument)
- Sierra Nevada (gold country, marble caverns, historic mines)
- Southern California Desert (desert mining districts)

## Tech Stack

**Backend:** Python 3.13, FastAPI, SQLAlchemy + GeoAlchemy2, asyncpg, PostGIS 16, Celery + Redis, PDAL 2.10, GDAL 3.12, WhiteboxTools

**Frontend:** React + TypeScript, MapLibre GL JS (MVT vector tiles), deck.gl (heatmap), framer-motion (swipe cards), Zustand, TanStack Query, Tailwind CSS v4

**ML:** scikit-learn (Random Forest), PyTorch + ROCm 7.2 (U-Net, YOLOv8) — infrastructure ready, models not yet trained

**Resilience:** httpx-retries with 4 Overpass API mirror rotation, 7-day file cache, robust CRS handling for compound CRS (UTM+NAVD88), 3-tier fill_depressions fallback (WBT → WBT Planchon-Darboux → skimage)

**Infrastructure:** Docker (pdal/pdal:latest base), GitHub Actions CI/CD, nginx reverse proxy, Ryzen 5800X3D compute node

## API

Interactive API docs available at `/api/docs` (Swagger UI) when running locally.

### Core Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/detections` | Query detections in a bounding box (GeoJSON FeatureCollection) |
| GET | `/api/detections/{id}` | Full detection detail with pass results and validation history |
| GET | `/api/detections/count` | Fast count of detections near a point (for area availability check) |

### Map Tiles

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/tiles/{z}/{x}/{y}.mvt` | Mapbox Vector Tiles for detection rendering (PostGIS ST_AsMVT) |
| GET | `/api/tiles/ground-truth/{z}/{x}/{y}.mvt` | Ground truth site vector tiles |
| GET | `/api/raster/{layer}/{z}/{x}/{y}.png` | Hillshade and terrain-RGB raster tiles |

### Consumer Flow

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/geocode?zip=15208` | Zip code geocoding via US Census Bureau (server-side proxy) |
| POST | `/api/explore/scan` | Start auto-processing job for viewport area (radius derived from viewport, 12 tile cap) |

### Job Management

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/jobs` | List processing jobs |
| POST | `/api/jobs` | Submit a new processing job (region, draw polygon, or pin) |
| GET | `/api/jobs/{id}` | Get job status and progress |
| POST | `/api/jobs/{id}/cancel` | Cancel a running job |
| WS | `/ws/jobs` | WebSocket for real-time job progress (stage, %, completion) |

### Validation & Comments

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/detections/{id}/validate` | Validate detection (confirm/reject/uncertain + notes) |
| GET/POST | `/api/detections/{id}/comments` | Read/add comments on a detection |
| POST | `/api/detections/{id}/save` | Bookmark a detection |

### Data & Export

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/regions` | List available region polygons (13 regions across 9 states) |
| GET | `/api/regions/{name}` | Get region GeoJSON boundary |
| GET/POST | `/api/ground-truth` | Ground truth site CRUD |
| GET | `/api/export/geojson` | Export detections as GeoJSON file |
| GET | `/api/export/csv` | Export detections as CSV file |

## Quick Start

```bash
# Clone
git clone https://github.com/boredhero/anomalies-browser.git
cd anomalies-browser

# Backend
uv sync --extra dev
uv run pytest tests/unit/ -v     # 177 tests

# Frontend
cd frontend && pnpm install && pnpm dev
```

## Data Sources

All data sources are free and require no API keys:

- **USGS 3DEP** via Planetary Computer STAC API (COPC from `s3://usgs-lidar-public/`)
- **PASDA** (Pennsylvania Spatial Data Access)
- **WV/NY/OH** state GIS portals
- **NC OneMap** (North Carolina statewide tile index)
- **MD iMAP** (Maryland enterprise GIS LiDAR portal)

## Validation

13 LiDAR-visible validation sites with natural exposed entrances across PA, WV, NC, MA, LA, and CA — wild caves, exposed mine portals, and open sinkholes. Commercialized show caves with buildings over entrances were intentionally excluded (LiDAR sees rooftops, not cave mouths). Includes 1 field-verified discovery: an undocumented cave entrance in Allegheny Cemetery, Pittsburgh (ground-truthed April 2026). Bulk validation against 111,000+ PASDA karst features, 11,249 PA abandoned mines, and USGS MRDS records across all target states.

## License

[GNU General Public License v3.0](LICENSE)
