"""FastAPI application factory."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: import passes to trigger registration
    import magic_eyes.detection.passes  # noqa: F401

    yield
    # Shutdown


def create_app() -> FastAPI:
    app = FastAPI(
        title="Magic Eyes",
        description="LiDAR terrain anomaly detection API — caves, mines, sinkholes",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register all routes
    from magic_eyes.api.routes import (
        detections,
        jobs,
        datasets,
        regions,
        validation,
        exports,
        tiles,
        raster_tiles,
        websocket,
    )

    app.include_router(detections.router, prefix="/api")
    app.include_router(jobs.router, prefix="/api")
    app.include_router(datasets.router, prefix="/api")
    app.include_router(regions.router, prefix="/api")
    app.include_router(validation.router, prefix="/api")
    app.include_router(exports.router, prefix="/api")
    app.include_router(tiles.router, prefix="/api")
    app.include_router(raster_tiles.router, prefix="/api")
    app.include_router(websocket.router)

    @app.get("/api/health")
    async def health():
        return {"status": "ok", "version": "0.1.0"}

    return app


app = create_app()
