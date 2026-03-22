"""Detection CRUD and spatial query endpoints."""

import uuid

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from magic_eyes.db.engine import get_session
from magic_eyes.db.repositories import (
    get_detection_by_id,
    get_detections_in_bbox,
)

router = APIRouter(tags=["detections"])


@router.get("/detections")
async def list_detections(
    west: float = Query(...),
    south: float = Query(...),
    east: float = Query(...),
    north: float = Query(...),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    limit: int = Query(10000, le=50000),
    session: AsyncSession = Depends(get_session),
):
    detections = await get_detections_in_bbox(
        session, west, south, east, north,
        min_confidence=min_confidence, limit=limit,
    )
    return {"type": "FeatureCollection", "features": [
        {
            "type": "Feature",
            "id": str(d.id),
            "geometry": {"type": "Point", "coordinates": [0, 0]},  # TODO: deserialize
            "properties": {
                "feature_type": d.feature_type.value if d.feature_type else None,
                "confidence": d.confidence,
                "depth_m": d.depth_m,
                "area_m2": d.area_m2,
                "circularity": d.circularity,
                "source_passes": d.source_passes,
                "validated": d.validated,
            },
        }
        for d in detections
    ]}


@router.get("/detections/{detection_id}")
async def get_detection(
    detection_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
):
    detection = await get_detection_by_id(session, detection_id)
    if not detection:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Detection not found")
    return {
        "id": str(detection.id),
        "feature_type": detection.feature_type.value if detection.feature_type else None,
        "confidence": detection.confidence,
        "morphometrics": detection.morphometrics,
        "source_passes": detection.source_passes,
        "validated": detection.validated,
        "validation_notes": detection.validation_notes,
    }
