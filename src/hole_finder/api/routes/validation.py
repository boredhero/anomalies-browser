"""Validation workflow — confirm/reject/annotate detections + add ground truth."""

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from geoalchemy2.shape import from_shape, to_shape
from shapely.geometry import Point
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from hole_finder.api.deps import get_db
from hole_finder.api.schemas import (
    GroundTruthCreate,
    GroundTruthSiteOut,
    ValidationRequest,
    ValidationResponse,
)
from hole_finder.db.models import (
    Detection,
    FeatureType,
    GroundTruthSite,
    GroundTruthSource,
    ValidationEvent,
    ValidationVerdict,
)

router = APIRouter(tags=["validation"])


@router.post("/detections/{detection_id}/validate", response_model=ValidationResponse)
async def validate_detection(
    detection_id: uuid.UUID,
    body: ValidationRequest,
    db: AsyncSession = Depends(get_db),
):
    """Record a validation verdict (confirm/reject/uncertain) for a detection."""
    detection = await db.get(Detection, detection_id)
    if not detection:
        raise HTTPException(status_code=404, detail="Detection not found")

    try:
        verdict = ValidationVerdict(body.verdict.upper())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid verdict: {body.verdict}")

    event = ValidationEvent(
        detection_id=detection_id,
        verdict=verdict,
        notes=body.notes,
    )
    db.add(event)

    # Update detection's validated flag based on latest verdict
    detection.validated = verdict == ValidationVerdict.CONFIRMED
    detection.validation_notes = body.notes

    await db.commit()

    return ValidationResponse(
        verdict=verdict.value,
        detection_id=str(detection_id),
    )


@router.get("/ground-truth")
async def list_ground_truth(
    west: float | None = Query(None),
    south: float | None = Query(None),
    east: float | None = Query(None),
    north: float | None = Query(None),
    limit: int = Query(1000, le=10000),
    db: AsyncSession = Depends(get_db),
):
    """List ground truth sites, optionally within a bounding box."""
    stmt = select(GroundTruthSite).limit(limit)

    if all(v is not None for v in [west, south, east, north]):
        from geoalchemy2.functions import ST_MakeEnvelope
        envelope = ST_MakeEnvelope(west, south, east, north, 4326)
        stmt = stmt.where(GroundTruthSite.geometry.ST_Within(envelope))

    result = await db.execute(stmt)
    sites = result.scalars().all()

    out = []
    for s in sites:
        try:
            pt = to_shape(s.geometry)
            lat, lon = pt.y, pt.x
        except Exception:
            lat, lon = 0.0, 0.0

        out.append(GroundTruthSiteOut(
            id=str(s.id),
            name=s.name,
            feature_type=s.feature_type.value if s.feature_type else "unknown",
            lat=lat,
            lon=lon,
            source=s.source.value if s.source else "unknown",
            metadata=s.metadata_,
        ))

    return {"type": "FeatureCollection", "features": [
        {
            "type": "Feature",
            "id": site.id,
            "geometry": {"type": "Point", "coordinates": [site.lon, site.lat]},
            "properties": {
                "name": site.name,
                "feature_type": site.feature_type,
                "source": site.source,
            },
        }
        for site in out
    ]}


@router.post("/ground-truth")
async def create_ground_truth(
    body: GroundTruthCreate,
    db: AsyncSession = Depends(get_db),
):
    """Add a new ground truth site (e.g., from map click in validation UI)."""
    try:
        feature_type = FeatureType(body.feature_type)
    except ValueError:
        feature_type = FeatureType.UNKNOWN

    site = GroundTruthSite(
        name=body.name,
        feature_type=feature_type,
        geometry=from_shape(Point(body.lon, body.lat), srid=4326),
        source=GroundTruthSource.MANUAL,
        metadata_={"notes": body.notes} if body.notes else None,
    )
    db.add(site)
    await db.commit()
    await db.refresh(site)

    return {"id": str(site.id), "name": site.name, "status": "created"}
