"""Data access layer for spatial queries."""

import uuid

from geoalchemy2.functions import ST_DWithin, ST_GeogFromWKB, ST_MakeEnvelope
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from magic_eyes.db.models import Detection, FeatureType, GroundTruthSite


async def get_detections_in_bbox(
    session: AsyncSession,
    west: float,
    south: float,
    east: float,
    north: float,
    feature_types: list[FeatureType] | None = None,
    min_confidence: float = 0.0,
    limit: int = 10000,
) -> list[Detection]:
    """Query detections within a bounding box."""
    envelope = ST_MakeEnvelope(west, south, east, north, 4326)
    stmt = (
        select(Detection)
        .where(Detection.geometry.ST_Within(envelope))
        .where(Detection.confidence >= min_confidence)
        .order_by(Detection.confidence.desc())
        .limit(limit)
    )
    if feature_types:
        stmt = stmt.where(Detection.feature_type.in_(feature_types))
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def get_detections_near_point(
    session: AsyncSession,
    lat: float,
    lon: float,
    radius_m: float = 200.0,
) -> list[Detection]:
    """Query detections within radius_m meters of a point."""
    point_wkt = f"SRID=4326;POINT({lon} {lat})"
    stmt = select(Detection).where(
        ST_DWithin(
            ST_GeogFromWKB(Detection.geometry),
            ST_GeogFromWKB(point_wkt),
            radius_m,
        )
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def get_ground_truth_near_point(
    session: AsyncSession,
    lat: float,
    lon: float,
    radius_m: float = 200.0,
) -> list[GroundTruthSite]:
    """Query ground truth sites within radius_m meters of a point."""
    point_wkt = f"SRID=4326;POINT({lon} {lat})"
    stmt = select(GroundTruthSite).where(
        ST_DWithin(
            ST_GeogFromWKB(GroundTruthSite.geometry),
            ST_GeogFromWKB(point_wkt),
            radius_m,
        )
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def get_detection_by_id(
    session: AsyncSession, detection_id: uuid.UUID
) -> Detection | None:
    return await session.get(Detection, detection_id)
