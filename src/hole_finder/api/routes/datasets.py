"""Dataset management endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from hole_finder.api.deps import get_db
from hole_finder.api.schemas import DatasetOut
from hole_finder.db.models import Dataset

router = APIRouter(tags=["datasets"])


@router.get("/datasets")
async def list_datasets(db: AsyncSession = Depends(get_db)):
    """List all ingested LiDAR datasets."""
    result = await db.execute(select(Dataset).order_by(Dataset.created_at.desc()))
    datasets = result.scalars().all()
    return {
        "datasets": [
            DatasetOut(
                id=str(d.id),
                name=d.name,
                source=d.source.value if d.source else "unknown",
                state=d.state,
                tile_count=d.tile_count or 0,
                status=d.status.value if d.status else "unknown",
                created_at=d.created_at,
            )
            for d in datasets
        ]
    }
