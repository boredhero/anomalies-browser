"""Dataset management endpoints."""

from fastapi import APIRouter

router = APIRouter(tags=["datasets"])


@router.get("/datasets")
async def list_datasets():
    # TODO: implement
    return {"datasets": []}
