"""Region / AOI management endpoints."""

from fastapi import APIRouter

router = APIRouter(tags=["regions"])


@router.get("/regions")
async def list_regions():
    # TODO: implement
    return {"regions": []}
