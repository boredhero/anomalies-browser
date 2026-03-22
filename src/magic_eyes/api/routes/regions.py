"""Region / AOI management endpoints."""

import json
from pathlib import Path

from fastapi import APIRouter

from magic_eyes.api.schemas import RegionOut

router = APIRouter(tags=["regions"])

REGIONS_DIR = Path(__file__).parent.parent.parent.parent.parent / "configs" / "regions"


@router.get("/regions")
async def list_regions():
    """List all available regions with their GeoJSON geometries."""
    regions = []
    if REGIONS_DIR.exists():
        for f in sorted(REGIONS_DIR.glob("*.geojson")):
            with open(f) as fh:
                data = json.load(fh)
            props = data.get("properties", {})
            geom = data.get("geometry", data)
            if data.get("type") == "FeatureCollection":
                geom = data["features"][0]["geometry"]
                props = data["features"][0].get("properties", {})
            regions.append(
                RegionOut(
                    name=f.stem,
                    description=props.get("description"),
                    geometry=geom,
                )
            )
    return {"regions": regions}


@router.get("/regions/{region_name}")
async def get_region(region_name: str):
    """Get a specific region's GeoJSON."""
    region_file = REGIONS_DIR / f"{region_name}.geojson"
    if not region_file.exists():
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Region {region_name!r} not found")

    with open(region_file) as f:
        return json.load(f)
