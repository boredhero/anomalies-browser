"""Zip code geocoding proxy — avoids CORS issues with Census geocoder."""

import httpx
from fastapi import APIRouter, HTTPException, Query

router = APIRouter(tags=["geocode"])

CENSUS_URL = "https://geocoding.geo.census.gov/geocoder/locations/address"


@router.get("/geocode")
async def geocode_zip(
    zip: str = Query(..., min_length=5, max_length=5, pattern=r"^\d{5}$"),
):
    """Geocode a US zip code via the Census Bureau geocoder.

    Returns lat/lon coordinates for the zip code centroid.
    Free API, no auth required.
    """
    params = {
        "zip": zip,
        "benchmark": "Public_AR_Current",
        "format": "json",
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(CENSUS_URL, params=params)
            resp.raise_for_status()
        except httpx.HTTPError:
            raise HTTPException(status_code=502, detail="Census geocoder unavailable")

    data = resp.json()
    matches = data.get("result", {}).get("addressMatches", [])
    if not matches:
        raise HTTPException(status_code=404, detail="Invalid or unrecognized zip code")

    coords = matches[0].get("coordinates", {})
    address = matches[0].get("addressComponents", {})

    return {
        "lat": coords.get("y"),
        "lon": coords.get("x"),
        "city": address.get("city", ""),
        "state": address.get("state", ""),
    }
