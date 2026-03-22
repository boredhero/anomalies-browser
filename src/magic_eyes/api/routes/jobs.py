"""Job submission and status endpoints."""

from fastapi import APIRouter

router = APIRouter(tags=["jobs"])


@router.get("/jobs")
async def list_jobs():
    # TODO: implement
    return {"jobs": []}


@router.post("/jobs")
async def create_job():
    # TODO: implement
    return {"status": "not_implemented"}


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    # TODO: implement
    return {"job_id": job_id, "status": "not_implemented"}
