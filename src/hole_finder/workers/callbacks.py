"""Progress reporting callbacks for Celery tasks."""

import asyncio
from datetime import UTC, datetime

from hole_finder.db.engine import async_session_factory
from hole_finder.db.models import Job


async def _update_progress(job_id: str, percent: float, message: str) -> None:
    """Update job progress in the database."""
    from uuid import UUID
    async with async_session_factory() as session:
        job = await session.get(Job, UUID(job_id))
        if job:
            job.progress = percent
            if percent >= 100:
                from hole_finder.db.models import JobStatus
                job.status = JobStatus.COMPLETED
                job.completed_at = datetime.now(UTC)
            await session.commit()


def update_job_progress(job_id: str, percent: float, message: str = "") -> None:
    """Update job progress — sync wrapper for use in Celery tasks."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(_update_progress(job_id, percent, message))
        else:
            asyncio.run(_update_progress(job_id, percent, message))
    except RuntimeError:
        asyncio.run(_update_progress(job_id, percent, message))
