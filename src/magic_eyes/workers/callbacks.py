"""Progress reporting callbacks for Celery tasks."""


def update_job_progress(job_id: str, percent: float, message: str = "") -> None:
    """Update job progress in the database. Called from within tasks."""
    # TODO: implement async DB update
    pass
