"""Celery application factory."""

from celery import Celery

from magic_eyes.config import settings

app = Celery("magic_eyes")
app.config_from_object(
    {
        "broker_url": settings.redis_url,
        "result_backend": settings.redis_url.replace("/0", "/1"),
        "task_serializer": "json",
        "result_serializer": "json",
        "accept_content": ["json"],
        "task_track_started": True,
        "task_acks_late": True,
        "worker_prefetch_multiplier": 1,
        "task_routes": {
            "magic_eyes.workers.tasks.download_tile": {"queue": "ingest"},
            "magic_eyes.workers.tasks.process_tile": {"queue": "process"},
            "magic_eyes.workers.tasks.run_detection": {"queue": "detect"},
            "magic_eyes.workers.tasks.run_ml_pass": {"queue": "gpu"},
        },
        "task_time_limit": 3600,
        "task_soft_time_limit": 3000,
    }
)

app.autodiscover_tasks(["magic_eyes.workers"])
