"""Structured logging configuration — console + file output."""

import logging
import sys
from pathlib import Path

import structlog


def setup_logging(level: str = "INFO", log_dir: str | None = None) -> None:
    """Configure structlog with console + file output.

    Log files go to log_dir/magic_eyes.log (rotated daily in production
    via Docker log driver or logrotate).
    """
    # Set up stdlib logging for file output
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Console handler
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.DEBUG)
    root_logger.addHandler(console)

    # File handler (if log_dir specified or /data/magic-eyes/logs exists)
    if log_dir is None:
        for candidate in ["/data/magic-eyes/logs", "/app/logs"]:
            if Path(candidate).parent.exists():
                log_dir = candidate
                break

    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path / "magic_eyes.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        )
        root_logger.addHandler(file_handler)

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            structlog.get_level_from_name(level)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


log = structlog.get_logger()
