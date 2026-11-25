"""Application logging configuration."""

import logging
import sys

from app.core.config import settings


def setup_logging() -> None:
    """Configure Python logging with level derived from LOG_LEVEL."""
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


logger = logging.getLogger("waferinspect")
