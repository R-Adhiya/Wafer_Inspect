"""Programmatic Alembic migration runner shared by app startup and tests.

Ensures development startup and CLI (`alembic upgrade head`) execute the EXACT
same migration script rather than maintaining a separate `create_all` code path.
"""

from pathlib import Path
from typing import Optional
from alembic import command
from alembic.config import Config
from sqlalchemy.engine import Connection

BACKEND_DIR = Path(__file__).resolve().parents[2]
ALEMBIC_INI_PATH = BACKEND_DIR / "alembic.ini"
ALEMBIC_SCRIPT_LOCATION = BACKEND_DIR / "alembic"


def build_alembic_config(database_url: Optional[str] = None) -> Config:
    """Create an Alembic Config pointing at `backend/alembic.ini` and optional DB URL."""
    cfg = Config(str(ALEMBIC_INI_PATH))
    cfg.set_main_option("script_location", str(ALEMBIC_SCRIPT_LOCATION))
    if database_url:
        cfg.set_main_option("sqlalchemy.url", database_url)
    return cfg


def upgrade_database_to_head(
    database_url: Optional[str] = None,
    connection: Optional[Connection] = None,
) -> None:
    """Apply all pending Alembic migrations (`alembic upgrade head`)."""
    cfg = build_alembic_config(database_url=database_url)
    if connection is not None:
        cfg.attributes["connection"] = connection
    command.upgrade(cfg, "head")
