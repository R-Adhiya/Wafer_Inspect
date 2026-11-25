"""Alembic migration environment configured from WaferInspect settings and ORM metadata."""

import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context

# Ensure backend directory is importable regardless of current working directory
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import app.db.models  # noqa: F401, E402  # Register all ORM models on Base.metadata
from app.core.config import settings  # noqa: E402
from app.db.database import Base, build_engine  # noqa: E402

config = context.config

if config.config_file_name is not None:
    try:
        fileConfig(config.config_file_name, disable_existing_loggers=False)
    except Exception:
        pass

target_metadata = Base.metadata


def _resolve_database_url() -> str:
    override = config.get_main_option("sqlalchemy.url")
    if override:
        return override
    return settings.DATABASE_URL


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (SQL script generation)."""
    url = _resolve_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode against a live SQLAlchemy Engine."""
    connectable = config.attributes.get("connection")
    if connectable is not None:
        context.configure(
            connection=connectable,
            target_metadata=target_metadata,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()
        return

    url = _resolve_database_url()
    eng = build_engine(url)
    try:
        with eng.connect() as connection:
            context.configure(
                connection=connection,
                target_metadata=target_metadata,
                compare_type=True,
            )
            with context.begin_transaction():
                context.run_migrations()
    finally:
        eng.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
