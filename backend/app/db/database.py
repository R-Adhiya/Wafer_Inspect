"""SQLAlchemy 2.0 database engine, session factory, and FastAPI dependency.

Supports SQLite (default: `<repo root>/data/waferinspect.db`) and PostgreSQL
simply by switching `DATABASE_URL` in environment configuration.
"""

from pathlib import Path
from typing import Any, Generator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import ROOT_DIR, settings


class Base(DeclarativeBase):
    """Declarative base shared by all ORM models and Alembic migrations."""


def _is_sqlite_url(url: str) -> bool:
    return url.strip().lower().startswith("sqlite")


def _ensure_sqlite_parent_dir(url: str) -> None:
    """Create parent directory for file-backed SQLite URLs if it does not yet exist."""
    if not _is_sqlite_url(url) or ":memory:" in url:
        return
    # Strip sqlite:/// or sqlite+pysqlite:///
    prefix_sep = ":///"
    if prefix_sep in url:
        raw_path = url.split(prefix_sep, 1)[1].split("?", 1)[0]
        if raw_path and raw_path != ":memory:":
            db_path = Path(raw_path)
            if not db_path.is_absolute():
                db_path = (ROOT_DIR / db_path).resolve()
            db_path.parent.mkdir(parents=True, exist_ok=True)


def build_engine(database_url: str) -> Engine:
    """Create a configured SQLAlchemy 2.0 Engine with SQLite pragmas when using SQLite."""
    _ensure_sqlite_parent_dir(database_url)
    connect_args: dict[str, Any] = {}
    if _is_sqlite_url(database_url):
        connect_args["check_same_thread"] = False

    eng = create_engine(
        database_url,
        connect_args=connect_args,
        future=True,
        pool_pre_ping=not _is_sqlite_url(database_url),
    )

    if _is_sqlite_url(database_url):

        @event.listens_for(eng, "connect")
        def _set_sqlite_pragmas(dbapi_connection: Any, _connection_record: Any) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            # WAL improves concurrent read/write throughput for file-backed SQLite DBs
            if ":memory:" not in database_url:
                cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()

    return eng


engine: Engine = build_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False, class_=Session)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a database session and ensuring cleanup."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
