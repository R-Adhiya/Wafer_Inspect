"""Shared pytest fixtures for deterministic database engine disposal and resource cleanup."""

from pathlib import Path
from typing import Generator, Tuple
from fastapi.testclient import TestClient
import pytest
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings
from app.db.database import Base, build_engine, engine as default_engine, get_db
from app.db.migrations import upgrade_database_to_head
from app.main import app


@pytest.fixture(scope="session", autouse=True)
def dispose_default_engine_after_session() -> Generator[None, None, None]:
    """Ensure the default module SQLAlchemy engine is disposed cleanly at pytest shutdown."""
    yield
    default_engine.dispose()


@pytest.fixture
def memory_db_session() -> Generator[Session, None, None]:
    """Yield an isolated in-memory SQLite session and dispose its engine on teardown."""
    eng = build_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=eng)
    SessionFactory = sessionmaker(bind=eng, autoflush=False, autocommit=False, expire_on_commit=False)
    db = SessionFactory()
    try:
        yield db
    finally:
        db.close()
        eng.dispose()


@pytest.fixture
def isolated_api_client(
    tmp_path: Path,
) -> Generator[Tuple[TestClient, Engine, sessionmaker, Path, str], None, None]:
    """Yield a TestClient wired to a fresh Alembic-migrated SQLite DB with full teardown cleanup."""
    db_file = tmp_path / "test_waferinspect.db"
    db_url = f"sqlite:///{db_file.as_posix()}"
    upgrade_database_to_head(db_url)
    eng = build_engine(db_url)
    SessionFactory = sessionmaker(bind=eng, autoflush=False, autocommit=False, expire_on_commit=False)

    def _override_get_db() -> Generator[Session, None, None]:
        session = SessionFactory()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = _override_get_db
    client = TestClient(app)
    try:
        yield client, eng, SessionFactory, db_file, db_url
    finally:
        client.close()
        app.dependency_overrides.clear()
        eng.dispose()
