"""FastAPI application entrypoint for WaferInspect."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import dashboard, experiments, inspection, samples
from app.core.config import settings
from app.core.logging import setup_logging, logger
from app.db.database import engine
from app.db.migrations import upgrade_database_to_head

# Initialize logging configuration at startup
setup_logging()

_MIGRATED_URLS: set[str] = set()


def ensure_startup_migrations(database_url: str) -> None:
    """Run Alembic upgrade head once per process for a given database URL."""
    if database_url in _MIGRATED_URLS:
        return
    upgrade_database_to_head(database_url)
    _MIGRATED_URLS.add(database_url)


# Ensure schema is upgraded through Alembic once on first module load
try:
    ensure_startup_migrations(settings.DATABASE_URL)
except Exception as exc:
    logger.error("Failed to apply Alembic migrations at import time: %s", str(exc), exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context manager handling Alembic schema upgrades and engine disposal."""
    ensure_startup_migrations(settings.DATABASE_URL)
    logger.info("WaferInspect backend started successfully (schema up to date)")
    try:
        yield
    finally:
        engine.dispose()


app = FastAPI(
    title="WaferInspect Backend",
    description="Automated Wafer Defect Inspection & Measurement API",
    lifespan=lifespan,
)

# Configure CORS restricted strictly to origins in CORS_ORIGINS, not "*"
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Ensure no unhandled stack traces are leaked to clients in responses."""
    logger.error("Unhandled exception processing request '%s': %s", request.url.path, str(exc), exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": {"code": "INTERNAL_ERROR", "message": "An unexpected internal error occurred."}},
    )


@app.get("/health")
def get_health() -> Dict[str, str]:
    """Return operational status and service identifier."""
    return {"status": "ok", "service": "waferinspect-backend"}


# Mount API routers
app.include_router(inspection.router)
app.include_router(dashboard.router)
app.include_router(samples.router)
app.include_router(experiments.router)
