"""FastAPI application entrypoint for WaferInspect."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.logging import setup_logging, logger

# Initialize logging configuration at startup
setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context manager handling startup and shutdown events."""
    logger.info("WaferInspect backend started successfully")
    yield


app = FastAPI(
    title="WaferInspect Backend",
    description="Automated Wafer Defect Inspection & Measurement API",
    lifespan=lifespan,
)

# Configure CORS restricted strictly to origins in CORS_ORIGINS (default http://localhost:5173), not "*"
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def get_health() -> Dict[str, str]:
    """Return operational status and service identifier."""
    return {"status": "ok", "service": "waferinspect-backend"}
