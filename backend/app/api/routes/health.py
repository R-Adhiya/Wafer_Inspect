from fastapi import APIRouter
from app.core.config import settings
from app.core.logging import logger
from app.db.schemas import HealthResponse

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Returns the operational status and version of the WaferInspect API."""
    logger.info("Health check endpoint accessed")
    return HealthResponse(
        status="healthy",
        service=settings.APP_NAME,
        version=settings.APP_VERSION
    )
