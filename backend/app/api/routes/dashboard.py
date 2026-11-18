"""API route handler for aggregated dashboard statistics."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.schemas import DashboardStatsResponse
from app.services.inspection_service import get_dashboard_statistics

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


@router.get("/stats", response_model=DashboardStatsResponse)
def get_dashboard_stats(db: Session = Depends(get_db)) -> DashboardStatsResponse:
    """Return SQL-aggregated dashboard KPIs, recent inspections, and chart series."""
    return get_dashboard_statistics(db=db)
