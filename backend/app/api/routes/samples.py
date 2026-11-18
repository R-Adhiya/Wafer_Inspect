"""API route handlers for synthetic demonstration wafer samples."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.schemas import SampleListResponse, UploadInspectionResponse
from app.services.sample_service import create_sample_upload, list_samples

router = APIRouter(prefix="/samples", tags=["Samples"])


@router.get("", response_model=SampleListResponse)
def get_samples() -> SampleListResponse:
    """List available synthetic demonstration wafer samples."""
    return list_samples()


@router.post("/{name}", response_model=UploadInspectionResponse)
async def create_sample(
    name: str,
    db: Session = Depends(get_db),
) -> UploadInspectionResponse:
    """Generate a deterministic synthetic demonstration wafer and upload it through the standard pipeline."""
    return await create_sample_upload(name=name, db=db)
