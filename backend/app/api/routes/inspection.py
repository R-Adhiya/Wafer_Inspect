"""API route handlers for inspection workflows and image upload."""

from typing import Optional
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import FileResponse
from app.db.schemas import UploadInspectionResponse
from app.services.image_service import get_image_file_path, process_and_store_upload

router = APIRouter(prefix="/inspection", tags=["Inspection"])


@router.post("/upload", response_model=UploadInspectionResponse)
async def upload_inspection_images(
    test_image: UploadFile = File(..., description="Required test wafer image"),
    reference_image: Optional[UploadFile] = File(None, description="Optional golden reference wafer image"),
) -> UploadInspectionResponse:
    """Upload and validate wafer inspection images."""
    return await process_and_store_upload(
        test_image_file=test_image,
        reference_image_file=reference_image,
    )


@router.get("/upload/{upload_id}/{kind}")
def get_uploaded_image(
    upload_id: str,
    kind: str,
) -> FileResponse:
    """Serve a previously uploaded test or reference image."""
    file_path, media_type = get_image_file_path(upload_id=upload_id, kind=kind)
    return FileResponse(
        path=str(file_path),
        media_type=media_type,
    )
