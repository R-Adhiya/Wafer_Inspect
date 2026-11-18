"""API route handlers for inspection workflows, image upload, and pipeline execution."""

from typing import Optional
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import FileResponse

from app.db.schemas import (
    InspectionRunRequest,
    InspectionRunResponse,
    UploadInspectionResponse,
)
from app.services.image_service import get_image_file_path, process_and_store_upload
from app.services.inspection_service import (
    get_run_image_file_path,
    run_inspection_pipeline,
)

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


@router.post("/run", response_model=InspectionRunResponse)
def run_inspection(
    payload: InspectionRunRequest,
) -> InspectionRunResponse:
    """Execute preprocessing, detection, segmentation, filtering, measurement, and severity pipeline."""
    return run_inspection_pipeline(
        upload_id=payload.upload_id,
        config_overrides=payload.config,
        preprocessing_overrides=payload.preprocessing_config,
        detection_overrides=payload.detection_config,
        noise_filter_overrides=payload.noise_filter_config,
        measurement_overrides=payload.measurement_config,
        severity_overrides=payload.severity_config,
    )


@router.get("/run/{upload_id}/image/{kind}")
def get_run_image(
    upload_id: str,
    kind: str,
) -> FileResponse:
    """Serve either original or preprocessed image from an inspection run."""
    file_path, media_type = get_run_image_file_path(upload_id=upload_id, kind=kind)
    return FileResponse(
        path=str(file_path),
        media_type=media_type,
    )
