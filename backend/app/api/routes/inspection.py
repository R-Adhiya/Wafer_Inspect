"""API route handlers for inspection workflows, database history, reports, and artifacts.

Route ordering note: `/inspection/history` is intentionally registered BEFORE
`/inspection/{inspection_id}` so path matching never tries to parse `"history"` as an integer ID.
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, File, Query, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.schemas import (
    AcceptedRegionResponse,
    DeleteInspectionResponse,
    HistoryListResponse,
    InspectionImagesResponse,
    InspectionNoiseExperimentResponse,
    InspectionRunRequest,
    InspectionRunResponse,
    InspectionStatusResponse,
    UploadInspectionResponse,
)
from app.services.image_service import get_image_file_path, process_and_store_upload
from app.services.inspection_service import (
    delete_inspection_by_id,
    get_inspection_defects_by_id,
    get_inspection_image_file_by_id,
    get_inspection_images_by_id,
    get_inspection_noise_experiment_by_id,
    get_inspection_report_by_id,
    get_inspection_result,
    get_inspection_status,
    get_inspection_status_by_id,
    get_run_image_file_path,
    list_inspection_history,
    run_inspection_pipeline,
)

router = APIRouter(prefix="/inspection", tags=["Inspection"])


@router.post("/upload", response_model=UploadInspectionResponse)
async def upload_inspection_images(
    test_image: UploadFile = File(..., description="Required test wafer image"),
    reference_image: Optional[UploadFile] = File(None, description="Optional golden reference wafer image"),
    db: Session = Depends(get_db),
) -> UploadInspectionResponse:
    """Upload, validate, store images, and create a pending database inspection record."""
    return await process_and_store_upload(
        test_image_file=test_image,
        reference_image_file=reference_image,
        db=db,
    )


@router.get("/upload/{upload_id}/{kind}")
def get_uploaded_image(
    upload_id: str,
    kind: str,
) -> FileResponse:
    """Serve a previously uploaded test or reference image."""
    file_path, media_type = get_image_file_path(upload_id=upload_id, kind=kind)
    return FileResponse(path=str(file_path), media_type=media_type)


@router.post("/run", response_model=InspectionRunResponse)
def run_inspection(
    payload: InspectionRunRequest,
    db: Session = Depends(get_db),
) -> InspectionRunResponse:
    """Execute full inspection pipeline synchronously and persist results atomically."""
    return run_inspection_pipeline(
        upload_id=payload.upload_id,
        inspection_id=payload.inspection_id,
        config_overrides=payload.config,
        preprocessing_overrides=payload.preprocessing_config,
        detection_overrides=payload.detection_config,
        noise_filter_overrides=payload.noise_filter_config,
        measurement_overrides=payload.measurement_config,
        severity_overrides=payload.severity_config,
        pattern_overrides=payload.pattern_config,
        db=db,
    )


# Defined strictly BEFORE /inspection/{inspection_id}
@router.get("/history", response_model=HistoryListResponse)
def get_inspection_history(
    search: Optional[str] = Query(default=None, description="Case-insensitive filename substring search"),
    severity: Optional[str] = Query(default=None, description="Filter by severity (NONE, LOW, MEDIUM, HIGH)"),
    sort: str = Query(
        default="created_at", description="Sort key: created_at, defect_count, severity, processing_time"
    ),
    order: str = Query(default="desc", description="Sort order: asc or desc"),
    page: int = Query(default=1, ge=1, description="1-based page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page"),
    db: Session = Depends(get_db),
) -> HistoryListResponse:
    """List paginated, searchable, filterable, and sortable inspection history."""
    return list_inspection_history(
        db,
        search=search,
        severity=severity,
        sort=sort,
        order=order,
        page=page,
        page_size=page_size,
    )


# Legacy upload_id run routes retained for backward compatibility with Phases 3-7 API consumers (deprecated in OpenAPI)
@router.get("/run/{upload_id}/image/{kind}", deprecated=True)
def get_run_image(upload_id: str, kind: str) -> FileResponse:
    """Serve an image artifact from an inspection run by UUID upload_id. Deprecated: use /inspection/{id}/image/{kind}."""
    file_path, media_type = get_run_image_file_path(upload_id=upload_id, kind=kind)
    return FileResponse(path=str(file_path), media_type=media_type)


@router.get("/run/{upload_id}/result", response_model=InspectionRunResponse, deprecated=True)
def get_run_result(upload_id: str, db: Session = Depends(get_db)) -> InspectionRunResponse:
    """Return completed inspection result by UUID upload_id (404 unknown, 409 unfinished). Deprecated: use /inspection/{id}."""
    return get_inspection_result(upload_id=upload_id, db=db)


@router.get("/run/{upload_id}/status", response_model=InspectionStatusResponse, deprecated=True)
def get_run_status(upload_id: str) -> InspectionStatusResponse:
    """Return live single-process in-memory stage status by UUID upload_id. Deprecated: use /inspection/{id}/status."""
    return get_inspection_status(upload_id=upload_id)


# Phase 8 integer inspection_id REST endpoints (built from database rows)
@router.get("/{inspection_id}", response_model=InspectionRunResponse)
def get_inspection_by_id(
    inspection_id: int,
    db: Session = Depends(get_db),
) -> InspectionRunResponse:
    """Return the full inspection report reconstructed strictly from database rows."""
    return get_inspection_report_by_id(db=db, inspection_id=inspection_id)


@router.get("/{inspection_id}/defects", response_model=List[AcceptedRegionResponse])
def get_inspection_defects(
    inspection_id: int,
    db: Session = Depends(get_db),
) -> List[AcceptedRegionResponse]:
    """Return all defects persisted for an inspection."""
    return get_inspection_defects_by_id(db=db, inspection_id=inspection_id)


@router.get("/{inspection_id}/images", response_model=InspectionImagesResponse)
def get_inspection_images(
    inspection_id: int,
    db: Session = Depends(get_db),
) -> InspectionImagesResponse:
    """Return map of available image artifact kinds to URLs for an inspection."""
    return get_inspection_images_by_id(db=db, inspection_id=inspection_id)


@router.get("/{inspection_id}/image/{kind}")
def get_inspection_image(
    inspection_id: int,
    kind: str,
    db: Session = Depends(get_db),
) -> FileResponse:
    """Serve an image artifact by integer inspection_id with allowlist and UUID safety."""
    file_path, media_type = get_inspection_image_file_by_id(db=db, inspection_id=inspection_id, kind=kind)
    return FileResponse(path=str(file_path), media_type=media_type)


@router.get("/{inspection_id}/status", response_model=InspectionStatusResponse)
def get_inspection_status_route(
    inspection_id: int,
    db: Session = Depends(get_db),
) -> InspectionStatusResponse:
    """Return live single-process stage status for an inspection by integer ID."""
    return get_inspection_status_by_id(db=db, inspection_id=inspection_id)


@router.get("/{inspection_id}/noise-experiment", response_model=InspectionNoiseExperimentResponse)
def get_inspection_noise_experiment(
    inspection_id: int,
    db: Session = Depends(get_db),
) -> InspectionNoiseExperimentResponse:
    """Return four-stage noise experiment candidate metrics for a completed inspection."""
    return get_inspection_noise_experiment_by_id(db=db, inspection_id=inspection_id)


@router.delete("/{inspection_id}", response_model=DeleteInspectionResponse)
def delete_inspection_route(
    inspection_id: int,
    db: Session = Depends(get_db),
) -> DeleteInspectionResponse:
    """Delete inspection rows (cascading defects & runs) and remove upload/result files."""
    return delete_inspection_by_id(db=db, inspection_id=inspection_id)
