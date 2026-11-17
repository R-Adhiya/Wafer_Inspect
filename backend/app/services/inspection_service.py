"""Service orchestrating wafer defect inspection workflows and preprocessing."""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from fastapi import HTTPException
import numpy as np
from PIL import Image

from app.core.config import settings
from app.core.logging import logger
from app.cv.pipeline import run_preprocessing
from app.cv.preprocessing import PreprocessingConfig
from app.db.schemas import ImageStats, InspectionRunResponse, StepLogItem

ALLOWED_RUN_IMAGE_KINDS = {"original", "preprocessed"}
MIME_TYPE_MAP = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}


def run_inspection_pipeline(
    upload_id: str,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> InspectionRunResponse:
    """Load uploaded test image, execute CV preprocessing pipeline, and store run results.

    Args:
        upload_id: UUID4 identifier of an existing uploaded wafer inspection session.
        config_overrides: Optional dictionary of parameters to override PreprocessingConfig.

    Returns:
        InspectionRunResponse: Execution summary, step log, stats, and image endpoints.
    """
    # 1. Validate upload_id format
    try:
        uuid_obj = uuid.UUID(upload_id)
        clean_uuid = str(uuid_obj)
    except (ValueError, TypeError):
        logger.warning("Invalid UUID supplied for inspection run: %s", upload_id)
        raise HTTPException(
            status_code=404,
            detail={"code": "INVALID_UPLOAD_ID", "message": "Upload ID must be a valid UUID."},
        )

    # 2. Locate uploaded test image
    upload_dir = settings.upload_path / clean_uuid
    if not upload_dir.is_dir():
        logger.warning("Upload session directory not found: %s", clean_uuid)
        raise HTTPException(
            status_code=404,
            detail={"code": "UPLOAD_NOT_FOUND", "message": f"Upload session '{clean_uuid}' was not found."},
        )

    target_files = list(upload_dir.glob("test.*"))
    target_files = [f for f in target_files if f.suffix.lower() in MIME_TYPE_MAP]

    if not target_files or not target_files[0].is_file():
        logger.warning("Test image file not found for upload '%s'", clean_uuid)
        raise HTTPException(
            status_code=404,
            detail={"code": "TEST_IMAGE_NOT_FOUND", "message": f"Test image not found for upload '{clean_uuid}'."},
        )

    test_image_path = target_files[0]

    # 3. Load image into NumPy array
    try:
        with Image.open(test_image_path) as pil_img:
            img_array = np.array(pil_img)
            orig_w, orig_h = pil_img.size
    except Exception as exc:
        logger.error("Failed to read image at '%s': %s", test_image_path, str(exc), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"code": "IMAGE_LOAD_ERROR", "message": "Failed to decode stored wafer image."},
        )

    # Compute original statistics
    orig_mean = float(np.mean(img_array))
    orig_std = float(np.std(img_array))
    original_stats = ImageStats(
        width=orig_w,
        height=orig_h,
        mean=round(orig_mean, 2),
        std=round(orig_std, 2),
    )

    # 4. Parse configuration
    try:
        if config_overrides:
            config = PreprocessingConfig(**config_overrides)
        else:
            config = PreprocessingConfig()
    except Exception as exc:
        logger.warning("Invalid preprocessing config overrides for upload '%s': %s", clean_uuid, str(exc))
        raise HTTPException(
            status_code=422,
            detail={"code": "INVALID_CONFIG", "message": f"Invalid preprocessing configuration: {str(exc)}"},
        )

    # 5. Execute preprocessing pipeline
    logger.info("Executing preprocessing pipeline for upload_id=%s", clean_uuid)
    preprocessed_img, step_log = run_preprocessing(img_array, config)

    # Compute preprocessed statistics
    prep_h, prep_w = preprocessed_img.shape[:2]
    prep_mean = float(np.mean(preprocessed_img))
    prep_std = float(np.std(preprocessed_img))
    preprocessed_stats = ImageStats(
        width=prep_w,
        height=prep_h,
        mean=round(prep_mean, 2),
        std=round(prep_std, 2),
    )

    # 6. Save results to data/results/<upload_id>/
    results_dir = settings.results_path / clean_uuid
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save preprocessed image as 8-bit PNG
    prep_out_path = results_dir / "preprocessed.png"
    Image.fromarray(preprocessed_img).save(prep_out_path, format="PNG")

    # Save run.json (full config plus step log for complete reproducibility)
    run_meta = {
        "upload_id": clean_uuid,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config.model_dump(),
        "original_stats": original_stats.model_dump(),
        "preprocessed_stats": preprocessed_stats.model_dump(),
        "step_log": step_log,
    }
    with open(results_dir / "run.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    logger.info("Preprocessing completed and stored for upload_id=%s", clean_uuid)

    step_log_items = [StepLogItem(**step) for step in step_log]

    return InspectionRunResponse(
        upload_id=clean_uuid,
        step_log=step_log_items,
        original_stats=original_stats,
        preprocessed_stats=preprocessed_stats,
        original_image_url=f"/inspection/run/{clean_uuid}/image/original",
        preprocessed_image_url=f"/inspection/run/{clean_uuid}/image/preprocessed",
    )


def get_run_image_file_path(upload_id: str, kind: str) -> Tuple[Path, str]:
    """Retrieve file path and media type for original or preprocessed run images.

    Args:
        upload_id: UUID4 identifier of the inspection session.
        kind: "original" or "preprocessed".

    Returns:
        Tuple[Path, str]: (absolute_file_path, mime_type)
    """
    clean_kind = kind.strip().lower()
    if clean_kind not in ALLOWED_RUN_IMAGE_KINDS:
        logger.warning("Invalid run image kind requested: %s", kind)
        raise HTTPException(
            status_code=400,
            detail={"code": "INVALID_KIND", "message": f"Image kind '{kind}' is invalid. Allowed: 'original', 'preprocessed'."},
        )

    try:
        uuid_obj = uuid.UUID(upload_id)
        clean_uuid = str(uuid_obj)
    except (ValueError, TypeError):
        logger.warning("Invalid UUID supplied: %s", upload_id)
        raise HTTPException(
            status_code=404,
            detail={"code": "INVALID_UPLOAD_ID", "message": "Upload ID must be a valid UUID."},
        )

    if clean_kind == "original":
        upload_dir = settings.upload_path / clean_uuid
        if not upload_dir.is_dir():
            raise HTTPException(
                status_code=404,
                detail={"code": "UPLOAD_NOT_FOUND", "message": f"Upload '{clean_uuid}' was not found."},
            )
        target_files = list(upload_dir.glob("test.*"))
        target_files = [f for f in target_files if f.suffix.lower() in MIME_TYPE_MAP]
        if not target_files or not target_files[0].is_file():
            raise HTTPException(
                status_code=404,
                detail={"code": "IMAGE_NOT_FOUND", "message": f"Original image not found for upload {clean_uuid}."},
            )
        file_path = target_files[0]
        media_type = MIME_TYPE_MAP.get(file_path.suffix.lower(), "image/png")
        return file_path, media_type

    # clean_kind == "preprocessed"
    results_dir = settings.results_path / clean_uuid
    prep_path = results_dir / "preprocessed.png"
    if not prep_path.is_file():
        raise HTTPException(
            status_code=404,
            detail={"code": "PREPROCESSED_IMAGE_NOT_FOUND", "message": f"Preprocessed image not found for upload {clean_uuid}."},
        )
    return prep_path, "image/png"
