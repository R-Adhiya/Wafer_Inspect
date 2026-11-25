"""Service handling image storage, retrieval, and transformations."""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import HTTPException, UploadFile
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import logger
from app.db import crud
from app.db.database import SessionLocal
from app.db.schemas import ImageMetaResponse, UploadInspectionResponse
from app.utils.validators import ValidatedImage, validate_uploaded_image

ALLOWED_KINDS = {"test", "reference"}
MIME_TYPE_MAP = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}


async def process_and_store_upload(
    test_image_file: UploadFile,
    reference_image_file: Optional[UploadFile] = None,
    db: Optional[Session] = None,
    disclaimer: Optional[str] = None,
    ground_truth: Optional[List[Dict[str, Any]]] = None,
) -> UploadInspectionResponse:
    """Validate uploaded images and persist them safely under data/uploads/<upload_id>/."""
    # 1. Validate test image (required)
    validated_test: ValidatedImage = await validate_uploaded_image(test_image_file, field_name="test_image")

    # 2. Validate reference image (optional)
    validated_ref: Optional[ValidatedImage] = None
    if reference_image_file is not None and reference_image_file.filename:
        validated_ref = await validate_uploaded_image(reference_image_file, field_name="reference_image")

    # 3. Generate UUID4 upload_id and create destination folder
    upload_id = str(uuid.uuid4())
    upload_dir = settings.upload_path / upload_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    # 4. Save test image as test.<ext>
    test_filename = f"test{validated_test.extension}"
    test_path = upload_dir / test_filename
    with open(test_path, "wb") as f:
        f.write(validated_test.content_bytes)

    # 5. Save reference image as reference.<ext> if present
    ref_filename = None
    if validated_ref is not None:
        ref_filename = f"reference{validated_ref.extension}"
        ref_path = upload_dir / ref_filename
        with open(ref_path, "wb") as f:
            f.write(validated_ref.content_bytes)

    # 6. Check for reference dimension / aspect ratio difference
    reference_warning: Optional[str] = None
    if validated_ref is not None:
        aspect_test = validated_test.width / validated_test.height
        aspect_ref = validated_ref.width / validated_ref.height
        if (
            validated_test.width != validated_ref.width
            or validated_test.height != validated_ref.height
            or abs(aspect_test - aspect_ref) > 1e-4
        ):
            reference_warning = (
                f"Reference image dimensions ({validated_ref.width}x{validated_ref.height}) or aspect ratio "
                f"differ from test image ({validated_test.width}x{validated_test.height}). "
                f"The pipeline will align images in later stages."
            )

    # 7. Write meta.json
    metadata = {
        "upload_id": upload_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "test_image": {
            "original_filename": validated_test.sanitized_filename,
            "detected_format": validated_test.detected_format,
            "mode": validated_test.mode,
            "width": validated_test.width,
            "height": validated_test.height,
            "file_size_bytes": validated_test.file_size_bytes,
            "stored_filename": test_filename,
        },
        "reference_image": (
            {
                "original_filename": validated_ref.sanitized_filename,
                "detected_format": validated_ref.detected_format,
                "mode": validated_ref.mode,
                "width": validated_ref.width,
                "height": validated_ref.height,
                "file_size_bytes": validated_ref.file_size_bytes,
                "stored_filename": ref_filename,
            }
            if validated_ref is not None
            else None
        ),
        "reference_warning": reference_warning,
        "is_synthetic": (disclaimer is not None) or (ground_truth is not None),
        "disclaimer": disclaimer,
        "ground_truth": ground_truth,
    }
    meta_path = upload_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Create pending inspections row in database
    owns_session = db is None
    session = db if db is not None else SessionLocal()
    try:
        inspection_row = crud.create_inspection(
            session,
            upload_id=upload_id,
            original_filename=validated_test.sanitized_filename,
            image_width=validated_test.width,
            image_height=validated_test.height,
            image_mode=validated_test.mode,
            image_format=validated_test.detected_format,
            file_size=validated_test.file_size_bytes,
            has_reference=validated_ref is not None,
            reference_filename=validated_ref.sanitized_filename if validated_ref is not None else None,
            status="pending",
        )
    except SQLAlchemyError as exc:
        logger.error(
            "Database failure creating pending inspection for upload_id=%s: %s", upload_id, str(exc), exc_info=True
        )
        raise HTTPException(
            status_code=503,
            detail={
                "code": "DATABASE_ERROR",
                "message": "Database error while creating inspection record. Please retry.",
            },
        )
    finally:
        if owns_session:
            session.close()

    logger.info(
        "Inspection upload stored successfully: inspection_id=%s, upload_id=%s",
        inspection_row.id,
        upload_id,
    )

    return UploadInspectionResponse(
        inspection_id=inspection_row.id,
        upload_id=upload_id,
        test_image=ImageMetaResponse(
            original_filename=validated_test.sanitized_filename,
            detected_format=validated_test.detected_format,
            mode=validated_test.mode,
            width=validated_test.width,
            height=validated_test.height,
            file_size_bytes=validated_test.file_size_bytes,
        ),
        reference_image=(
            ImageMetaResponse(
                original_filename=validated_ref.sanitized_filename,
                detected_format=validated_ref.detected_format,
                mode=validated_ref.mode,
                width=validated_ref.width,
                height=validated_ref.height,
                file_size_bytes=validated_ref.file_size_bytes,
            )
            if validated_ref is not None
            else None
        ),
        reference_warning=reference_warning,
        disclaimer=disclaimer,
    )


def get_image_file_path(upload_id: str, kind: str) -> Tuple[Path, str]:
    """
    Validate upload_id as UUID and kind against allowlist, then return image path and media type.
    Prevents path traversal attacks.
    """
    if kind not in ALLOWED_KINDS:
        logger.warning("Invalid kind requested: %s", kind)
        raise HTTPException(
            status_code=400,
            detail={
                "code": "INVALID_KIND",
                "message": f"Image kind '{kind}' is invalid. Allowed: 'test', 'reference'.",
            },
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

    upload_dir = settings.upload_path / clean_uuid
    if not upload_dir.is_dir():
        logger.warning("Upload directory not found for upload_id: %s", clean_uuid)
        raise HTTPException(
            status_code=404,
            detail={"code": "UPLOAD_NOT_FOUND", "message": f"Upload '{clean_uuid}' was not found."},
        )

    # Search for target file test.* or reference.*
    target_files = list(upload_dir.glob(f"{kind}.*"))
    target_files = [f for f in target_files if f.suffix.lower() in MIME_TYPE_MAP]

    if not target_files or not target_files[0].is_file():
        logger.warning("File for kind '%s' not found in upload '%s'", kind, clean_uuid)
        raise HTTPException(
            status_code=404,
            detail={
                "code": "FILE_NOT_FOUND",
                "message": f"{kind.capitalize()} image not found for upload {clean_uuid}.",
            },
        )

    file_path = target_files[0]
    media_type = MIME_TYPE_MAP.get(file_path.suffix.lower(), "application/octet-stream")
    return file_path, media_type
