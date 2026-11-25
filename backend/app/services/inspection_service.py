"""Service orchestrating wafer defect inspection workflows, persistence, history, and stats.

All CV computation happens inside `app.cv.*` (zero `cv2` imports here).
Database persistence uses SQLAlchemy 2.0 via `app.db.crud`; `run.json` is written
solely as a reproducibility artifact while `GET /inspection/{id}` rebuilds reports
strictly from database rows.
"""

import json
import math
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from fastapi import HTTPException
import numpy as np
from PIL import Image
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import logger
from app.cv.measurement import MeasurementConfig
from app.cv.noise_filter import NoiseFilterConfig
from app.cv.pattern_analysis import PatternConfig
from app.cv.pipeline import (
    DetectionConfig,
    PreprocessingConfig,
    ReferenceIncompatibleError,
    run_full_pipeline,
)
from app.cv.severity import SeverityConfig
from app.db import crud
from app.db.database import SessionLocal
from app.db.models import Defect, Inspection
from app.db.schemas import (
    AcceptedRegionResponse,
    AreaHistogramPoint,
    DashboardStatsResponse,
    DeleteInspectionResponse,
    DetectionSummaryResponse,
    HistoryItemResponse,
    HistoryListResponse,
    ImageStats,
    InspectionImagesResponse,
    InspectionNoiseExperimentResponse,
    InspectionRunResponse,
    InspectionStatusResponse,
    InspectionSummaryResponse,
    ModelInfoResponse,
    PatternCountPoint,
    PatternResultResponse,
    RecentDefectCountPoint,
    RejectedRegionResponse,
    SeverityCountPoint,
    StageCount,
    StepLogItem,
)
from app.services.status_store import PIPELINE_STAGES, status_store

ALLOWED_RUN_IMAGE_KINDS = {
    "original",
    "preprocessed",
    "difference",
    "mask",
    "filtered_mask",
    "bboxes",
    "contours",
    "final_overlay",
    "pattern_overlay",
}

RESULT_PNG_MAP = {
    "preprocessed": "preprocessed.png",
    "difference": "difference.png",
    "mask": "mask.png",
    "filtered_mask": "filtered_mask.png",
    "bboxes": "bboxes.png",
    "contours": "contours.png",
    "final_overlay": "final_overlay.png",
    "pattern_overlay": "pattern_overlay.png",
}

MIME_TYPE_MAP = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}


def _db_error_http(context: str, exc: Exception) -> HTTPException:
    logger.error("Database error (%s): %s", context, str(exc), exc_info=True)
    return HTTPException(
        status_code=503,
        detail={
            "code": "DATABASE_ERROR",
            "message": "Database temporarily unavailable. Please retry.",
        },
    )


def _safe_mark_failed(db: Session, inspection_id: Optional[int], sanitized_msg: str) -> None:
    if inspection_id is None:
        return
    try:
        crud.set_status(db, inspection_id, "failed", error_message=sanitized_msg)
    except Exception as exc:
        logger.error("Failed to persist failed status for inspection_id=%s: %s", inspection_id, str(exc))


def _defect_row_to_schema(d: Defect) -> AcceptedRegionResponse:
    contour_pts: List[Tuple[int, int]] = []
    if d.contour:
        contour_pts = [(int(pt[0]), int(pt[1])) for pt in d.contour if len(pt) >= 2]
    bbox = (int(d.bbox_x), int(d.bbox_y), int(d.bbox_width), int(d.bbox_height))
    centroid = (float(d.centroid_x), float(d.centroid_y))
    return AcceptedRegionResponse(
        id=int(d.region_no),
        bbox=bbox,
        width_px=int(d.width),
        height_px=int(d.height),
        area_px=int(d.area),
        detected_area_px=int(d.detected_area),
        centroid=centroid,
        perimeter_px=float(d.perimeter),
        equivalent_diameter_px=float(d.equivalent_diameter),
        length_px=float(d.length),
        thickness_px=float(d.thickness),
        angle_deg=float(d.angle_deg),
        elongation=float(d.elongation),
        polarity=d.polarity,  # type: ignore[arg-type]
        mean_contrast=float(d.mean_contrast),
        peak_contrast=float(d.peak_contrast),
        contrast_to_noise=float(d.contrast_to_noise),
        touches_edge=bool(d.touches_edge),
        detected_by=d.detected_by,  # type: ignore[arg-type]
        why_flagged=d.why_flagged,
        severity=d.severity,  # type: ignore[arg-type]
        area_um2=d.area_um2,
        length_um=d.length_um,
        thickness_um=d.thickness_um,
        contour=contour_pts,
        reason=d.reason or "",
        original_bbox=bbox,
        original_centroid=centroid,
    )


def _history_item_from_row(row: Inspection) -> HistoryItemResponse:
    return HistoryItemResponse(
        id=int(row.id),
        upload_id=row.upload_id,
        status=row.status,
        original_filename=row.original_filename,
        image_width=int(row.image_width),
        image_height=int(row.image_height),
        has_reference=bool(row.has_reference),
        defect_count=int(row.defect_count),
        total_defect_area=int(row.total_defect_area),
        largest_defect_area=int(row.largest_defect_area),
        severity=row.severity,
        pattern_type=row.pattern_type,
        pattern_confidence=row.pattern_confidence,
        detection_path=row.detection_path,
        detection_label=row.detection_label,
        detection_method=row.detection_method,
        model_name=row.model_name,
        model_version=row.model_version,
        processing_time=row.processing_time,
        created_at=row.created_at.isoformat() if row.created_at else "",
    )


def _build_report_from_db(row: Inspection) -> InspectionRunResponse:
    """Reconstruct the complete InspectionRunResponse strictly from SQLAlchemy DB rows."""
    clean_uuid = row.upload_id
    image_urls = {
        "original_image_url": f"/inspection/run/{clean_uuid}/image/original",
        "preprocessed_image_url": f"/inspection/run/{clean_uuid}/image/preprocessed",
        "difference_image_url": f"/inspection/run/{clean_uuid}/image/difference",
        "mask_image_url": f"/inspection/run/{clean_uuid}/image/mask",
        "filtered_mask_image_url": f"/inspection/run/{clean_uuid}/image/filtered_mask",
        "bboxes_image_url": f"/inspection/run/{clean_uuid}/image/bboxes",
        "contours_image_url": f"/inspection/run/{clean_uuid}/image/contours",
        "final_overlay_image_url": f"/inspection/run/{clean_uuid}/image/final_overlay",
        "pattern_overlay_image_url": f"/inspection/run/{clean_uuid}/image/pattern_overlay",
    }

    stage_counts = [StageCount(**sc) for sc in (row.stage_counts or [])]
    step_log = [StepLogItem(**st) for st in (row.step_log or [])]

    orig_stats = (
        ImageStats(**row.original_stats)
        if row.original_stats
        else ImageStats(width=row.image_width, height=row.image_height, mean=0.0, std=0.0)
    )
    prep_stats = (
        ImageStats(**row.preprocessed_stats)
        if row.preprocessed_stats
        else ImageStats(width=row.image_width, height=row.image_height, mean=0.0, std=0.0)
    )

    regions = [_defect_row_to_schema(d) for d in sorted(row.defects, key=lambda x: (x.region_no, x.id))]
    rejected_regions = [
        RejectedRegionResponse(
            id=int(rr["id"]),
            area_px=int(rr["area_px"]),
            mean_contrast=float(rr["mean_contrast"]),
            reason=str(rr["reason"]),
        )
        for rr in (row.rejected_regions or [])
    ]

    pattern_response: Optional[PatternResultResponse] = None
    if row.pattern_type:
        pattern_response = PatternResultResponse(
            type=row.pattern_type,  # type: ignore[arg-type]
            confidence=float(row.pattern_confidence or 0.0),
            p_value=row.pattern_p_value,
            explanation=row.pattern_explanation or "",
            metrics=row.pattern_metrics or {},
            supporting_defect_ids=list(row.pattern_supporting_ids or []),
            warnings=list(row.pattern_warnings or []),
        )

    summary_response: Optional[InspectionSummaryResponse] = None
    if row.severity is not None:
        summary_response = InspectionSummaryResponse(
            defect_count=int(row.defect_count),
            total_defect_area_px=int(row.total_defect_area),
            largest_defect_area_px=int(row.largest_defect_area),
            average_defect_area_px=float(row.average_defect_area),
            severity=row.severity,  # type: ignore[arg-type]
            severity_reason=row.severity_reason or "",
            calibration_status=row.calibration_status or "uncalibrated: measurements are in image pixels",
            pixel_size_um=row.pixel_size_um,
            pattern=pattern_response,
        )

    model_response: Optional[ModelInfoResponse] = None
    if row.model_name:
        model_response = ModelInfoResponse(
            name=row.model_name,
            version=row.model_version or "1.0.0",
            detection_method=row.detection_method or "Classical CV Baseline",
        )

    det_response: Optional[DetectionSummaryResponse] = None
    if row.detection_path:
        det_extra = row.detection_details or {}
        wc = det_extra.get("wafer_center", (0.0, 0.0))
        dp = det_extra.get("die_pitch")
        det_response = DetectionSummaryResponse(
            path=row.detection_path,
            label=row.detection_label or row.detection_path,
            threshold_method=str(det_extra.get("threshold_method", "otsu")),
            threshold_value=float(det_extra.get("threshold_value", 0.0)),
            threshold_reason=str(det_extra.get("threshold_reason", "")),
            wafer_center=(float(wc[0]), float(wc[1])),
            wafer_radius=float(det_extra.get("wafer_radius", 0.0)),
            is_full_frame=bool(det_extra.get("is_full_frame", False)),
            alignment_info=det_extra.get("alignment_info") or {},
            stage_counts=stage_counts,
            warnings=list(row.warnings or []),
            die_pitch=(int(dp[0]), int(dp[1])) if dp is not None else None,
            peak_strength=det_extra.get("peak_strength"),
            coverage=det_extra.get("coverage"),
            unverifiable_fraction=det_extra.get("unverifiable_fraction"),
        )

    latest_run = row.processing_runs[-1] if row.processing_runs else None
    stage_timings = dict(latest_run.stage_timings) if latest_run and latest_run.stage_timings else {}

    return InspectionRunResponse(
        inspection_id=int(row.id),
        upload_id=clean_uuid,
        status=row.status,
        created_at=row.created_at.isoformat() if row.created_at else None,
        step_log=step_log,
        original_stats=orig_stats,
        preprocessed_stats=prep_stats,
        detection=det_response,
        stage_counts=stage_counts,
        regions=regions,
        rejected_regions=rejected_regions,
        summary=summary_response,
        model=model_response,
        pattern=pattern_response,
        processing_time_s=float(row.processing_time or 0.0),
        stage_timings_ms=stage_timings,
        test_filename=row.original_filename,
        test_mode=row.image_mode,
        reference_used=bool(row.has_reference),
        reference_filename=row.reference_filename,
        **image_urls,
    )


def _validate_upload_uuid(upload_id: str) -> str:
    """Validate upload_id as a UUID string or raise HTTP 404."""
    try:
        return str(uuid.UUID(upload_id))
    except (ValueError, TypeError):
        logger.warning("Invalid UUID supplied: %s", upload_id)
        raise HTTPException(
            status_code=404,
            detail={"code": "INVALID_UPLOAD_ID", "message": "Upload ID must be a valid UUID."},
        )


def run_inspection_pipeline(
    upload_id: Optional[str] = None,
    inspection_id: Optional[int] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
    preprocessing_overrides: Optional[Dict[str, Any]] = None,
    detection_overrides: Optional[Dict[str, Any]] = None,
    noise_filter_overrides: Optional[Dict[str, Any]] = None,
    measurement_overrides: Optional[Dict[str, Any]] = None,
    severity_overrides: Optional[Dict[str, Any]] = None,
    pattern_overrides: Optional[Dict[str, Any]] = None,
    db: Optional[Session] = None,
) -> InspectionRunResponse:
    """Load uploaded wafer images, execute full CV pipeline, and persist results atomically."""
    owns_session = db is None
    session = db if db is not None else SessionLocal()

    inspection_row: Optional[Inspection] = None
    clean_uuid: Optional[str] = None

    try:
        # 1. Resolve inspection row & upload_id
        try:
            if inspection_id is not None:
                inspection_row = crud.get_inspection(session, int(inspection_id))
                if inspection_row is None:
                    raise HTTPException(
                        status_code=404,
                        detail={
                            "code": "INSPECTION_NOT_FOUND",
                            "message": f"Inspection '{inspection_id}' was not found.",
                        },
                    )
                clean_uuid = inspection_row.upload_id
            elif upload_id is not None:
                clean_uuid = _validate_upload_uuid(upload_id)
                inspection_row = crud.get_inspection_by_upload_id(session, clean_uuid)
            else:
                raise HTTPException(
                    status_code=422,
                    detail={"code": "MISSING_IDENTIFIER", "message": "Provide inspection_id or upload_id."},
                )
        except HTTPException:
            raise
        except SQLAlchemyError as exc:
            raise _db_error_http("resolve_inspection", exc)

        assert clean_uuid is not None

        # Mark status running in DB (if row exists) and in-memory live status store
        if inspection_row is not None:
            try:
                crud.set_status(session, inspection_row.id, "running")
            except SQLAlchemyError as exc:
                raise _db_error_http("set_running", exc)

        status_store.start_run(clean_uuid)

        # 2. Locate uploaded test image
        upload_dir = settings.upload_path / clean_uuid
        if not upload_dir.is_dir():
            sanitized = "Uploaded wafer session directory is missing on storage."
            logger.error("Upload directory missing for upload_id=%s at %s", clean_uuid, upload_dir)
            _safe_mark_failed(session, inspection_row.id if inspection_row else None, sanitized)
            status_store.fail_run(clean_uuid, sanitized)
            raise HTTPException(
                status_code=404,
                detail={"code": "UPLOAD_NOT_FOUND", "message": f"Upload session '{clean_uuid}' was not found."},
            )

        test_files = [f for f in upload_dir.glob("test.*") if f.suffix.lower() in MIME_TYPE_MAP]
        if not test_files or not test_files[0].is_file():
            sanitized = "Uploaded test image file is missing on storage."
            logger.error("Test image file missing for upload_id=%s in %s", clean_uuid, upload_dir)
            _safe_mark_failed(session, inspection_row.id if inspection_row else None, sanitized)
            status_store.fail_run(clean_uuid, sanitized)
            raise HTTPException(
                status_code=404,
                detail={"code": "TEST_IMAGE_NOT_FOUND", "message": sanitized},
            )

        test_image_path = test_files[0]

        # Read upload meta.json if present
        test_filename: str = inspection_row.original_filename if inspection_row else test_image_path.name
        test_mode: Optional[str] = inspection_row.image_mode if inspection_row else None
        test_format: str = inspection_row.image_format if inspection_row else "PNG"
        test_file_size: int = (
            int(inspection_row.file_size) if inspection_row else int(test_image_path.stat().st_size)
        )
        reference_filename: Optional[str] = inspection_row.reference_filename if inspection_row else None

        upload_meta_path = upload_dir / "meta.json"
        if upload_meta_path.is_file():
            try:
                u_meta = json.loads(upload_meta_path.read_text(encoding="utf-8"))
                if isinstance(u_meta.get("test_image"), dict):
                    test_filename = u_meta["test_image"].get("original_filename", test_filename)
                    test_mode = u_meta["test_image"].get("mode", test_mode)
                    test_format = u_meta["test_image"].get("detected_format", test_format)
                    test_file_size = int(u_meta["test_image"].get("file_size_bytes", test_file_size))
                if isinstance(u_meta.get("reference_image"), dict):
                    reference_filename = u_meta["reference_image"].get("original_filename", reference_filename)
            except Exception:
                pass

        # 3. Check for golden reference image
        ref_files = [f for f in upload_dir.glob("reference.*") if f.suffix.lower() in MIME_TYPE_MAP]
        ref_image_path = ref_files[0] if (ref_files and ref_files[0].is_file()) else None
        if ref_image_path is not None and not reference_filename:
            reference_filename = ref_image_path.name

        try:
            # 4. Load images into NumPy arrays
            try:
                with Image.open(test_image_path) as pil_test:
                    test_array = np.array(pil_test)
                    orig_w, orig_h = pil_test.size
                    if not test_mode:
                        test_mode = pil_test.mode
            except Exception as exc:
                logger.error("Failed to read test image '%s': %s", test_image_path, str(exc), exc_info=True)
                raise HTTPException(
                    status_code=500,
                    detail={"code": "IMAGE_LOAD_ERROR", "message": "Failed to decode stored wafer image."},
                )

            # Backfill DB row if upload existed on disk prior to DB creation
            if inspection_row is None:
                try:
                    inspection_row = crud.create_inspection(
                        session,
                        upload_id=clean_uuid,
                        original_filename=test_filename,
                        image_width=orig_w,
                        image_height=orig_h,
                        image_mode=test_mode or "L",
                        image_format=test_format,
                        file_size=test_file_size,
                        has_reference=ref_image_path is not None,
                        reference_filename=reference_filename,
                        status="running",
                    )
                except SQLAlchemyError as exc:
                    raise _db_error_http("backfill_inspection", exc)

            ref_array: Optional[np.ndarray] = None
            if ref_image_path is not None:
                try:
                    with Image.open(ref_image_path) as pil_ref:
                        ref_array = np.array(pil_ref)
                        ref_w, ref_h = pil_ref.size

                    test_aspect = float(orig_w) / float(orig_h)
                    ref_aspect = float(ref_w) / float(ref_h)
                    aspect_delta = abs(ref_aspect - test_aspect) / test_aspect
                    if aspect_delta > 0.05:
                        raise HTTPException(
                            status_code=422,
                            detail={
                                "code": "REFERENCE_INCOMPATIBLE",
                                "message": (
                                    f"Reference image aspect ratio ({ref_aspect:.3f}) differs from test image "
                                    f"({test_aspect:.3f}) by {aspect_delta * 100.0:.1f}%, exceeding the 5% tolerance."
                                ),
                            },
                        )
                except HTTPException:
                    raise
                except Exception as exc:
                    logger.error("Failed to read reference image '%s': %s", ref_image_path, str(exc), exc_info=True)
                    raise HTTPException(
                        status_code=500,
                        detail={"code": "IMAGE_LOAD_ERROR", "message": "Failed to decode stored reference image."},
                    )

            orig_mean = float(np.mean(test_array))
            orig_std = float(np.std(test_array))
            original_stats = ImageStats(
                width=orig_w,
                height=orig_h,
                mean=round(orig_mean, 2),
                std=round(orig_std, 2),
            )

            # 5. Parse configurations
            prep_overrides = preprocessing_overrides or config_overrides
            try:
                prep_config = PreprocessingConfig(**prep_overrides) if prep_overrides else PreprocessingConfig()
                det_config = DetectionConfig(**detection_overrides) if detection_overrides else DetectionConfig()
                filter_config = NoiseFilterConfig(**noise_filter_overrides) if noise_filter_overrides else NoiseFilterConfig()
                measurement_config = (
                    MeasurementConfig(**measurement_overrides) if measurement_overrides else MeasurementConfig()
                )
                severity_config = SeverityConfig(**severity_overrides) if severity_overrides else SeverityConfig()
                pattern_config = PatternConfig(**pattern_overrides) if pattern_overrides else PatternConfig()
            except Exception as exc:
                raise HTTPException(
                    status_code=422,
                    detail={"code": "INVALID_CONFIG", "message": f"Invalid pipeline configuration: {str(exc)}"},
                )

            # 6. Execute full CV pipeline
            logger.info(
                "Executing full inspection pipeline: inspection_id=%s, upload_id=%s, has_reference=%s",
                inspection_row.id,
                clean_uuid,
                ref_array is not None,
            )
            try:
                pipeline_result = run_full_pipeline(
                    test_image=test_array,
                    reference_image=ref_array,
                    prep_config=prep_config,
                    det_config=det_config,
                    filter_config=filter_config,
                    measurement_config=measurement_config,
                    severity_config=severity_config,
                    pattern_config=pattern_config,
                    on_stage=lambda stage_name: status_store.set_stage(clean_uuid, stage_name),
                )
            except ReferenceIncompatibleError as exc:
                raise HTTPException(
                    status_code=422,
                    detail={"code": "REFERENCE_INCOMPATIBLE", "message": str(exc)},
                )

            status_store.set_stage(clean_uuid, "generating_report")

            preprocessed_test = pipeline_result.preprocessed_test
            det_result = pipeline_result.detection

            prep_h, prep_w = preprocessed_test.shape[:2]
            preprocessed_stats = ImageStats(
                width=prep_w,
                height=prep_h,
                mean=round(float(np.mean(preprocessed_test)), 2),
                std=round(float(np.std(preprocessed_test)), 2),
            )

            # 7. Save image overlays to data/results/<upload_id>/
            results_dir = settings.results_path / clean_uuid
            results_dir.mkdir(parents=True, exist_ok=True)

            Image.fromarray(preprocessed_test).save(results_dir / "preprocessed.png", format="PNG")
            Image.fromarray(det_result.difference_image).save(results_dir / "difference.png", format="PNG")
            Image.fromarray(det_result.morphology_mask).save(results_dir / "mask.png", format="PNG")
            Image.fromarray(pipeline_result.filtered_mask).save(results_dir / "filtered_mask.png", format="PNG")
            Image.fromarray(pipeline_result.bboxes_image).save(results_dir / "bboxes.png", format="PNG")
            Image.fromarray(pipeline_result.contours_image).save(results_dir / "contours.png", format="PNG")
            Image.fromarray(pipeline_result.final_overlay_image).save(results_dir / "final_overlay.png", format="PNG")
            Image.fromarray(pipeline_result.pattern_overlay_image).save(results_dir / "pattern_overlay.png", format="PNG")

            regions_by_id = {r.id: r for r in pipeline_result.accepted_regions}
            defect_db_records: List[Dict[str, Any]] = []
            accepted_json_records: List[Dict[str, Any]] = []
            for m in pipeline_result.measurements:
                src_region = regions_by_id.get(m.id)
                bx, by, bw, bh = m.bbox
                cx, cy = m.centroid
                reason_str = src_region.decision_reason if src_region else ""
                contour_lists = [[int(px), int(py)] for px, py in m.contour]
                defect_db_records.append(
                    {
                        "region_no": int(m.id),
                        "area": int(m.area_px),
                        "detected_area": int(m.detected_area_px),
                        "width": int(m.width_px),
                        "height": int(m.height_px),
                        "centroid_x": float(cx),
                        "centroid_y": float(cy),
                        "bbox_x": int(bx),
                        "bbox_y": int(by),
                        "bbox_width": int(bw),
                        "bbox_height": int(bh),
                        "perimeter": float(m.perimeter_px),
                        "equivalent_diameter": float(m.equivalent_diameter_px),
                        "length": float(m.length_px),
                        "thickness": float(m.thickness_px),
                        "angle_deg": float(m.angle_deg),
                        "elongation": float(m.elongation),
                        "polarity": m.polarity,
                        "mean_contrast": float(m.mean_contrast),
                        "peak_contrast": float(m.peak_contrast),
                        "contrast_to_noise": float(m.contrast_to_noise),
                        "touches_edge": bool(m.touches_edge),
                        "detected_by": m.detected_by,
                        "severity": m.severity,
                        "defect_type": None,
                        "confidence": None,
                        "why_flagged": m.why_flagged,
                        "contour": contour_lists,
                        "reason": reason_str,
                        "area_um2": m.area_um2,
                        "length_um": m.length_um,
                        "thickness_um": m.thickness_um,
                    }
                )
                accepted_json_records.append(
                    AcceptedRegionResponse(
                        id=m.id,
                        bbox=m.bbox,
                        width_px=m.width_px,
                        height_px=m.height_px,
                        area_px=m.area_px,
                        detected_area_px=m.detected_area_px,
                        centroid=m.centroid,
                        perimeter_px=m.perimeter_px,
                        equivalent_diameter_px=m.equivalent_diameter_px,
                        length_px=m.length_px,
                        thickness_px=m.thickness_px,
                        angle_deg=m.angle_deg,
                        elongation=m.elongation,
                        polarity=m.polarity,
                        mean_contrast=m.mean_contrast,
                        peak_contrast=m.peak_contrast,
                        contrast_to_noise=m.contrast_to_noise,
                        touches_edge=m.touches_edge,
                        detected_by=m.detected_by,
                        why_flagged=m.why_flagged,
                        severity=m.severity,
                        area_um2=m.area_um2,
                        length_um=m.length_um,
                        thickness_um=m.thickness_um,
                        contour=m.contour,
                        reason=reason_str,
                        original_bbox=m.bbox,
                        original_centroid=m.centroid,
                    ).model_dump()
                )

            rejected_json_records = [
                {
                    "id": rr.id,
                    "area_px": rr.area_px,
                    "mean_contrast": rr.mean_contrast,
                    "touches_edge": rr.touches_edge,
                    "rule": rr.rule,
                    "reason": rr.reason,
                }
                for rr in pipeline_result.rejected_regions
            ]

            detection_details = {
                "threshold_method": det_result.threshold_method,
                "threshold_value": round(float(det_result.threshold_value), 2),
                "threshold_reason": det_result.threshold_reason,
                "wafer_center": [round(det_result.wafer_center[0], 2), round(det_result.wafer_center[1], 2)],
                "wafer_radius": round(float(det_result.wafer_radius), 2),
                "is_full_frame": det_result.is_full_frame,
                "alignment_info": det_result.alignment_info,
                "die_pitch": list(det_result.die_pitch) if det_result.die_pitch is not None else None,
                "peak_strength": det_result.peak_strength,
                "coverage": det_result.coverage,
                "unverifiable_fraction": det_result.unverifiable_fraction,
            }

            summary = pipeline_result.summary
            model_info = pipeline_result.model
            pattern = pipeline_result.pattern
            pattern_dict = summary.get("pattern")

            # 8. Atomically save Inspection fields, Defects, and ProcessingRun in ONE transaction
            full_config_snapshot = {
                "preprocessing": prep_config.model_dump(),
                "detection": det_config.model_dump(),
                "noise_filter": filter_config.model_dump(),
                "measurement": measurement_config.model_dump(),
                "severity": severity_config.model_dump(),
                "pattern": pattern_config.model_dump(),
            }
            prep_method_str = f"gray={prep_config.to_grayscale_enabled},denoise={prep_config.denoise_method if prep_config.denoise_enabled else 'off'},contrast={prep_config.contrast_method}"
            morph_method_str = ",".join(f"{op}:{k}" for op, k in det_config.morphology_ops)

            inspection_fields = {
                "detection_path": det_result.path,
                "detection_label": det_result.label,
                "defect_count": int(summary["defect_count"]),
                "total_defect_area": int(summary["total_defect_area_px"]),
                "largest_defect_area": int(summary["largest_defect_area_px"]),
                "average_defect_area": float(summary["average_defect_area_px"]),
                "severity": summary["severity"],
                "severity_reason": summary["severity_reason"],
                "calibration_status": summary["calibration_status"],
                "pixel_size_um": summary.get("pixel_size_um"),
                "pattern_type": pattern.type,
                "pattern_confidence": float(pattern.confidence),
                "pattern_p_value": pattern.p_value,
                "pattern_explanation": pattern.explanation,
                "pattern_metrics": pattern.metrics,
                "pattern_supporting_ids": list(pattern.supporting_defect_ids),
                "pattern_warnings": list(pattern.warnings),
                "stage_counts": pipeline_result.stage_counts,
                "rejected_regions": rejected_json_records,
                "warnings": list(det_result.warnings),
                "original_stats": original_stats.model_dump(),
                "preprocessed_stats": preprocessed_stats.model_dump(),
                "step_log": pipeline_result.step_log,
                "detection_details": detection_details,
                "model_name": model_info["name"],
                "model_version": model_info["version"],
                "detection_method": model_info["detection_method"],
                "processing_time": float(pipeline_result.processing_time_s),
            }
            processing_run_record = {
                "preprocessing_method": prep_method_str,
                "threshold_method": det_result.threshold_method,
                "threshold_value": round(float(det_result.threshold_value), 2),
                "morphology_method": morph_method_str,
                "minimum_area": int(filter_config.min_area_px),
                "config": full_config_snapshot,
                "stage_timings": pipeline_result.stage_timings_ms,
                "processing_time": float(pipeline_result.processing_time_s),
            }

            try:
                saved_row = crud.save_results(
                    session,
                    inspection_row.id,
                    inspection_fields=inspection_fields,
                    defect_records=defect_db_records,
                    processing_run_record=processing_run_record,
                )
            except SQLAlchemyError as exc:
                status_store.fail_run(clean_uuid, "Database error while saving inspection results.")
                raise _db_error_http("save_results", exc)

            # 9. Write run.json strictly as a reproducibility artifact
            response_obj = _build_report_from_db(saved_row)
            run_meta = {
                **response_obj.model_dump(),
                "scale_factor": pipeline_result.scale_factor,
                "preprocessing_config": prep_config.model_dump(),
                "detection_config": det_config.model_dump(),
                "noise_filter_config": filter_config.model_dump(),
                "measurement_config": measurement_config.model_dump(),
                "severity_config": severity_config.model_dump(),
                "pattern_config": pattern_config.model_dump(),
                "noise_sigma": pipeline_result.noise_sigma,
                "derived_min_mean_contrast": pipeline_result.derived_min_mean_contrast,
                "regions": accepted_json_records,
                "rejected_regions": rejected_json_records,
                "pattern": pattern_dict,
            }
            with open(results_dir / "run.json", "w", encoding="utf-8") as f:
                json.dump(run_meta, f, indent=2)

            # 10. Compute and cache four-stage noise experiment evaluation
            try:
                meta_path = upload_dir / "meta.json"
                gt_spec = None
                if meta_path.is_file():
                    with open(meta_path, "r", encoding="utf-8") as mf:
                        m_data = json.load(mf)
                        gt_spec = m_data.get("ground_truth")

                from app.cv.noise_experiment import evaluate_inspection_noise_experiment
                noise_data = evaluate_inspection_noise_experiment(
                    result=pipeline_result,
                    min_area_px=filter_config.min_area_px,
                    ground_truth=gt_spec,
                )
                noise_data["inspection_id"] = int(saved_row.id)
                noise_data["upload_id"] = clean_uuid
                with open(results_dir / "noise_experiment.json", "w", encoding="utf-8") as nef:
                    json.dump(noise_data, nef, indent=2)
            except Exception as exp_err:
                logger.warning("Failed to save noise_experiment.json for upload_id=%s: %s", clean_uuid, exp_err)

            status_store.complete_run(clean_uuid)
            logger.info(
                "Inspection pipeline completed and persisted: inspection_id=%s, upload_id=%s",
                saved_row.id,
                clean_uuid,
            )
            return response_obj

        except HTTPException as http_exc:
            if http_exc.status_code != 503:
                detail = http_exc.detail
                err_msg = detail.get("message", str(detail)) if isinstance(detail, dict) else str(detail)
                _safe_mark_failed(session, inspection_row.id if inspection_row else None, err_msg)
                status_store.fail_run(clean_uuid, err_msg)
            raise
        except Exception as exc:
            logger.error("Unexpected pipeline failure for upload_id=%s: %s", clean_uuid, str(exc), exc_info=True)
            sanitized = "Inspection failed during image processing."
            _safe_mark_failed(session, inspection_row.id if inspection_row else None, sanitized)
            status_store.fail_run(clean_uuid, sanitized)
            raise HTTPException(
                status_code=500,
                detail={"code": "PIPELINE_ERROR", "message": sanitized},
            )
    finally:
        if owns_session:
            session.close()


# ---------------------------------------------------------------------------
# Database-backed Phase 8 Inspection Query Services
# ---------------------------------------------------------------------------
def get_inspection_report_by_id(db: Session, inspection_id: int) -> InspectionRunResponse:
    """Build the full InspectionRunResponse strictly from database rows (never run.json)."""
    try:
        row = crud.get_inspection(db, inspection_id)
    except SQLAlchemyError as exc:
        raise _db_error_http("get_inspection", exc)

    if row is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "INSPECTION_NOT_FOUND", "message": f"Inspection '{inspection_id}' was not found."},
        )

    if row.status in ("pending", "running"):
        raise HTTPException(
            status_code=409,
            detail={
                "code": "RUN_NOT_FINISHED",
                "message": f"Inspection '{inspection_id}' has status '{row.status}' and has not finished yet.",
            },
        )
    if row.status == "failed":
        raise HTTPException(
            status_code=409,
            detail={
                "code": "RUN_FAILED",
                "message": row.error_message or "Inspection failed during processing.",
            },
        )

    return _build_report_from_db(row)


def get_inspection_defects_by_id(db: Session, inspection_id: int) -> List[AcceptedRegionResponse]:
    """Return all measured defects for an inspection from the database."""
    try:
        row = crud.get_inspection(db, inspection_id)
    except SQLAlchemyError as exc:
        raise _db_error_http("get_inspection_defects", exc)

    if row is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "INSPECTION_NOT_FOUND", "message": f"Inspection '{inspection_id}' was not found."},
        )
    return [_defect_row_to_schema(d) for d in sorted(row.defects, key=lambda x: (x.region_no, x.id))]


def get_inspection_images_by_id(db: Session, inspection_id: int) -> InspectionImagesResponse:
    """Return map of available image kinds to `/inspection/{id}/image/{kind}` URLs."""
    try:
        row = crud.get_inspection(db, inspection_id)
    except SQLAlchemyError as exc:
        raise _db_error_http("get_inspection_images", exc)

    if row is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "INSPECTION_NOT_FOUND", "message": f"Inspection '{inspection_id}' was not found."},
        )

    images: Dict[str, str] = {}
    upload_dir = settings.upload_path / row.upload_id
    if upload_dir.is_dir():
        test_files = [f for f in upload_dir.glob("test.*") if f.suffix.lower() in MIME_TYPE_MAP]
        if test_files and test_files[0].is_file():
            images["original"] = f"/inspection/{row.id}/image/original"

    results_dir = settings.results_path / row.upload_id
    for kind, fname in RESULT_PNG_MAP.items():
        if (results_dir / fname).is_file():
            images[kind] = f"/inspection/{row.id}/image/{kind}"

    return InspectionImagesResponse(
        inspection_id=int(row.id),
        upload_id=row.upload_id,
        images=images,
    )


def get_inspection_image_file_by_id(db: Session, inspection_id: int, kind: str) -> Tuple[Path, str]:
    """Serve an image artifact by integer inspection_id using the same allowlist & UUID safety."""
    try:
        row = crud.get_inspection(db, inspection_id)
    except SQLAlchemyError as exc:
        raise _db_error_http("get_inspection_image", exc)

    if row is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "INSPECTION_NOT_FOUND", "message": f"Inspection '{inspection_id}' was not found."},
        )
    return get_run_image_file_path(upload_id=row.upload_id, kind=kind)


def get_inspection_status_by_id(db: Session, inspection_id: int) -> InspectionStatusResponse:
    """Return live single-process stage status for an inspection by integer ID."""
    try:
        row = crud.get_inspection(db, inspection_id)
    except SQLAlchemyError as exc:
        raise _db_error_http("get_inspection_status", exc)

    if row is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "INSPECTION_NOT_FOUND", "message": f"Inspection '{inspection_id}' was not found."},
        )

    live = status_store.get_status(row.upload_id)
    if live is not None:
        return InspectionStatusResponse(inspection_id=int(row.id), **live)

    if row.status == "completed":
        return InspectionStatusResponse(
            inspection_id=int(row.id),
            upload_id=row.upload_id,
            status="completed",
            current_stage="completed",
            completed_stages=list(PIPELINE_STAGES),
            stage_history=list(PIPELINE_STAGES) + ["completed"],
            stages=list(PIPELINE_STAGES),
            error=None,
            updated_at=row.updated_at.isoformat() if row.updated_at else None,
        )
    if row.status == "failed":
        return InspectionStatusResponse(
            inspection_id=int(row.id),
            upload_id=row.upload_id,
            status="failed",
            current_stage="failed",
            completed_stages=[],
            stage_history=["failed"],
            stages=list(PIPELINE_STAGES),
            error=row.error_message,
            updated_at=row.updated_at.isoformat() if row.updated_at else None,
        )
    return InspectionStatusResponse(
        inspection_id=int(row.id),
        upload_id=row.upload_id,
        status="pending",
        current_stage="pending",
        completed_stages=[],
        stage_history=[],
        stages=list(PIPELINE_STAGES),
        error=None,
        updated_at=row.updated_at.isoformat() if row.updated_at else None,
    )


def get_inspection_noise_experiment_by_id(
    db: Session, inspection_id: int
) -> InspectionNoiseExperimentResponse:
    """Return four-stage noise experiment candidate metrics for a completed inspection."""
    try:
        row = crud.get_inspection(db, inspection_id)
    except SQLAlchemyError as exc:
        raise _db_error_http("get_inspection_noise_experiment", exc)

    if row is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "INSPECTION_NOT_FOUND", "message": f"Inspection '{inspection_id}' was not found."},
        )

    if row.status in ("pending", "running"):
        raise HTTPException(
            status_code=409,
            detail={
                "code": "RUN_NOT_FINISHED",
                "message": f"Inspection '{inspection_id}' has status '{row.status}' and has not finished yet.",
            },
        )
    if row.status == "failed":
        raise HTTPException(
            status_code=409,
            detail={
                "code": "RUN_FAILED",
                "message": row.error_message or "Inspection failed during processing.",
            },
        )

    results_dir = settings.results_path / row.upload_id
    noise_json_path = results_dir / "noise_experiment.json"
    if noise_json_path.is_file():
        try:
            with open(noise_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                data["inspection_id"] = int(row.id)
                data["upload_id"] = row.upload_id
                return InspectionNoiseExperimentResponse.model_validate(data)
        except Exception as exc:
            logger.warning("Could not read cached noise_experiment.json for %s: %s", row.upload_id, exc)

    # Re-evaluate from stored images and metadata
    upload_dir = settings.upload_path / row.upload_id
    if not upload_dir.is_dir():
        raise HTTPException(
            status_code=404,
            detail={"code": "UPLOAD_NOT_FOUND", "message": f"Upload session '{row.upload_id}' was not found."},
        )

    test_files = [f for f in upload_dir.glob("test.*") if f.suffix.lower() in MIME_TYPE_MAP]
    if not test_files or not test_files[0].is_file():
        raise HTTPException(
            status_code=404,
            detail={"code": "TEST_IMAGE_NOT_FOUND", "message": "Uploaded test image file is missing."},
        )

    test_img = np.array(Image.open(test_files[0]))
    ref_files = [f for f in upload_dir.glob("reference.*") if f.suffix.lower() in MIME_TYPE_MAP]
    ref_img = np.array(Image.open(ref_files[0])) if ref_files and ref_files[0].is_file() else None

    gt_spec = None
    meta_path = upload_dir / "meta.json"
    if meta_path.is_file():
        try:
            with open(meta_path, "r", encoding="utf-8") as mf:
                m_data = json.load(mf)
                gt_spec = m_data.get("ground_truth")
        except Exception:
            gt_spec = None

    from app.cv.noise_experiment import run_single_noise_evaluation

    det_cfg = DetectionConfig(
        path="golden" if ref_img is not None else "self_reference",
        no_reference_mode="self_reference",
    )
    res = run_single_noise_evaluation(
        test_image=test_img,
        reference_image=ref_img,
        det_config=det_cfg,
        ground_truth=gt_spec,
    )
    res["inspection_id"] = int(row.id)
    res["upload_id"] = row.upload_id

    # Cache for future calls
    try:
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(noise_json_path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
    except Exception as exc:
        logger.warning("Failed to cache noise_experiment.json: %s", exc)

    return InspectionNoiseExperimentResponse.model_validate(res)


def list_inspection_history(
    db: Session,
    *,
    search: Optional[str] = None,
    severity: Optional[str] = None,
    sort: str = "created_at",
    order: str = "desc",
    page: int = 1,
    page_size: int = 20,
) -> HistoryListResponse:
    """Return paginated, filtered, and sorted inspection history from SQLite/PostgreSQL."""
    try:
        rows, total = crud.list_history(
            db,
            search=search,
            severity=severity,
            sort=sort,
            order=order,
            page=page,
            page_size=page_size,
        )
    except SQLAlchemyError as exc:
        raise _db_error_http("list_history", exc)

    safe_size = max(1, min(100, int(page_size)))
    total_pages = max(1, int(math.ceil(total / float(safe_size)))) if total > 0 else 1
    return HistoryListResponse(
        items=[_history_item_from_row(r) for r in rows],
        total=total,
        page=max(1, int(page)),
        page_size=safe_size,
        total_pages=total_pages,
    )


def delete_inspection_by_id(db: Session, inspection_id: int) -> DeleteInspectionResponse:
    """Delete an inspection row (cascading defects & processing runs) and remove upload/result files."""
    try:
        upload_id = crud.delete_inspection(db, inspection_id)
    except SQLAlchemyError as exc:
        raise _db_error_http("delete_inspection", exc)

    if upload_id is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "INSPECTION_NOT_FOUND", "message": f"Inspection '{inspection_id}' was not found."},
        )

    # Remove stored upload and result directories safely (validated UUID)
    clean_uuid = _validate_upload_uuid(upload_id)
    upload_dir = settings.upload_path / clean_uuid
    results_dir = settings.results_path / clean_uuid
    if upload_dir.is_dir():
        shutil.rmtree(upload_dir, ignore_errors=True)
    if results_dir.is_dir():
        shutil.rmtree(results_dir, ignore_errors=True)

    logger.info("Deleted inspection_id=%s and artifacts for upload_id=%s", inspection_id, clean_uuid)
    return DeleteInspectionResponse(deleted=True, inspection_id=int(inspection_id), upload_id=clean_uuid)


def get_dashboard_statistics(db: Session) -> DashboardStatsResponse:
    """Return SQL-aggregated dashboard KPIs and chart series (empty-safe)."""
    try:
        stats = crud.get_dashboard_stats(db)
    except SQLAlchemyError as exc:
        raise _db_error_http("dashboard_stats", exc)

    return DashboardStatsResponse(
        total_inspections=stats["total_inspections"],
        total_defects=stats["total_defects"],
        high_severity_inspections=stats["high_severity_inspections"],
        average_defect_area=stats["average_defect_area"],
        recent_inspections=[_history_item_from_row(r) for r in stats["recent_inspections"]],
        recent_defect_counts=[RecentDefectCountPoint(**pt) for pt in stats["recent_defect_counts"]],
        severity_distribution=[SeverityCountPoint(**pt) for pt in stats["severity_distribution"]],
        defect_area_histogram=[AreaHistogramPoint(**pt) for pt in stats["defect_area_histogram"]],
        pattern_distribution=[PatternCountPoint(**pt) for pt in stats["pattern_distribution"]],
    )


# ---------------------------------------------------------------------------
# Compatibility Upload-ID Helpers (kept for Phases 3-7 API compatibility)
# ---------------------------------------------------------------------------
def get_inspection_result(upload_id: str, db: Optional[Session] = None) -> InspectionRunResponse:
    """Return completed inspection result by UUID upload_id (preferring DB rows, fallback run.json)."""
    clean_uuid = _validate_upload_uuid(upload_id)

    live_status = status_store.get_status(clean_uuid)
    if live_status and live_status["status"] == "running":
        raise HTTPException(
            status_code=409,
            detail={
                "code": "RUN_IN_PROGRESS",
                "message": f"Inspection run '{clean_uuid}' is still in progress (stage: {live_status['current_stage']}).",
            },
        )

    owns_session = db is None
    session = db if db is not None else SessionLocal()
    try:
        row = crud.get_inspection_by_upload_id(session, clean_uuid)
        if row is not None:
            if row.status == "completed":
                return _build_report_from_db(row)
            if row.status in ("pending", "running"):
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "RUN_NOT_FINISHED",
                        "message": f"Inspection run for upload '{clean_uuid}' has not completed yet.",
                    },
                )
    except HTTPException:
        raise
    except SQLAlchemyError as exc:
        raise _db_error_http("get_inspection_result", exc)
    finally:
        if owns_session:
            session.close()

    upload_dir = settings.upload_path / clean_uuid
    run_json_path = settings.results_path / clean_uuid / "run.json"
    if not run_json_path.is_file():
        if upload_dir.is_dir() or live_status is not None:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "RUN_NOT_FINISHED",
                    "message": f"Inspection run for upload '{clean_uuid}' has not completed yet.",
                },
            )
        raise HTTPException(
            status_code=404,
            detail={"code": "UPLOAD_NOT_FOUND", "message": f"Upload session '{clean_uuid}' was not found."},
        )

    raw = json.loads(run_json_path.read_text(encoding="utf-8"))
    for kind in ALLOWED_RUN_IMAGE_KINDS:
        raw.setdefault(f"{kind}_image_url", f"/inspection/run/{clean_uuid}/image/{kind}")
    return InspectionRunResponse.model_validate(raw)


def get_inspection_status(upload_id: str) -> InspectionStatusResponse:
    """Return live single-process in-memory stage status for an inspection run by UUID upload_id."""
    clean_uuid = _validate_upload_uuid(upload_id)
    live = status_store.get_status(clean_uuid)
    if live is not None:
        return InspectionStatusResponse(**live)

    run_json_path = settings.results_path / clean_uuid / "run.json"
    if run_json_path.is_file():
        return InspectionStatusResponse(
            upload_id=clean_uuid,
            status="completed",
            current_stage="completed",
            completed_stages=list(PIPELINE_STAGES),
            stage_history=list(PIPELINE_STAGES) + ["completed"],
            stages=list(PIPELINE_STAGES),
            error=None,
            updated_at=None,
        )

    upload_dir = settings.upload_path / clean_uuid
    if upload_dir.is_dir():
        return InspectionStatusResponse(
            upload_id=clean_uuid,
            status="not_started",
            current_stage="not_started",
            completed_stages=[],
            stage_history=[],
            stages=list(PIPELINE_STAGES),
            error=None,
            updated_at=None,
        )

    raise HTTPException(
        status_code=404,
        detail={"code": "UPLOAD_NOT_FOUND", "message": f"Upload session '{clean_uuid}' was not found."},
    )


def get_run_image_file_path(upload_id: str, kind: str) -> Tuple[Path, str]:
    """Retrieve file path and media type for inspection run artifacts."""
    clean_kind = kind.strip().lower()
    if clean_kind not in ALLOWED_RUN_IMAGE_KINDS:
        logger.warning("Invalid run image kind requested: %s", kind)
        raise HTTPException(
            status_code=400,
            detail={
                "code": "INVALID_KIND",
                "message": f"Image kind '{kind}' is invalid. Allowed: {sorted(list(ALLOWED_RUN_IMAGE_KINDS))}.",
            },
        )

    clean_uuid = _validate_upload_uuid(upload_id)

    if clean_kind == "original":
        upload_dir = settings.upload_path / clean_uuid
        if not upload_dir.is_dir():
            raise HTTPException(
                status_code=404,
                detail={"code": "UPLOAD_NOT_FOUND", "message": f"Upload '{clean_uuid}' was not found."},
            )
        test_files = [f for f in upload_dir.glob("test.*") if f.suffix.lower() in MIME_TYPE_MAP]
        if not test_files or not test_files[0].is_file():
            raise HTTPException(
                status_code=404,
                detail={"code": "IMAGE_NOT_FOUND", "message": f"Original image not found for upload {clean_uuid}."},
            )
        file_path = test_files[0]
        media_type = MIME_TYPE_MAP.get(file_path.suffix.lower(), "image/png")
        return file_path, media_type

    results_dir = settings.results_path / clean_uuid
    filename = RESULT_PNG_MAP.get(clean_kind)
    if not filename:
        raise HTTPException(
            status_code=400,
            detail={"code": "INVALID_KIND", "message": f"Unsupported image kind '{kind}'."},
        )

    artifact_path = results_dir / filename
    if not artifact_path.is_file():
        raise HTTPException(
            status_code=404,
            detail={"code": "IMAGE_NOT_FOUND", "message": f"{clean_kind} image not found for upload {clean_uuid}."},
        )
    return artifact_path, "image/png"
