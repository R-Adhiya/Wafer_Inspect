"""Service orchestrating wafer defect inspection workflows, preprocessing, detection, and filtering."""

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
from app.db.schemas import (
    AcceptedRegionResponse,
    DetectionSummaryResponse,
    ImageStats,
    InspectionRunResponse,
    InspectionStatusResponse,
    InspectionSummaryResponse,
    ModelInfoResponse,
    PatternResultResponse,
    RejectedRegionResponse,
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


def run_inspection_pipeline(
    upload_id: str,
    config_overrides: Optional[Dict[str, Any]] = None,
    preprocessing_overrides: Optional[Dict[str, Any]] = None,
    detection_overrides: Optional[Dict[str, Any]] = None,
    noise_filter_overrides: Optional[Dict[str, Any]] = None,
    measurement_overrides: Optional[Dict[str, Any]] = None,
    severity_overrides: Optional[Dict[str, Any]] = None,
    pattern_overrides: Optional[Dict[str, Any]] = None,
) -> InspectionRunResponse:
    """Load uploaded wafer images, execute full CV pipeline, and store run artifacts.

    Args:
        upload_id: UUID4 identifier of an existing uploaded wafer inspection session.
        config_overrides: Optional dictionary of parameters to override PreprocessingConfig.
        preprocessing_overrides: Optional dictionary alias for PreprocessingConfig overrides.
        detection_overrides: Optional dictionary of parameters to override DetectionConfig.
        noise_filter_overrides: Optional dictionary of parameters to override NoiseFilterConfig.

    Returns:
        InspectionRunResponse: Execution summary, step log, stats, detection results,
        accepted/rejected regions, full stage_counts funnel, and image URLs.
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

    test_files = list(upload_dir.glob("test.*"))
    test_files = [f for f in test_files if f.suffix.lower() in MIME_TYPE_MAP]

    if not test_files or not test_files[0].is_file():
        logger.warning("Test image file not found for upload '%s'", clean_uuid)
        raise HTTPException(
            status_code=404,
            detail={"code": "TEST_IMAGE_NOT_FOUND", "message": f"Test image not found for upload '{clean_uuid}'."},
        )

    test_image_path = test_files[0]

    # Mark live status store as started at stage "preparing_image"
    status_store.start_run(clean_uuid)

    # Read upload meta.json if present for report details card
    test_filename: Optional[str] = test_image_path.name
    test_mode: Optional[str] = None
    reference_filename: Optional[str] = None
    upload_meta_path = upload_dir / "meta.json"
    if upload_meta_path.is_file():
        try:
            u_meta = json.loads(upload_meta_path.read_text(encoding="utf-8"))
            if isinstance(u_meta.get("test_image"), dict):
                test_filename = u_meta["test_image"].get("original_filename", test_filename)
                test_mode = u_meta["test_image"].get("mode")
            if isinstance(u_meta.get("reference_image"), dict):
                reference_filename = u_meta["reference_image"].get("original_filename")
        except Exception:
            pass

    # 3. Check for golden reference image
    ref_files = list(upload_dir.glob("reference.*"))
    ref_files = [f for f in ref_files if f.suffix.lower() in MIME_TYPE_MAP]
    ref_image_path = ref_files[0] if (ref_files and ref_files[0].is_file()) else None
    if ref_image_path is not None and not reference_filename:
        reference_filename = ref_image_path.name

    try:
        # 4. Load test image into NumPy array
        try:
            with Image.open(test_image_path) as pil_test:
                test_array = np.array(pil_test)
                orig_w, orig_h = pil_test.size
                if not test_mode:
                    test_mode = pil_test.mode
        except Exception as exc:
            logger.error("Failed to read image at '%s': %s", test_image_path, str(exc), exc_info=True)
            raise HTTPException(
                status_code=500,
                detail={"code": "IMAGE_LOAD_ERROR", "message": "Failed to decode stored wafer image."},
            )

        ref_array: Optional[np.ndarray] = None
        if ref_image_path is not None:
            try:
                with Image.open(ref_image_path) as pil_ref:
                    ref_array = np.array(pil_ref)
                    ref_w, ref_h = pil_ref.size

                # Enforce 5% aspect ratio compatibility rule
                test_aspect = float(orig_w) / float(orig_h)
                ref_aspect = float(ref_w) / float(ref_h)
                aspect_delta = abs(ref_aspect - test_aspect) / test_aspect

                if aspect_delta > 0.05:
                    logger.warning(
                        "Reference incompatible for upload '%s': test_aspect=%.3f, ref_aspect=%.3f, delta=%.1f%%",
                        clean_uuid, test_aspect, ref_aspect, aspect_delta * 100.0,
                    )
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
                logger.error("Failed to read reference image at '%s': %s", ref_image_path, str(exc), exc_info=True)
                raise HTTPException(
                    status_code=500,
                    detail={"code": "IMAGE_LOAD_ERROR", "message": "Failed to decode stored reference image."},
                )

        # Compute original test statistics
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
        except Exception as exc:
            logger.warning("Invalid preprocessing config for upload '%s': %s", clean_uuid, str(exc))
            raise HTTPException(
                status_code=422,
                detail={"code": "INVALID_CONFIG", "message": f"Invalid preprocessing configuration: {str(exc)}"},
            )

        try:
            det_config = DetectionConfig(**detection_overrides) if detection_overrides else DetectionConfig()
        except Exception as exc:
            logger.warning("Invalid detection config for upload '%s': %s", clean_uuid, str(exc))
            raise HTTPException(
                status_code=422,
                detail={"code": "INVALID_CONFIG", "message": f"Invalid detection configuration: {str(exc)}"},
            )

        try:
            filter_config = NoiseFilterConfig(**noise_filter_overrides) if noise_filter_overrides else NoiseFilterConfig()
        except Exception as exc:
            logger.warning("Invalid noise filter config for upload '%s': %s", clean_uuid, str(exc))
            raise HTTPException(
                status_code=422,
                detail={"code": "INVALID_CONFIG", "message": f"Invalid noise filter configuration: {str(exc)}"},
            )

        try:
            measurement_config = (
                MeasurementConfig(**measurement_overrides) if measurement_overrides else MeasurementConfig()
            )
        except Exception as exc:
            logger.warning("Invalid measurement config for upload '%s': %s", clean_uuid, str(exc))
            raise HTTPException(
                status_code=422,
                detail={"code": "INVALID_CONFIG", "message": f"Invalid measurement configuration: {str(exc)}"},
            )

        try:
            severity_config = SeverityConfig(**severity_overrides) if severity_overrides else SeverityConfig()
        except Exception as exc:
            logger.warning("Invalid severity config for upload '%s': %s", clean_uuid, str(exc))
            raise HTTPException(
                status_code=422,
                detail={"code": "INVALID_CONFIG", "message": f"Invalid severity configuration: {str(exc)}"},
            )

        try:
            pattern_config = PatternConfig(**pattern_overrides) if pattern_overrides else PatternConfig()
        except Exception as exc:
            logger.warning("Invalid pattern config for upload '%s': %s", clean_uuid, str(exc))
            raise HTTPException(
                status_code=422,
                detail={"code": "INVALID_CONFIG", "message": f"Invalid pattern configuration: {str(exc)}"},
            )

        # 6. Execute full CV pipeline (reporting live stages into single-process status_store)
        logger.info("Executing full inspection pipeline for upload_id=%s (has_reference=%s)", clean_uuid, ref_array is not None)
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
        prep_mean = float(np.mean(preprocessed_test))
        prep_std = float(np.std(preprocessed_test))
        preprocessed_stats = ImageStats(
            width=prep_w,
            height=prep_h,
            mean=round(prep_mean, 2),
            std=round(prep_std, 2),
        )

        # 7. Save output artifacts to data/results/<upload_id>/
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

        # Serialize accepted regions with full original-pixel measurements, contours, and severity
        regions_by_id = {r.id: r for r in pipeline_result.accepted_regions}
        accepted_responses = []
        accepted_json_records = []
        for m in pipeline_result.measurements:
            src_region = regions_by_id.get(m.id)
            item = AcceptedRegionResponse(
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
                reason=src_region.decision_reason if src_region else "",
                original_bbox=m.bbox,
                original_centroid=m.centroid,
            )
            accepted_responses.append(item)
            accepted_json_records.append(item.model_dump())

        rejected_responses = []
        rejected_json_records = []
        for rr in pipeline_result.rejected_regions:
            item = RejectedRegionResponse(
                id=rr.id,
                area_px=rr.area_px,
                mean_contrast=rr.mean_contrast,
                reason=rr.reason,
            )
            rejected_responses.append(item)
            rejected_json_records.append(
                {
                    "id": rr.id,
                    "area_px": rr.area_px,
                    "mean_contrast": rr.mean_contrast,
                    "touches_edge": rr.touches_edge,
                    "rule": rr.rule,
                    "reason": rr.reason,
                }
            )

        detection_summary = {
            "path": det_result.path,
            "label": det_result.label,
            "threshold_method": det_result.threshold_method,
            "threshold_value": round(float(det_result.threshold_value), 2),
            "threshold_reason": det_result.threshold_reason,
            "wafer_center": (round(det_result.wafer_center[0], 2), round(det_result.wafer_center[1], 2)),
            "wafer_radius": round(float(det_result.wafer_radius), 2),
            "is_full_frame": det_result.is_full_frame,
            "alignment_info": det_result.alignment_info,
            "stage_counts": pipeline_result.stage_counts,
            "warnings": det_result.warnings,
            "die_pitch": det_result.die_pitch,
            "peak_strength": det_result.peak_strength,
            "coverage": det_result.coverage,
            "unverifiable_fraction": det_result.unverifiable_fraction,
        }

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

        # Save run.json (full config, logs, detection, summary, model, pattern, timings, and measurements)
        pattern_dict = pipeline_result.summary.get("pattern")
        run_meta = {
            "upload_id": clean_uuid,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "test_filename": test_filename,
            "test_mode": test_mode,
            "reference_used": ref_array is not None,
            "reference_filename": reference_filename,
            "scale_factor": pipeline_result.scale_factor,
            "preprocessing_config": prep_config.model_dump(),
            "detection_config": det_config.model_dump(),
            "noise_filter_config": filter_config.model_dump(),
            "measurement_config": measurement_config.model_dump(),
            "severity_config": severity_config.model_dump(),
            "pattern_config": pattern_config.model_dump(),
            "noise_sigma": pipeline_result.noise_sigma,
            "derived_min_mean_contrast": pipeline_result.derived_min_mean_contrast,
            "original_stats": original_stats.model_dump(),
            "preprocessed_stats": preprocessed_stats.model_dump(),
            "step_log": pipeline_result.step_log,
            "detection": detection_summary,
            "stage_counts": pipeline_result.stage_counts,
            "summary": pipeline_result.summary,
            "model": pipeline_result.model,
            "pattern": pattern_dict,
            "processing_time_s": pipeline_result.processing_time_s,
            "stage_timings_ms": pipeline_result.stage_timings_ms,
            "regions": accepted_json_records,
            "rejected_regions": rejected_json_records,
            **image_urls,
        }
        with open(results_dir / "run.json", "w", encoding="utf-8") as f:
            json.dump(run_meta, f, indent=2)

        status_store.complete_run(clean_uuid)
        logger.info("Inspection pipeline completed and stored for upload_id=%s", clean_uuid)

        step_log_items = [StepLogItem(**step) for step in pipeline_result.step_log]
        stage_counts = [StageCount(**sc) for sc in pipeline_result.stage_counts]
        summary_response = InspectionSummaryResponse(**pipeline_result.summary)
        model_response = ModelInfoResponse(**pipeline_result.model)
        pattern_response = PatternResultResponse(**pattern_dict) if pattern_dict else None

        det_response = DetectionSummaryResponse(
            path=det_result.path,
            label=det_result.label,
            threshold_method=det_result.threshold_method,
            threshold_value=round(float(det_result.threshold_value), 2),
            threshold_reason=det_result.threshold_reason,
            wafer_center=(round(det_result.wafer_center[0], 2), round(det_result.wafer_center[1], 2)),
            wafer_radius=round(float(det_result.wafer_radius), 2),
            is_full_frame=det_result.is_full_frame,
            alignment_info=det_result.alignment_info,
            stage_counts=stage_counts,
            warnings=det_result.warnings,
            die_pitch=det_result.die_pitch,
            peak_strength=det_result.peak_strength,
            coverage=det_result.coverage,
            unverifiable_fraction=det_result.unverifiable_fraction,
        )

        return InspectionRunResponse(
            upload_id=clean_uuid,
            step_log=step_log_items,
            original_stats=original_stats,
            preprocessed_stats=preprocessed_stats,
            detection=det_response,
            stage_counts=stage_counts,
            regions=accepted_responses,
            rejected_regions=rejected_responses,
            summary=summary_response,
            model=model_response,
            pattern=pattern_response,
            processing_time_s=pipeline_result.processing_time_s,
            stage_timings_ms=pipeline_result.stage_timings_ms,
            test_filename=test_filename,
            test_mode=test_mode,
            reference_used=ref_array is not None,
            reference_filename=reference_filename,
            **image_urls,
        )
    except HTTPException as http_exc:
        detail = http_exc.detail
        err_msg = detail.get("message", str(detail)) if isinstance(detail, dict) else str(detail)
        status_store.fail_run(clean_uuid, err_msg)
        raise
    except Exception as exc:
        status_store.fail_run(clean_uuid, str(exc))
        raise


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


def get_inspection_result(upload_id: str) -> InspectionRunResponse:
    """Load and validate persisted run.json for a completed inspection.

    Returns:
      - 404 if upload_id is invalid or unknown (neither uploads nor results exist).
      - 409 if the upload exists (or is currently running) and no completed run.json is ready.
      - 200 validated InspectionRunResponse loaded from run.json.
    """
    clean_uuid = _validate_upload_uuid(upload_id)
    upload_dir = settings.upload_path / clean_uuid
    results_dir = settings.results_path / clean_uuid
    run_json_path = results_dir / "run.json"

    live_status = status_store.get_status(clean_uuid)
    if live_status and live_status["status"] == "running":
        raise HTTPException(
            status_code=409,
            detail={
                "code": "RUN_IN_PROGRESS",
                "message": f"Inspection run '{clean_uuid}' is still in progress (stage: {live_status['current_stage']}).",
            },
        )

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

    try:
        raw = json.loads(run_json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error("Failed to read run.json for '%s': %s", clean_uuid, str(exc), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"code": "RESULT_READ_ERROR", "message": "Failed to read stored inspection result."},
        )

    # Backfill image URLs if loading a run.json produced before URLs were persisted
    raw.setdefault("original_image_url", f"/inspection/run/{clean_uuid}/image/original")
    raw.setdefault("preprocessed_image_url", f"/inspection/run/{clean_uuid}/image/preprocessed")
    raw.setdefault("difference_image_url", f"/inspection/run/{clean_uuid}/image/difference")
    raw.setdefault("mask_image_url", f"/inspection/run/{clean_uuid}/image/mask")
    raw.setdefault("filtered_mask_image_url", f"/inspection/run/{clean_uuid}/image/filtered_mask")
    raw.setdefault("bboxes_image_url", f"/inspection/run/{clean_uuid}/image/bboxes")
    raw.setdefault("contours_image_url", f"/inspection/run/{clean_uuid}/image/contours")
    raw.setdefault("final_overlay_image_url", f"/inspection/run/{clean_uuid}/image/final_overlay")
    raw.setdefault("pattern_overlay_image_url", f"/inspection/run/{clean_uuid}/image/pattern_overlay")

    return InspectionRunResponse.model_validate(raw)


def get_inspection_status(upload_id: str) -> InspectionStatusResponse:
    """Return live single-process in-memory stage status for an inspection run."""
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
    """Retrieve file path and media type for inspection run artifacts.

    Args:
        upload_id: UUID4 identifier of the inspection session.
        kind: One of "original", "preprocessed", "difference", "mask",
              "filtered_mask", "bboxes", "contours".

    Returns:
        Tuple[Path, str]: (absolute_file_path, mime_type)
    """
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
        test_files = list(upload_dir.glob("test.*"))
        test_files = [f for f in test_files if f.suffix.lower() in MIME_TYPE_MAP]
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
