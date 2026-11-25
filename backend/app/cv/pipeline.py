"""Computer vision inspection pipeline orchestration.

Implements the multi-stage image preprocessing, classical defect detection,
connected-component segmentation, noise filtering, and comprehensive step logging.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import cv2
import numpy as np
from pydantic import BaseModel, Field, field_validator

from app.core.config import DETECTION_METHOD_LABEL, MODEL_NAME, MODEL_VERSION
from app.cv.detection import (
    DetectionResult,
    align_reference,
    apply_morphology,
    apply_threshold,
    compute_golden_difference,
    compute_no_reference_residual,
    count_connected_candidates,
    estimate_noise_sigma,
    estimate_wafer_mask,
    match_photometry,
)
from app.cv.measurement import (
    DetectedByPath,
    MeasurementConfig,
    RegionMeasurement,
    format_calibration_status,
    measure_regions,
    refine_regions,
)
from app.cv.noise_filter import (
    NoiseFilterConfig,
    NoiseFilterResult,
    RejectedRegion,
    filter_regions,
)
from app.cv.pattern_analysis import (
    PatternConfig,
    PatternResult,
    analyze_spatial_pattern,
    draw_pattern_overlay,
)
from app.cv.preprocessing import (
    PreprocessingConfig,
    denoise,
    enhance_contrast,
    normalize,
    resize_preserving_aspect,
    tile_image,
    to_grayscale,
)
from app.cv.segmentation import (
    Region,
    build_filtered_mask_and_labels,
    draw_bounding_boxes,
    draw_contours,
    draw_final_overlay,
    segment_regions,
)
from app.cv.self_reference import build_self_reference, estimate_die_pitch
from app.cv.severity import (
    SeverityConfig,
    assess_inspection_severity,
    classify_defect_severity,
)

logger = logging.getLogger("waferinspect.cv.pipeline")


class ReferenceIncompatibleError(ValueError):
    """Raised when golden reference aspect ratio differs from test image by > 5%."""


class DetectionConfig(BaseModel):
    """Configuration and parameter validation for defect detection."""

    path: Literal["auto", "golden", "self_reference", "no_reference"] = Field(
        default="auto",
        description="Detection strategy: 'auto', 'golden', 'self_reference', or 'no_reference'",
    )
    no_reference_mode: Literal["auto", "self_reference", "background"] = Field(
        default="auto",
        description=(
            "Strategy when no golden reference is supplied: "
            "'auto' (self-reference if periodic, else Gaussian background fallback), "
            "'self_reference', or 'background'."
        ),
    )
    self_ref_min_pitch: int = Field(
        default=8,
        ge=4,
        description="Minimum integer die pitch in pixels for self-reference periodicity",
    )
    self_ref_min_peak_strength: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Minimum normalized FFT autocorrelation peak prominence for periodicity",
    )
    self_ref_min_neighbors: int = Field(
        default=3,
        ge=1,
        le=8,
        description="Minimum valid in-wafer neighbouring dies required per verifiable pixel",
    )
    self_ref_shift_tolerance: bool = Field(
        default=True,
        description="Use 3x3 shift-window minimum absdiff to absorb +/- 1 px pattern-edge misregistration",
    )
    edge_margin_ratio: float = Field(
        default=0.02,
        ge=0.0,
        le=0.2,
        description="Wafer disk edge exclusion margin as fraction of radius",
    )
    max_alignment_shift: float = Field(
        default=10.0,
        ge=0.0,
        description="Maximum translation shift in pixels permitted for phase correlation",
    )
    min_alignment_response: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Minimum phase correlation peak response required to apply translation",
    )
    match_photometric: bool = Field(
        default=False,
        description="Photometrically match reference mean and standard deviation to test image",
    )
    background_blur_sigma_ratio: float = Field(
        default=0.05,
        gt=0.0,
        le=0.5,
        description="Gaussian filter sigma ratio relative to max image dimension for background subtraction",
    )
    threshold_method: Literal["otsu", "fixed", "adaptive"] = Field(
        default="otsu",
        description="Segmentation threshold method: 'otsu', 'fixed', or 'adaptive'",
    )
    fixed_threshold: float = Field(
        default=30.0,
        ge=0.0,
        le=255.0,
        description="Intensity difference threshold for 'fixed' method",
    )
    adaptive_block_size: int = Field(
        default=31,
        description="Block size for adaptive thresholding (must be odd positive integer)",
    )
    adaptive_c: float = Field(
        default=5.0,
        description="Subtracted constant for adaptive thresholding",
    )
    min_robust_contrast: float = Field(
        default=8.0,
        ge=0.0,
        description="Minimum 99.9th-percentile contrast cutoff guarding Otsu on flat/near-zero diffs",
    )
    noise_floor_k: float = Field(
        default=3.5,
        ge=1.0,
        description="Statistical sigma multiplier k for deriving the noise ceiling from MAD",
    )
    morphology_ops: List[Tuple[str, int]] = Field(
        default_factory=lambda: [("open", 3), ("close", 3)],
        description="Ordered sequence of (op_name, kernel_size) morphology operations",
    )
    merge_gap_px: int = Field(
        default=3,
        ge=0,
        description=(
            "Elliptical radius in pixels for grouping nearby defect fragments "
            "(e.g. a scratch split across periodic die streets) before noise filtering; "
            "0 disables fragment merging."
        ),
    )

    @field_validator("adaptive_block_size")
    @classmethod
    def validate_adaptive_block_size(cls, v: int) -> int:
        if v <= 1 or v % 2 == 0:
            raise ValueError(f"adaptive_block_size must be an odd integer > 1, got {v}")
        return v


def run_preprocessing(
    image: np.ndarray,
    config: Optional[PreprocessingConfig] = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Execute the configured preprocessing pipeline on an image array.

    Pure functional pipeline that does not mutate the input array.
    Tracks each step's application status, execution duration, and image statistics.

    Args:
        image: Input NumPy array (2D grayscale, 3D RGB/RGBA, or 16-bit).
        config: PreprocessingConfig instance. If None, default configuration is used.

    Returns:
        Tuple[np.ndarray, List[Dict[str, Any]]]:
            - final_image: Preprocessed 2D uint8 NumPy array.
            - step_log: Ordered list of step execution records.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(image).__name__}")

    if config is None:
        config = PreprocessingConfig()

    current_image = image.copy()
    step_log: List[Dict[str, Any]] = []

    def record_step(
        name: str,
        parameters: Dict[str, Any],
        applied: bool,
        reason: str,
        duration_ms: float,
        img: np.ndarray,
    ) -> None:
        mean_val = float(np.mean(img)) if img.size > 0 else 0.0
        std_val = float(np.std(img)) if img.size > 0 else 0.0
        step_log.append(
            {
                "step": name,
                "parameters": parameters,
                "applied": applied,
                "reason": reason,
                "duration_ms": round(duration_ms, 2),
                "mean": round(mean_val, 2),
                "std": round(std_val, 2),
            }
        )

    # 1. Grayscale Conversion
    t0 = time.perf_counter()
    if config.to_grayscale_enabled:
        is_already_gray = current_image.ndim == 2 and current_image.dtype == np.uint8
        current_image = to_grayscale(current_image)
        t_ms = (time.perf_counter() - t0) * 1000.0

        if is_already_gray:
            record_step(
                name="grayscale",
                parameters={"to_grayscale_enabled": True},
                applied=False,
                reason="Image is already single-channel 8-bit grayscale.",
                duration_ms=t_ms,
                img=current_image,
            )
        else:
            record_step(
                name="grayscale",
                parameters={"to_grayscale_enabled": True, "weights": "ITU-R 601 (0.299, 0.587, 0.114)"},
                applied=True,
                reason="Converted image to single-channel 8-bit grayscale using ITU-R 601 luma weights.",
                duration_ms=t_ms,
                img=current_image,
            )
    else:
        t_ms = (time.perf_counter() - t0) * 1000.0
        record_step(
            name="grayscale",
            parameters={"to_grayscale_enabled": False},
            applied=False,
            reason="Grayscale conversion disabled by configuration.",
            duration_ms=t_ms,
            img=current_image,
        )

    # 2. Aspect-preserving Resize
    t0 = time.perf_counter()
    if config.max_side is not None:
        orig_h, orig_w = current_image.shape[:2]
        current_image, scale = resize_preserving_aspect(current_image, max_side=config.max_side)
        t_ms = (time.perf_counter() - t0) * 1000.0

        if scale < 1.0:
            record_step(
                name="resize",
                parameters={"max_side": config.max_side, "scale_factor": scale},
                applied=True,
                reason=f"Downscaled image from ({orig_w}x{orig_h}) by factor {scale:.4f} using cv2.INTER_AREA.",
                duration_ms=t_ms,
                img=current_image,
            )
        else:
            record_step(
                name="resize",
                parameters={"max_side": config.max_side, "scale_factor": 1.0},
                applied=False,
                reason="Image dimensions already within max_side constraint; no resizing needed.",
                duration_ms=t_ms,
                img=current_image,
            )
    else:
        t_ms = (time.perf_counter() - t0) * 1000.0
        record_step(
            name="resize",
            parameters={"max_side": None},
            applied=False,
            reason="Resize step not configured (max_side is None).",
            duration_ms=t_ms,
            img=current_image,
        )

    # 3. Denoising
    t0 = time.perf_counter()
    if config.denoise_enabled:
        current_image = denoise(
            current_image,
            method=config.denoise_method,
            kernel_size=config.denoise_kernel_size,
        )
        t_ms = (time.perf_counter() - t0) * 1000.0
        record_step(
            name="denoise",
            parameters={
                "method": config.denoise_method,
                "kernel_size": config.denoise_kernel_size,
            },
            applied=True,
            reason=f"Applied {config.denoise_method} filter with kernel size {config.denoise_kernel_size}.",
            duration_ms=t_ms,
            img=current_image,
        )
    else:
        t_ms = (time.perf_counter() - t0) * 1000.0
        record_step(
            name="denoise",
            parameters={"denoise_enabled": False},
            applied=False,
            reason="Denoising disabled by configuration.",
            duration_ms=t_ms,
            img=current_image,
        )

    # 4. Contrast Enhancement
    t0 = time.perf_counter()
    if config.contrast_method != "none":
        current_image, decision = enhance_contrast(
            current_image,
            method=config.contrast_method,
            clip_limit=config.clahe_clip_limit,
            tile_grid_size=config.clahe_tile_grid_size,
            low_contrast_std_threshold=config.low_contrast_std_threshold,
            gradient_threshold=config.gradient_threshold,
        )
        t_ms = (time.perf_counter() - t0) * 1000.0
        record_step(
            name="contrast_enhancement",
            parameters={
                "method": config.contrast_method,
                "clip_limit": config.clahe_clip_limit,
                "low_contrast_std_threshold": config.low_contrast_std_threshold,
                "gradient_threshold": config.gradient_threshold,
                "measured_std": decision.get("std"),
                "measured_quadrant_delta": decision.get("quadrant_delta"),
            },
            applied=decision["applied"],
            reason=decision["reason"],
            duration_ms=t_ms,
            img=current_image,
        )
    else:
        t_ms = (time.perf_counter() - t0) * 1000.0
        record_step(
            name="contrast_enhancement",
            parameters={"contrast_method": "none"},
            applied=False,
            reason="Contrast enhancement disabled by configuration.",
            duration_ms=t_ms,
            img=current_image,
        )

    # 5. Normalization
    t0 = time.perf_counter()
    if config.normalize_enabled:
        current_image = normalize(current_image, mode=config.normalize_mode)
        t_ms = (time.perf_counter() - t0) * 1000.0
        record_step(
            name="normalize",
            parameters={"mode": config.normalize_mode},
            applied=True,
            reason=f"Normalized image intensities using min-max stretching to '{config.normalize_mode}' range.",
            duration_ms=t_ms,
            img=current_image,
        )
    else:
        t_ms = (time.perf_counter() - t0) * 1000.0
        record_step(
            name="normalize",
            parameters={"normalize_enabled": False},
            applied=False,
            reason="Normalization disabled by configuration.",
            duration_ms=t_ms,
            img=current_image,
        )

    # 6. Tiling
    t0 = time.perf_counter()
    if config.tiling_enabled:
        tiles = tile_image(current_image, tile_size=config.tile_size, overlap=config.tile_overlap)
        t_ms = (time.perf_counter() - t0) * 1000.0
        record_step(
            name="tiling",
            parameters={
                "tile_size": config.tile_size,
                "tile_overlap": config.tile_overlap,
                "num_tiles": len(tiles),
            },
            applied=True,
            reason=f"Generated {len(tiles)} overlapping tiles ({config.tile_size}x{config.tile_size}, overlap={config.tile_overlap}) covering 100% of image.",
            duration_ms=t_ms,
            img=current_image,
        )
    else:
        t_ms = (time.perf_counter() - t0) * 1000.0
        record_step(
            name="tiling",
            parameters={"tiling_enabled": False},
            applied=False,
            reason="Tiling disabled by configuration (full-frame processing).",
            duration_ms=t_ms,
            img=current_image,
        )

    return current_image, step_log


def _annotate_stage_count_notes(stage_counts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Attach an honest explanatory note whenever a pipeline stage increases candidate count."""
    annotated: List[Dict[str, Any]] = []
    prev_count: Optional[int] = None
    for item in stage_counts:
        stage = str(item["stage"])
        curr = int(item["candidate_count"])
        note = item.get("note")
        if prev_count is not None and curr > prev_count and not note:
            note = (
                f"Count increased from {prev_count} to {curr}: "
                f"{stage} split connected regions into separate components."
            )
        entry: Dict[str, Any] = {"stage": stage, "candidate_count": curr}
        if note:
            entry["note"] = note
        annotated.append(entry)
        prev_count = curr
    return annotated


def run_detection(
    test_image: np.ndarray,
    reference_image: Optional[np.ndarray] = None,
    config: Optional[DetectionConfig] = None,
) -> DetectionResult:
    """Execute the configured defect detection workflow on preprocessed wafer imagery.

    Supports Golden-Image differential inspection, Die-to-Die Self-Reference inspection
    (for periodic patterned wafers without a golden reference), and Experimental No-Reference
    Gaussian background residual inspection.

    Args:
        test_image: 2D uint8 preprocessed test wafer image.
        reference_image: Optional 2D uint8 preprocessed reference image.
        config: DetectionConfig parameters. Defaults applied if omitted.

    Returns:
        DetectionResult: Comprehensive detection artifacts, difference maps, masks, and diagnostics.
    """
    if config is None:
        config = DetectionConfig()

    warnings: List[str] = []
    test_h, test_w = test_image.shape[:2]

    # 1. Determine execution strategy
    has_reference = reference_image is not None
    is_golden = has_reference and (config.path in ("auto", "golden"))

    # 2. Segment Wafer Substrate Domain
    wafer_mask, center, radius, is_full_frame = estimate_wafer_mask(
        test_image,
        edge_margin_ratio=config.edge_margin_ratio,
    )

    alignment_info: Dict[str, Any] = {}
    die_pitch: Optional[Tuple[int, int]] = None
    peak_strength: Optional[float] = None
    coverage: Optional[float] = None
    unverifiable_fraction: Optional[float] = None
    verifiable_mask: Optional[np.ndarray] = None
    active_mask = wafer_mask
    noise_sigma_override: Optional[float] = None
    exact_diff_image: Optional[np.ndarray] = None

    if is_golden and reference_image is not None:
        path_name: Literal["golden", "self_reference", "no_reference"] = "golden"
        label = "Golden Reference Inspection"

        ref_h, ref_w = reference_image.shape[:2]
        test_aspect = float(test_w) / float(test_h)
        ref_aspect = float(ref_w) / float(ref_h)
        aspect_delta = abs(ref_aspect - test_aspect) / test_aspect

        if aspect_delta > 0.05:
            raise ReferenceIncompatibleError(
                f"Reference image aspect ratio ({ref_aspect:.3f}) differs from test image "
                f"({test_aspect:.3f}) by {aspect_delta * 100.0:.1f}%, exceeding the 5% tolerance."
            )

        # Scale reference to test image canvas if needed
        if (ref_w != test_w) or (ref_h != test_h):
            interp = cv2.INTER_AREA if (ref_w > test_w or ref_h > test_h) else cv2.INTER_LINEAR
            proc_ref = cv2.resize(reference_image, (test_w, test_h), interpolation=interp)
            warnings.append(f"Resized reference from ({ref_w}x{ref_h}) to ({test_w}x{test_h}).")
        else:
            proc_ref = reference_image.copy()

        # Phase correlation alignment
        aligned_ref, align_info = align_reference(
            test_image=test_image,
            reference_image=proc_ref,
            max_shift=config.max_alignment_shift,
            min_response=config.min_alignment_response,
        )
        alignment_info = align_info

        # Optional photometric matching
        if config.match_photometric:
            aligned_ref, photo_applied = match_photometry(test_image, aligned_ref, wafer_mask)
            alignment_info["photometric_matching_applied"] = photo_applied

        diff_image, signed_diff = compute_golden_difference(test_image, aligned_ref, wafer_mask)
        exact_diff_image = diff_image
        _, computed_sigma = estimate_noise_sigma(diff_image, wafer_mask)

    else:
        # No-reference branch: resolve mode ("auto" | "self_reference" | "background")
        effective_mode = config.no_reference_mode
        if config.path == "self_reference":
            effective_mode = "self_reference"
        elif config.path == "no_reference" and effective_mode == "auto":
            effective_mode = "background"

        if effective_mode == "background":
            path_name = "no_reference"
            label = "Experimental no-reference detection"
            alignment_info = {"applied": False, "reason": "Gaussian background no-reference mode active"}
            diff_image, signed_diff = compute_no_reference_residual(
                test_image=test_image,
                wafer_mask=wafer_mask,
                background_blur_sigma_ratio=config.background_blur_sigma_ratio,
            )
            exact_diff_image = diff_image
            _, computed_sigma = estimate_noise_sigma(diff_image, wafer_mask)
        else:
            pitch_est = estimate_die_pitch(
                image=test_image,
                wafer_mask=wafer_mask,
                min_pitch=config.self_ref_min_pitch,
                min_peak_strength=config.self_ref_min_peak_strength,
            )
            peak_strength = pitch_est.peak_strength

            if pitch_est.periodic and pitch_est.pitch_x is not None and pitch_est.pitch_y is not None:
                self_ref = build_self_reference(
                    image=test_image,
                    pitch_x=pitch_est.pitch_x,
                    pitch_y=pitch_est.pitch_y,
                    wafer_mask=wafer_mask,
                    min_valid_neighbors=config.self_ref_min_neighbors,
                    use_shift_tolerance=config.self_ref_shift_tolerance,
                )
                path_name = "self_reference"
                label = "Die-to-die self-reference detection (experimental)"
                die_pitch = (self_ref.pitch_x, self_ref.pitch_y)
                coverage = self_ref.coverage
                unverifiable_fraction = self_ref.unverifiable_fraction
                verifiable_mask = self_ref.verifiable_mask
                active_mask = self_ref.verifiable_mask
                diff_image = self_ref.difference_image
                exact_diff_image = (
                    self_ref.exact_difference_image
                    if self_ref.exact_difference_image is not None
                    else self_ref.difference_image
                )
                signed_diff = self_ref.signed_difference
                computed_sigma = self_ref.noise_sigma
                noise_sigma_override = self_ref.noise_sigma
                alignment_info = {
                    "applied": self_ref.shift_tolerance_applied,
                    "reason": (
                        f"Die-to-die self-reference (pitch={die_pitch}, "
                        f"3x3 shift tolerance={'on' if self_ref.shift_tolerance_applied else 'off'})"
                    ),
                }
            else:
                path_name = "no_reference"
                label = "Experimental no-reference detection"
                warnings.append(
                    f"Self-reference fallback: {pitch_est.reason}; fell back to Gaussian background subtraction."
                )
                alignment_info = {"applied": False, "reason": f"Fallback to background ({pitch_est.reason})"}
                diff_image, signed_diff = compute_no_reference_residual(
                    test_image=test_image,
                    wafer_mask=wafer_mask,
                    background_blur_sigma_ratio=config.background_blur_sigma_ratio,
                )
                exact_diff_image = diff_image
                _, computed_sigma = estimate_noise_sigma(diff_image, wafer_mask)

    # 3. Thresholding with Noise Floor Guard (strictly inside active_mask)
    raw_mask, thresh_val, thresh_reason = apply_threshold(
        diff_image=diff_image,
        wafer_mask=active_mask,
        method=config.threshold_method,
        fixed_threshold=config.fixed_threshold,
        adaptive_block_size=config.adaptive_block_size,
        adaptive_c=config.adaptive_c,
        min_robust_contrast=config.min_robust_contrast,
        noise_floor_k=config.noise_floor_k,
        noise_sigma_override=noise_sigma_override,
    )

    raw_candidates = count_connected_candidates(raw_mask)
    logger.info("Candidate regions: %d at stage 'raw_threshold' (threshold: %.1f)", raw_candidates, thresh_val)

    # 4. Mathematical Morphology Filtering
    morph_mask = apply_morphology(raw_mask, ops=config.morphology_ops)
    morph_candidates = count_connected_candidates(morph_mask)
    logger.info("Candidate regions: %d at stage 'after_morphology'", morph_candidates)

    stage_counts = _annotate_stage_count_notes(
        [
            {"stage": "raw_threshold", "candidate_count": raw_candidates},
            {"stage": "after_morphology", "candidate_count": morph_candidates},
        ]
    )

    return DetectionResult(
        path=path_name,
        label=label,
        threshold_method=config.threshold_method,
        threshold_value=thresh_val,
        threshold_reason=thresh_reason,
        difference_image=diff_image,
        signed_difference=signed_diff,
        wafer_mask=wafer_mask,
        wafer_center=center,
        wafer_radius=radius,
        is_full_frame=is_full_frame,
        raw_mask=raw_mask,
        morphology_mask=morph_mask,
        alignment_info=alignment_info,
        stage_counts=stage_counts,
        warnings=warnings,
        die_pitch=die_pitch,
        peak_strength=peak_strength,
        coverage=coverage,
        unverifiable_fraction=unverifiable_fraction,
        verifiable_mask=verifiable_mask,
        noise_sigma=round(float(computed_sigma), 4),
        exact_difference_image=exact_diff_image,
    )


@dataclass
class FullPipelineResult:
    """Complete end-to-end result of preprocessing, detection, segmentation, filtering, measurement, severity, and pattern analysis."""

    preprocessed_test: np.ndarray
    step_log: List[Dict[str, Any]]
    scale_factor: float
    detection: DetectionResult
    accepted_regions: List[Region]
    rejected_regions: List[RejectedRegion]
    measurements: List[RegionMeasurement]
    filtered_mask: np.ndarray
    label_map: np.ndarray
    noise_sigma: float
    derived_min_mean_contrast: float
    stage_counts: List[Dict[str, Any]]
    summary: Dict[str, Any]
    model: Dict[str, str]
    pattern: PatternResult
    processing_time_s: float
    stage_timings_ms: Dict[str, float]
    bboxes_image: np.ndarray
    contours_image: np.ndarray
    final_overlay_image: np.ndarray
    pattern_overlay_image: np.ndarray


def _extract_scale_factor(step_log: List[Dict[str, Any]]) -> float:
    """Extract resize scale_factor from preprocessing step log (defaults to 1.0)."""
    for step in step_log:
        if step.get("step") == "resize":
            scale = step.get("parameters", {}).get("scale_factor")
            if isinstance(scale, (int, float)) and scale > 0:
                return float(scale)
    return 1.0


def run_full_pipeline(
    test_image: np.ndarray,
    reference_image: Optional[np.ndarray] = None,
    prep_config: Optional[PreprocessingConfig] = None,
    det_config: Optional[DetectionConfig] = None,
    filter_config: Optional[NoiseFilterConfig] = None,
    measurement_config: Optional[MeasurementConfig] = None,
    severity_config: Optional[SeverityConfig] = None,
    pattern_config: Optional[PatternConfig] = None,
    on_stage: Optional[Callable[[str], None]] = None,
) -> FullPipelineResult:
    """Execute full classical CV inspection pipeline: preprocess -> detect -> merge -> filter -> refine -> measure -> classify -> pattern.

    Pure functional orchestration that does not mutate input arrays.

    Args:
        test_image: Raw test wafer image NumPy array.
        reference_image: Optional raw golden reference image NumPy array.
        prep_config: PreprocessingConfig (uses defaults if None).
        det_config: DetectionConfig (uses defaults if None).
        filter_config: NoiseFilterConfig (uses defaults if None).
        measurement_config: MeasurementConfig (uses defaults if None).
        severity_config: SeverityConfig (uses defaults if None).
        pattern_config: PatternConfig (uses defaults if None).

    Returns:
        FullPipelineResult with all intermediate and final artifacts, accepted/rejected
        regions, refined original-pixel measurements, severity summary, spatial pattern
        diagnosis, timings, and overlays.
    """
    t_total_start = time.perf_counter()
    stage_timings_ms: Dict[str, float] = {}

    if prep_config is None:
        prep_config = PreprocessingConfig()
    if det_config is None:
        det_config = DetectionConfig()
    if filter_config is None:
        filter_config = NoiseFilterConfig()
    if measurement_config is None:
        measurement_config = MeasurementConfig()
    if severity_config is None:
        severity_config = SeverityConfig()
    if pattern_config is None:
        pattern_config = PatternConfig()

    def _report_stage(stage_name: str) -> None:
        if on_stage is not None:
            on_stage(stage_name)

    # 1. Preprocessing (reference uses the exact same PreprocessingConfig)
    _report_stage("preprocessing")
    t0 = time.perf_counter()
    preprocessed_test, step_log = run_preprocessing(test_image, prep_config)
    scale_factor = _extract_scale_factor(step_log)

    preprocessed_ref: Optional[np.ndarray] = None
    if reference_image is not None:
        preprocessed_ref, _ = run_preprocessing(reference_image, prep_config)
    stage_timings_ms["preprocessing"] = round((time.perf_counter() - t0) * 1000.0, 2)

    # 2. Classical Detection (golden, self-reference, or experimental background)
    _report_stage("detecting")
    t0 = time.perf_counter()
    det_result = run_detection(
        test_image=preprocessed_test,
        reference_image=preprocessed_ref,
        config=det_config,
    )
    stage_timings_ms["detection"] = round((time.perf_counter() - t0) * 1000.0, 2)

    # 3. Connected-Component Segmentation & Fragment Merging (BEFORE noise filtering)
    _report_stage("segmenting")
    t0 = time.perf_counter()
    candidate_regions = segment_regions(
        morphology_mask=det_result.morphology_mask,
        signed_difference=det_result.signed_difference,
        wafer_mask=det_result.wafer_mask,
        scale_factor=scale_factor,
        merge_gap_px=det_config.merge_gap_px,
    )
    stage_timings_ms["segmentation_and_merge"] = round((time.perf_counter() - t0) * 1000.0, 2)
    logger.info(
        "Candidate regions after fragment merge (radius=%d px): %d", det_config.merge_gap_px, len(candidate_regions)
    )

    morph_count = det_result.stage_counts[-1]["candidate_count"] if det_result.stage_counts else len(candidate_regions)
    merged_count = len(candidate_regions)
    if det_config.merge_gap_px > 0 and merged_count < morph_count:
        merge_note = (
            f"Merged {morph_count - merged_count} nearby fragment(s) within "
            f"{det_config.merge_gap_px} px radius into {merged_count} candidate region(s)."
        )
    elif det_config.merge_gap_px > 0:
        merge_note = f"No fragments within {det_config.merge_gap_px} px merge radius."
    else:
        merge_note = "Fragment merging disabled (merge_gap_px=0)."

    # 4. Noise Filtering (reuses Phase 4 MAD-based noise sigma from detection stage)
    _report_stage("filtering_noise")
    t0 = time.perf_counter()
    noise_sigma = det_result.noise_sigma
    if noise_sigma <= 0.0:
        active_mask = det_result.verifiable_mask if det_result.verifiable_mask is not None else det_result.wafer_mask
        _, noise_sigma = estimate_noise_sigma(det_result.difference_image, active_mask)
    filter_result: NoiseFilterResult = filter_regions(
        regions=candidate_regions,
        noise_sigma=noise_sigma,
        config=filter_config,
    )
    stage_timings_ms["noise_filtering"] = round((time.perf_counter() - t0) * 1000.0, 2)
    logger.info("After noise filtering: %d", len(filter_result.accepted_regions))
    logger.info("Final defects: %d", len(filter_result.accepted_regions))

    # 5. Half-maximum boundary refinement (uses exact unshifted difference image)
    _report_stage("measuring")
    detected_regions = filter_result.accepted_regions
    t0 = time.perf_counter()
    if measurement_config.refine_measurements and detected_regions:
        exact_diff = (
            det_result.exact_difference_image
            if det_result.exact_difference_image is not None
            else det_result.difference_image
        )
        effective_regions = refine_regions(
            regions=detected_regions,
            exact_difference=exact_diff,
            wafer_mask=det_result.wafer_mask,
            pad_px=measurement_config.refine_pad_px,
        )
    else:
        effective_regions = detected_regions

    # Build final filtered mask and label map aligned with measured regions
    filtered_mask, label_map = build_filtered_mask_and_labels(
        image_shape=preprocessed_test.shape[:2],
        accepted_regions=effective_regions,
    )

    # 6. Build full candidate funnel stage_counts with honest increase notes
    raw_funnel: List[Dict[str, Any]] = list(det_result.stage_counts)
    raw_funnel.append(
        {
            "stage": "after_fragment_merge",
            "candidate_count": merged_count,
            "note": merge_note,
        }
    )
    raw_funnel.extend(filter_result.stage_counts)
    raw_funnel.append({"stage": "final", "candidate_count": len(effective_regions)})
    full_stage_counts = _annotate_stage_count_notes(raw_funnel)

    # Keep det_result.stage_counts synchronized with the full funnel
    det_result.stage_counts = full_stage_counts

    # 7. Measurement (original image pixels) & Prototype Severity Classification
    wafer_area_proc_px = int(np.count_nonzero(det_result.wafer_mask))
    if wafer_area_proc_px <= 0:
        wafer_area_proc_px = int(preprocessed_test.size)
    wafer_area_orig_px = float(wafer_area_proc_px) / (scale_factor * scale_factor)

    detected_by: DetectedByPath = "background" if det_result.path == "no_reference" else det_result.path
    orig_h, orig_w = test_image.shape[:2]
    measurements = measure_regions(
        regions=effective_regions,
        noise_sigma=noise_sigma,
        detected_by=detected_by,
        config=measurement_config,
        detected_regions=detected_regions,
        orig_image_size=(orig_w, orig_h),
    )
    for m in measurements:
        m.severity = classify_defect_severity(
            area_px=m.area_px,
            wafer_area_px=wafer_area_orig_px,
            config=severity_config,
        )

    inspection_severity, severity_reason = assess_inspection_severity(
        defect_areas_px=[m.area_px for m in measurements],
        defect_severities=[m.severity for m in measurements],
        wafer_area_px=wafer_area_orig_px,
        config=severity_config,
    )
    stage_timings_ms["measurement_and_severity"] = round((time.perf_counter() - t0) * 1000.0, 2)

    # 8. Wafer-level Spatial Pattern Diagnosis (does NOT modify inspection severity)
    _report_stage("diagnosing_pattern")
    t0 = time.perf_counter()
    pattern_result = analyze_spatial_pattern(
        measurements=measurements,
        wafer_center=det_result.wafer_center,
        wafer_radius=det_result.wafer_radius,
        image_size=(orig_w, orig_h),
        scale_factor=scale_factor,
        is_full_frame=det_result.is_full_frame,
        config=pattern_config,
    )
    stage_timings_ms["pattern_analysis"] = round((time.perf_counter() - t0) * 1000.0, 2)

    defect_count = len(measurements)
    total_defect_area_px = int(sum(m.area_px for m in measurements))
    largest_defect_area_px = int(max((m.area_px for m in measurements), default=0))
    average_defect_area_px = round(total_defect_area_px / float(defect_count), 2) if defect_count > 0 else 0.0
    calibration_status = format_calibration_status(measurement_config)

    pattern_summary_dict: Dict[str, Any] = {
        "type": pattern_result.type,
        "confidence": pattern_result.confidence,
        "p_value": pattern_result.p_value,
        "explanation": pattern_result.explanation,
        "metrics": pattern_result.metrics,
        "supporting_defect_ids": pattern_result.supporting_defect_ids,
        "warnings": pattern_result.warnings,
    }

    summary: Dict[str, Any] = {
        "defect_count": defect_count,
        "total_defect_area_px": total_defect_area_px,
        "largest_defect_area_px": largest_defect_area_px,
        "average_defect_area_px": average_defect_area_px,
        "severity": inspection_severity,
        "severity_reason": severity_reason,
        "calibration_status": calibration_status,
        "pixel_size_um": measurement_config.pixel_size_um,
        "pattern": pattern_summary_dict,
    }
    model_info: Dict[str, str] = {
        "name": MODEL_NAME,
        "version": MODEL_VERSION,
        "detection_method": DETECTION_METHOD_LABEL,
    }

    # 9. Render severity-styled overlays & pattern overlay on original image (all OpenCV stays inside cv/)
    t0 = time.perf_counter()
    severities_map = {m.id: m.severity for m in measurements}
    bboxes_image = draw_bounding_boxes(
        test_image,
        effective_regions,
        scale_factor=scale_factor,
        severities=severities_map,
    )
    contours_image = draw_contours(
        test_image,
        effective_regions,
        scale_factor=scale_factor,
        severities=severities_map,
    )
    final_overlay_image = draw_final_overlay(
        test_image,
        effective_regions,
        scale_factor=scale_factor,
        severities=severities_map,
    )
    pattern_overlay_image = draw_pattern_overlay(
        test_image,
        measurements=measurements,
        pattern=pattern_result,
    )
    stage_timings_ms["overlays"] = round((time.perf_counter() - t0) * 1000.0, 2)

    processing_time_s = max(round(time.perf_counter() - t_total_start, 6), 1e-6)

    return FullPipelineResult(
        preprocessed_test=preprocessed_test,
        step_log=step_log,
        scale_factor=scale_factor,
        detection=det_result,
        accepted_regions=effective_regions,
        rejected_regions=filter_result.rejected_regions,
        measurements=measurements,
        filtered_mask=filtered_mask,
        label_map=label_map,
        noise_sigma=filter_result.noise_sigma,
        derived_min_mean_contrast=filter_result.derived_min_mean_contrast,
        stage_counts=full_stage_counts,
        summary=summary,
        model=model_info,
        pattern=pattern_result,
        processing_time_s=processing_time_s,
        stage_timings_ms=stage_timings_ms,
        bboxes_image=bboxes_image,
        contours_image=contours_image,
        final_overlay_image=final_overlay_image,
        pattern_overlay_image=pattern_overlay_image,
    )
