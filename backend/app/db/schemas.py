"""Pydantic schemas and serialization models."""

from typing import Any, Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field


class ImageMetaResponse(BaseModel):
    """Metadata response for a single validated image."""

    original_filename: str
    detected_format: str
    mode: str
    width: int
    height: int
    file_size_bytes: int


class UploadInspectionResponse(BaseModel):
    """Response contract for POST /inspection/upload."""

    upload_id: str
    test_image: ImageMetaResponse
    reference_image: Optional[ImageMetaResponse] = None
    reference_warning: Optional[str] = None


class InspectionRunRequest(BaseModel):
    """Request payload for POST /inspection/run."""

    upload_id: str = Field(..., description="UUID4 identifier of uploaded inspection session")
    config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional overrides for PreprocessingConfig",
    )
    preprocessing_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional overrides for PreprocessingConfig (alias for config)",
    )
    detection_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional overrides for DetectionConfig",
    )
    noise_filter_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional overrides for NoiseFilterConfig",
    )
    measurement_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional overrides for MeasurementConfig (e.g. pixel_size_um)",
    )
    severity_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional overrides for SeverityConfig",
    )


class ImageStats(BaseModel):
    """Statistical summary of an image."""

    width: int
    height: int
    mean: float
    std: float


class StepLogItem(BaseModel):
    """Single pipeline step execution entry."""

    step: str
    parameters: Dict[str, Any]
    applied: bool
    reason: str
    duration_ms: float
    mean: float
    std: float


class StageCount(BaseModel):
    """Candidate region count at a specific inspection pipeline stage."""

    stage: str
    candidate_count: int
    note: Optional[str] = None


class AcceptedRegionResponse(BaseModel):
    """Accepted defect region metadata and original-pixel measurements returned by POST /inspection/run."""

    id: int
    bbox: Tuple[int, int, int, int]
    width_px: int = 0
    height_px: int = 0
    area_px: int
    centroid: Tuple[float, float]
    perimeter_px: float = 0.0
    equivalent_diameter_px: float = 0.0
    length_px: float = 0.0
    thickness_px: float = 0.0
    angle_deg: float = 0.0
    elongation: float = 1.0
    polarity: Literal["bright", "dark"]
    mean_contrast: float
    peak_contrast: float
    contrast_to_noise: float = 0.0
    touches_edge: bool
    detected_by: Literal["golden", "self_reference", "background"] = "golden"
    why_flagged: str = ""
    severity: Literal["LOW", "MEDIUM", "HIGH"] = "LOW"
    area_um2: Optional[float] = None
    length_um: Optional[float] = None
    thickness_um: Optional[float] = None
    reason: str = ""
    original_bbox: Optional[Tuple[int, int, int, int]] = None
    original_centroid: Optional[Tuple[float, float]] = None


class InspectionSummaryResponse(BaseModel):
    """Aggregated defect counts, area statistics, prototype severity, and calibration status."""

    defect_count: int
    total_defect_area_px: int
    largest_defect_area_px: int
    average_defect_area_px: float
    severity: Literal["NONE", "LOW", "MEDIUM", "HIGH"]
    severity_reason: str
    calibration_status: str
    pixel_size_um: Optional[float] = None


class ModelInfoResponse(BaseModel):
    """Single-source-of-truth inspection algorithm metadata."""

    name: str
    version: str
    detection_method: str


class RejectedRegionResponse(BaseModel):
    """Rejected candidate region with measured size and explicit rejection reason."""

    id: int
    area_px: int
    mean_contrast: float
    reason: str


class DetectionSummaryResponse(BaseModel):
    """Defect detection summary and candidate diagnostics."""

    path: str
    label: str
    threshold_method: str
    threshold_value: float
    threshold_reason: str
    wafer_center: Tuple[float, float]
    wafer_radius: float
    is_full_frame: bool
    alignment_info: Dict[str, Any]
    stage_counts: List[StageCount]
    warnings: List[str]
    die_pitch: Optional[Tuple[int, int]] = None
    peak_strength: Optional[float] = None
    coverage: Optional[float] = None
    unverifiable_fraction: Optional[float] = None


class InspectionRunResponse(BaseModel):
    """Extensible response contract for POST /inspection/run."""

    upload_id: str
    step_log: List[StepLogItem]
    original_stats: ImageStats
    preprocessed_stats: ImageStats
    original_image_url: str
    preprocessed_image_url: str
    difference_image_url: Optional[str] = None
    mask_image_url: Optional[str] = None
    filtered_mask_image_url: Optional[str] = None
    bboxes_image_url: Optional[str] = None
    contours_image_url: Optional[str] = None
    final_overlay_image_url: Optional[str] = None
    detection: Optional[DetectionSummaryResponse] = None
    stage_counts: List[StageCount] = Field(default_factory=list)
    regions: List[AcceptedRegionResponse] = Field(default_factory=list)
    rejected_regions: List[RejectedRegionResponse] = Field(default_factory=list)
    summary: Optional[InspectionSummaryResponse] = None
    model: Optional[ModelInfoResponse] = None
    processing_time_s: float = 0.0
    stage_timings_ms: Dict[str, float] = Field(default_factory=dict)
