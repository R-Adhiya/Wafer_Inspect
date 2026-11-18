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
    """Accepted defect region metadata returned by POST /inspection/run."""

    id: int
    bbox: Tuple[int, int, int, int]
    area_px: int
    centroid: Tuple[float, float]
    polarity: Literal["bright", "dark"]
    mean_contrast: float
    peak_contrast: float
    touches_edge: bool
    reason: str = ""
    original_bbox: Optional[Tuple[int, int, int, int]] = None
    original_centroid: Optional[Tuple[float, float]] = None


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
    detection: Optional[DetectionSummaryResponse] = None
    stage_counts: List[StageCount] = Field(default_factory=list)
    regions: List[AcceptedRegionResponse] = Field(default_factory=list)
    rejected_regions: List[RejectedRegionResponse] = Field(default_factory=list)
