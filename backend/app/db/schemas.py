"""Pydantic schemas and serialization models."""

from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, model_validator


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

    inspection_id: int
    upload_id: str
    test_image: ImageMetaResponse
    reference_image: Optional[ImageMetaResponse] = None
    reference_warning: Optional[str] = None
    disclaimer: Optional[str] = None


class InspectionRunRequest(BaseModel):
    """Request payload for POST /inspection/run (accepts inspection_id or legacy upload_id)."""

    inspection_id: Optional[int] = Field(
        default=None,
        description="Integer database ID of uploaded inspection session",
    )
    upload_id: Optional[str] = Field(
        default=None,
        description="UUID4 identifier of uploaded inspection session (compatibility alias)",
    )
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
    pattern_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional overrides for PatternConfig",
    )

    @model_validator(mode="after")
    def _require_identifier(self) -> "InspectionRunRequest":
        if self.inspection_id is None and not self.upload_id:
            raise ValueError("Either 'inspection_id' or 'upload_id' must be provided.")
        return self


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
    detected_area_px: int = 0
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
    contour: List[Tuple[int, int]] = Field(default_factory=list)
    reason: str = ""
    original_bbox: Optional[Tuple[int, int, int, int]] = None
    original_centroid: Optional[Tuple[float, float]] = None


class PatternResultResponse(BaseModel):
    """Wafer-level spatial pattern diagnosis with Monte Carlo significance metrics."""

    type: Literal[
        "INSUFFICIENT_DATA",
        "EDGE_RING",
        "CENTER_CLUSTER",
        "DONUT",
        "LOCAL_CLUSTER",
        "SCRATCH_LINE",
        "RANDOM_SCATTER",
    ]
    confidence: float
    p_value: Optional[float] = None
    explanation: str
    metrics: Dict[str, Any] = Field(default_factory=dict)
    supporting_defect_ids: List[int] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class InspectionSummaryResponse(BaseModel):
    """Aggregated defect counts, area statistics, prototype severity, calibration, and pattern diagnosis."""

    defect_count: int
    total_defect_area_px: int
    largest_defect_area_px: int
    average_defect_area_px: float
    severity: Literal["NONE", "LOW", "MEDIUM", "HIGH"]
    severity_reason: str
    calibration_status: str
    pixel_size_um: Optional[float] = None
    pattern: Optional[PatternResultResponse] = None


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
    """Extensible response contract for POST /inspection/run and GET /inspection/{id}."""

    inspection_id: Optional[int] = None
    upload_id: str
    status: str = "completed"
    created_at: Optional[str] = None
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
    pattern_overlay_image_url: Optional[str] = None
    detection: Optional[DetectionSummaryResponse] = None
    stage_counts: List[StageCount] = Field(default_factory=list)
    regions: List[AcceptedRegionResponse] = Field(default_factory=list)
    rejected_regions: List[RejectedRegionResponse] = Field(default_factory=list)
    summary: Optional[InspectionSummaryResponse] = None
    model: Optional[ModelInfoResponse] = None
    pattern: Optional[PatternResultResponse] = None
    processing_time_s: float = 0.0
    stage_timings_ms: Dict[str, float] = Field(default_factory=dict)
    test_filename: Optional[str] = None
    test_mode: Optional[str] = None
    reference_used: bool = False
    reference_filename: Optional[str] = None


class InspectionStatusResponse(BaseModel):
    """Live single-process in-memory stage status for GET /inspection/{id}/status."""

    inspection_id: Optional[int] = None
    upload_id: str
    status: Literal["not_started", "pending", "running", "completed", "failed"]
    current_stage: str
    completed_stages: List[str] = Field(default_factory=list)
    stage_history: List[str] = Field(default_factory=list)
    stages: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    updated_at: Optional[str] = None


class HistoryItemResponse(BaseModel):
    """Summary row returned by GET /inspection/history and recent dashboard list."""

    id: int
    upload_id: str
    status: str
    original_filename: str
    image_width: int
    image_height: int
    has_reference: bool
    defect_count: int
    total_defect_area: int
    largest_defect_area: int
    severity: Optional[str] = None
    pattern_type: Optional[str] = None
    pattern_confidence: Optional[float] = None
    detection_path: Optional[str] = None
    detection_label: Optional[str] = None
    detection_method: Optional[str] = None
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    processing_time: Optional[float] = None
    created_at: str


class HistoryListResponse(BaseModel):
    """Paginated response contract for GET /inspection/history."""

    items: List[HistoryItemResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class InspectionImagesResponse(BaseModel):
    """Map of available image artifact kinds to URLs for GET /inspection/{id}/images."""

    inspection_id: int
    upload_id: str
    images: Dict[str, str]


class DeleteInspectionResponse(BaseModel):
    """Response contract for DELETE /inspection/{id}."""

    deleted: bool
    inspection_id: int
    upload_id: str


class RecentDefectCountPoint(BaseModel):
    inspection_id: int
    filename: str
    defect_count: int
    severity: str
    created_at: Optional[str] = None


class SeverityCountPoint(BaseModel):
    severity: str
    count: int


class AreaHistogramPoint(BaseModel):
    bin: str
    count: int


class PatternCountPoint(BaseModel):
    pattern_type: str
    count: int


class DashboardStatsResponse(BaseModel):
    """SQL-aggregated KPIs and chart series for GET /dashboard/stats."""

    total_inspections: int
    total_defects: int
    high_severity_inspections: int
    average_defect_area: float
    recent_inspections: List[HistoryItemResponse]
    recent_defect_counts: List[RecentDefectCountPoint]
    severity_distribution: List[SeverityCountPoint]
    defect_area_histogram: List[AreaHistogramPoint]
    pattern_distribution: List[PatternCountPoint]


class SampleItemResponse(BaseModel):
    """Metadata for a single synthetic demonstration wafer sample."""

    name: str
    title: str
    description: str
    category: str
    has_reference: bool
    image_size: int
    disclaimer: str = "Synthetic demonstration image"


class SampleListResponse(BaseModel):
    """Response contract for GET /samples."""

    disclaimer: str = "Synthetic demonstration image"
    samples: List[SampleItemResponse]


class NoiseExperimentCaseResponse(BaseModel):
    """Candidate metrics at a single stage of the four-stage noise filtering process."""

    stage_id: int
    stage_name: str
    label: str
    candidate_count: int
    tp: Optional[int] = None
    fp: Optional[int] = None
    fn: Optional[int] = None
    precision: Optional[float] = None
    recall: Optional[float] = None


class InspectionNoiseExperimentResponse(BaseModel):
    """Four-stage noise filtering evaluation for an individual inspection run."""

    inspection_id: int
    upload_id: str
    has_ground_truth: bool
    ground_truth_count: Optional[int] = None
    tolerance_px: float = 15.0
    cases: List[NoiseExperimentCaseResponse]
    summary: str
    disclaimer: Optional[str] = None


class NoiseSweepCaseSummaryResponse(BaseModel):
    """Aggregated statistics across random seeds for a single filter stage at a given noise level."""

    stage_id: int
    stage_name: str
    label: str
    mean_count: float
    std_count: float
    mean_precision: float
    std_precision: float
    mean_recall: float
    std_recall: float
    mean_tp: float
    mean_fp: float
    mean_fn: float


class NoiseSweepPointResponse(BaseModel):
    """Sweep evaluation point for a specific Gaussian noise sigma."""

    noise_level: float
    seed_count: int
    cases: List[NoiseSweepCaseSummaryResponse]
    summary: Optional[str] = None


class NoiseSweepRequest(BaseModel):
    """Request payload for synchronous POST /experiments/noise-sweep."""

    layout: str = Field(default="edge_ring", description="Defect layout pattern")
    path: Literal["golden", "self_reference"] = Field(default="self_reference", description="Detection path")
    noise_levels: Optional[List[float]] = Field(default=None, description="Gaussian noise sigma levels (<= 8)")
    seeds: Optional[List[int]] = Field(default=None, description="Random seeds per level (<= 10)")
    size: int = Field(default=512, description="Wafer image dimension (<= 1024)")
    tolerance_px: float = Field(default=15.0, description="Centroid match tolerance")
    defect_count: int = Field(default=8, description="Target defect count")

    @model_validator(mode="after")
    def _validate_caps(self) -> "NoiseSweepRequest":
        levels = self.noise_levels if self.noise_levels is not None else [2.0, 5.0, 10.0, 15.0, 20.0]
        seed_list = self.seeds if self.seeds is not None else [42, 43, 44, 45, 46]
        if len(levels) > 8:
            raise ValueError(f"Too many noise levels: requested {len(levels)}, maximum allowed is 8.")
        if len(seed_list) > 10:
            raise ValueError(f"Too many seeds: requested {len(seed_list)}, maximum allowed is 10.")
        if self.size > 1024:
            raise ValueError(f"Image dimension ({self.size}) exceeds maximum allowed sweep size (1024).")
        if self.size < 64:
            raise ValueError(f"Image dimension ({self.size}) must be at least 64.")
        return self


class NoiseSweepResponse(BaseModel):
    """Response contract for POST /experiments/noise-sweep."""

    layout: str
    path: str
    size: int
    noise_levels: List[float]
    seeds: List[int]
    true_defect_count: int
    series: List[NoiseSweepPointResponse]
    summary: str
    disclaimer: str = "Synthetic demonstration data"
