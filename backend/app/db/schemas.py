"""Pydantic schemas and serialization models."""

from typing import Any, Dict, List, Optional
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


class InspectionRunResponse(BaseModel):
    """Extensible response contract for POST /inspection/run."""

    upload_id: str
    step_log: List[StepLogItem]
    original_stats: ImageStats
    preprocessed_stats: ImageStats
    original_image_url: str
    preprocessed_image_url: str
