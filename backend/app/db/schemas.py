"""Pydantic schemas and serialization models."""

from typing import Optional
from pydantic import BaseModel


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
