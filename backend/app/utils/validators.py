"""Input validation and sanitization helpers for uploaded images."""

import io
import os
import re
from dataclasses import dataclass
from typing import Optional
from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from app.core.config import settings
from app.core.logging import logger

# Prevent decompression bomb DOS attacks
Image.MAX_IMAGE_PIXELS = settings.MAX_IMAGE_PIXELS

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
EXTENSION_FORMAT_MAP = {
    ".png": "PNG",
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".tif": "TIFF",
    ".tiff": "TIFF",
}
ALLOWED_MODES = {"L", "RGB", "RGBA", "I;16", "I;16B", "I;16L", "I"}


@dataclass
class ValidatedImage:
    """Container holding validated image metadata and raw bytes."""

    original_filename: str
    sanitized_filename: str
    extension: str
    content_bytes: bytes
    detected_format: str
    mode: str
    width: int
    height: int
    file_size_bytes: int


def sanitize_filename(filename: str) -> str:
    """
    Sanitize client-supplied filename to prevent path traversal and shell injection.
    Strips directory separators and non-standard characters.
    """
    base = os.path.basename(filename.replace("\\", "/"))
    cleaned = re.sub(r"[^\w\s\.-]", "_", base)
    cleaned = cleaned.strip(". ")
    return cleaned if cleaned else "unnamed_image"


async def validate_uploaded_image(
    file: Optional[UploadFile],
    field_name: str = "image",
) -> ValidatedImage:
    """
    Validate an uploaded image according to the strict 7-step criteria:
      1. File present and not empty -> 400
      2. Extension in allowlist (.png, .jpg, .jpeg, .tif, .tiff) -> 415
      3. Size <= MAX_UPLOAD_SIZE, enforced in streaming chunks -> 413
      4. Real format detected by Pillow and matches extension -> 415
      5. Decode integrity: Image.verify() then reopen and fully load -> 422
      6. Dimensions within limits (default 64x64 to 8192x8192) -> 422
      7. Mode compatibility (L, RGB, RGBA, I;16) -> 422
    """
    # 1a. File presence check
    if file is None or not file.filename:
        logger.warning("Validation failed for '%s': missing file or filename", field_name)
        raise HTTPException(
            status_code=400,
            detail={"code": "FILE_MISSING", "message": f"'{field_name}' is missing or has no filename."},
        )

    raw_filename = file.filename
    clean_name = sanitize_filename(raw_filename)

    # Peek the first chunk to verify non-empty before proceeding
    first_chunk = await file.read(64 * 1024)
    if not first_chunk or len(first_chunk) == 0:
        logger.warning("Validation failed for '%s' ('%s'): empty file (0 bytes)", field_name, clean_name)
        raise HTTPException(
            status_code=400,
            detail={"code": "FILE_EMPTY", "message": f"Uploaded file '{clean_name}' is empty (0 bytes)."},
        )

    # 2. Extension check
    dot_idx = clean_name.rfind(".")
    ext = clean_name[dot_idx:].lower() if dot_idx != -1 else ""
    if ext not in ALLOWED_EXTENSIONS:
        logger.warning(
            "Validation failed for '%s' ('%s'): unsupported extension '%s'",
            field_name,
            clean_name,
            ext,
        )
        raise HTTPException(
            status_code=415,
            detail={
                "code": "UNSUPPORTED_EXTENSION",
                "message": f"File extension '{ext}' is not supported. Allowed extensions: .png, .jpg, .jpeg, .tif, .tiff.",
            },
        )

    # 3. Size check enforced while reading in chunks
    total_bytes = len(first_chunk)
    if total_bytes > settings.MAX_UPLOAD_SIZE:
        logger.warning(
            "Validation failed for '%s' ('%s'): initial chunk %d exceeds %d bytes",
            field_name,
            clean_name,
            total_bytes,
            settings.MAX_UPLOAD_SIZE,
        )
        raise HTTPException(
            status_code=413,
            detail={
                "code": "FILE_TOO_LARGE",
                "message": f"File size exceeds maximum allowed limit of {settings.MAX_UPLOAD_SIZE} bytes.",
            },
        )

    content_buffer = io.BytesIO()
    content_buffer.write(first_chunk)

    chunk_size = 64 * 1024
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total_bytes += len(chunk)
        if total_bytes > settings.MAX_UPLOAD_SIZE:
            logger.warning(
                "Validation failed for '%s' ('%s'): size %d exceeds limit of %d bytes",
                field_name,
                clean_name,
                total_bytes,
                settings.MAX_UPLOAD_SIZE,
            )
            raise HTTPException(
                status_code=413,
                detail={
                    "code": "FILE_TOO_LARGE",
                    "message": f"File size exceeds maximum allowed limit of {settings.MAX_UPLOAD_SIZE} bytes.",
                },
            )
        content_buffer.write(chunk)

    raw_bytes = content_buffer.getvalue()

    # 4. Real format detection with Pillow and consistency with extension
    try:
        img_inspect = Image.open(io.BytesIO(raw_bytes))
        detected_format = img_inspect.format
    except (UnidentifiedImageError, Exception) as exc:
        logger.warning(
            "Validation failed for '%s' ('%s'): Pillow format identification failed: %s",
            field_name,
            clean_name,
            str(exc),
        )
        raise HTTPException(
            status_code=415,
            detail={
                "code": "INVALID_FORMAT",
                "message": f"File content of '{clean_name}' is not a valid or identifiable PNG, JPEG, or TIFF image.",
            },
        )

    if detected_format not in {"PNG", "JPEG", "TIFF"}:
        logger.warning(
            "Validation failed for '%s' ('%s'): detected format '%s' not allowed",
            field_name,
            clean_name,
            detected_format,
        )
        raise HTTPException(
            status_code=415,
            detail={
                "code": "UNSUPPORTED_FORMAT",
                "message": f"Detected image format '{detected_format}' is not supported. Allowed formats: PNG, JPEG, TIFF.",
            },
        )

    expected_format = EXTENSION_FORMAT_MAP.get(ext)
    if detected_format != expected_format:
        logger.warning(
            "Validation failed for '%s' ('%s'): format mismatch, detected '%s' but extension is '%s'",
            field_name,
            clean_name,
            detected_format,
            ext,
        )
        raise HTTPException(
            status_code=415,
            detail={
                "code": "FORMAT_MISMATCH",
                "message": f"Detected format '{detected_format}' does not match file extension '{ext}'.",
            },
        )

    # 5. Decode integrity: Image.verify() then reopen and fully load raster data
    try:
        img_inspect.verify()
        # Re-open after verify() because verify() exhausts image data
        img_full = Image.open(io.BytesIO(raw_bytes))
        img_full.load()
    except (SyntaxError, OSError, IOError, Exception) as exc:
        logger.warning(
            "Validation failed for '%s' ('%s'): decode integrity failed: %s",
            field_name,
            clean_name,
            str(exc),
        )
        raise HTTPException(
            status_code=422,
            detail={
                "code": "CORRUPT_IMAGE",
                "message": f"Image decode integrity verification failed: file is corrupt or truncated ({str(exc)}).",
            },
        )

    # 6. Dimensions limits
    width, height = img_full.size
    if width < settings.MIN_IMAGE_DIMENSION or height < settings.MIN_IMAGE_DIMENSION:
        logger.warning(
            "Validation failed for '%s' ('%s'): dimensions %dx%d smaller than minimum %dx%d",
            field_name,
            clean_name,
            width,
            height,
            settings.MIN_IMAGE_DIMENSION,
            settings.MIN_IMAGE_DIMENSION,
        )
        raise HTTPException(
            status_code=422,
            detail={
                "code": "DIMENSIONS_TOO_SMALL",
                "message": f"Image dimensions ({width}x{height}) are smaller than minimum allowed ({settings.MIN_IMAGE_DIMENSION}x{settings.MIN_IMAGE_DIMENSION}).",
            },
        )

    if width > settings.MAX_IMAGE_DIMENSION or height > settings.MAX_IMAGE_DIMENSION:
        logger.warning(
            "Validation failed for '%s' ('%s'): dimensions %dx%d exceed maximum %dx%d",
            field_name,
            clean_name,
            width,
            height,
            settings.MAX_IMAGE_DIMENSION,
            settings.MAX_IMAGE_DIMENSION,
        )
        raise HTTPException(
            status_code=422,
            detail={
                "code": "DIMENSIONS_TOO_LARGE",
                "message": f"Image dimensions ({width}x{height}) exceed maximum allowed ({settings.MAX_IMAGE_DIMENSION}x{settings.MAX_IMAGE_DIMENSION}).",
            },
        )

    # 7. Mode compatibility
    mode = img_full.mode
    if mode not in ALLOWED_MODES:
        logger.warning(
            "Validation failed for '%s' ('%s'): mode '%s' not supported",
            field_name,
            clean_name,
            mode,
        )
        raise HTTPException(
            status_code=422,
            detail={
                "code": "UNSUPPORTED_MODE",
                "message": f"Image color mode '{mode}' is not supported. Supported modes: L, RGB, RGBA, I;16.",
            },
        )

    logger.info(
        "Validation successful for '%s' ('%s'): format=%s, mode=%s, %dx%d, %d bytes",
        field_name,
        clean_name,
        detected_format,
        mode,
        width,
        height,
        len(raw_bytes),
    )

    return ValidatedImage(
        original_filename=clean_name,
        sanitized_filename=clean_name,
        extension=ext,
        content_bytes=raw_bytes,
        detected_format=detected_format,
        mode=mode,
        width=width,
        height=height,
        file_size_bytes=len(raw_bytes),
    )
