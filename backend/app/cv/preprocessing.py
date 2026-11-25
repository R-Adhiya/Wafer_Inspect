"""Image preprocessing and normalization algorithms for wafer defect inspection.

Pure functions operating on NumPy arrays. No file I/O, no global state,
and never mutating input arrays.
"""

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from pydantic import BaseModel, Field, field_validator


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an image array to 8-bit single-channel grayscale.

    Why grayscale: Semiconductor wafer defect inspection operates primarily on surface
    optical reflectivity, emission intensity, and topographic contrast. Grayscale retains
    all structural intensity information required by downstream computer vision detectors
    while eliminating redundant color channels.

    Handles:
    - 2D 8-bit grayscale: returned unchanged as a copy.
    - 2D 16-bit grayscale (I;16): scaled to 8-bit by dividing by 257 (full 0-65535 to 0-255
      mapping, where 65535 -> 255 and 257 -> 1), not clipped.
    - 3D RGB (3 channels): converted using ITU-R 601 luma weights (0.299 R + 0.587 G + 0.114 B).
    - 3D RGBA (4 channels): alpha channel is ignored because transparency carries no
      semiconductor surface defect information; the first 3 color channels are converted via ITU-R 601.

    Args:
        image: Input NumPy array representing an image.

    Returns:
        np.ndarray: 2D uint8 grayscale image.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(image).__name__}")

    if image.size == 0:
        raise ValueError("Cannot convert empty array to grayscale.")

    # 1. 2D image (already single channel)
    if image.ndim == 2:
        if image.dtype == np.uint8:
            return image.copy()
        if image.dtype in (np.uint16, np.int32, np.uint32):
            # 16-bit grayscale scaled by dividing by 257 (full range: 65535 // 257 = 255)
            scaled = image.astype(np.float32) / 257.0
            return np.clip(scaled, 0, 255).astype(np.uint8)
        # Any other numeric dtype normalized
        scaled = np.clip(image.astype(np.float32), 0, 255)
        return scaled.astype(np.uint8)

    # 2. 3D image with single trailing dimension (H, W, 1)
    if image.ndim == 3 and image.shape[2] == 1:
        return to_grayscale(image[:, :, 0])

    # 3. 3D RGB or RGBA image
    if image.ndim == 3:
        num_channels = image.shape[2]
        if num_channels < 3:
            raise ValueError(f"Unsupported number of channels: {num_channels}")

        # If 16-bit color, scale down to 8-bit first
        if image.dtype in (np.uint16, np.int32, np.uint32):
            img_float = image.astype(np.float32) / 257.0
        else:
            img_float = image.astype(np.float32)

        # Alpha channel ignored (slice first 3 channels: R, G, B)
        r = img_float[:, :, 0]
        g = img_float[:, :, 1]
        b = img_float[:, :, 2]

        # Standard ITU-R 601 luma weights: Y = 0.299 R + 0.587 G + 0.114 B
        luma = 0.299 * r + 0.587 * g + 0.114 * b
        return np.clip(np.round(luma), 0, 255).astype(np.uint8)

    raise ValueError(f"Unsupported image dimensions: {image.ndim}D array")


def resize_preserving_aspect(
    image: np.ndarray,
    max_side: Optional[int],
) -> Tuple[np.ndarray, float]:
    """Resize an image preserving aspect ratio without upscaling.

    Uses cv2.INTER_AREA interpolation for area-based downscaling.

    Args:
        image: 2D or 3D NumPy array.
        max_side: Maximum allowed dimension for either width or height.
                  If None or <= 0, or if image max dimension is already <= max_side,
                  the original image is returned unchanged with scale 1.0.

    Returns:
        Tuple[np.ndarray, float]: (resized_image, scale_factor) where
        scale_factor = new_dimension / original_dimension (<= 1.0).
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(image).__name__}")

    h, w = image.shape[:2]
    if max_side is None or max_side <= 0:
        return image.copy(), 1.0

    current_max = max(h, w)
    if current_max <= max_side:
        # Never upscale
        return image.copy(), 1.0

    scale = float(max_side) / float(current_max)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def tile_image(
    image: np.ndarray,
    tile_size: int = 512,
    overlap: int = 64,
) -> List[Tuple[np.ndarray, int, int]]:
    """Partition an image into overlapping tiles covering the full canvas.

    Edge remainders are guaranteed to be covered by placing boundary-aligned
    tiles at the right and bottom edges.

    Args:
        image: 2D or 3D NumPy array.
        tile_size: Width and height of each square tile in pixels.
        overlap: Overlap between adjacent tiles in pixels. Must be < tile_size.

    Returns:
        List[Tuple[np.ndarray, int, int]]: List of (tile_data, x_offset, y_offset).
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(image).__name__}")
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError(f"overlap must be non-negative and < tile_size ({tile_size}), got {overlap}")

    h, w = image.shape[:2]

    # If the image is smaller than or equal to tile_size, return single tile
    if h <= tile_size and w <= tile_size:
        return [(image.copy(), 0, 0)]

    stride = tile_size - overlap

    def get_coordinates(total_len: int) -> List[int]:
        if total_len <= tile_size:
            return [0]
        coords = list(range(0, total_len - tile_size + 1, stride))
        # Ensure the final edge remainder is covered
        last_coord = total_len - tile_size
        if not coords or coords[-1] < last_coord:
            coords.append(last_coord)
        return coords

    x_coords = get_coordinates(w)
    y_coords = get_coordinates(h)

    tiles: List[Tuple[np.ndarray, int, int]] = []
    for y in y_coords:
        for x in x_coords:
            tile = image[y : y + tile_size, x : x + tile_size].copy()
            tiles.append((tile, x, y))

    return tiles


def verify_tiling_coverage(
    image_shape: Tuple[int, int],
    tiles: List[Tuple[np.ndarray, int, int]],
) -> bool:
    """Verify that a set of tiles covers 100% of the image pixels with no unvisited gaps.

    Args:
        image_shape: (height, width) of the original image.
        tiles: List of (tile, x, y) produced by tile_image.

    Returns:
        bool: True if every pixel is visited at least once.
    """
    h, w = image_shape[:2]
    if h == 0 or w == 0:
        return True
    coverage = np.zeros((h, w), dtype=bool)

    for tile, x, y in tiles:
        th, tw = tile.shape[:2]
        coverage[y : y + th, x : x + tw] = True

    return bool(np.all(coverage))


def denoise(
    image: np.ndarray,
    method: str = "gaussian",
    kernel_size: int = 5,
) -> np.ndarray:
    """Reduce high-frequency image noise using Gaussian or Median filtering.

    Args:
        image: 2D uint8 NumPy array.
        method: Denoising technique: "gaussian" or "median".
        kernel_size: Filter aperture size. Must be an odd positive integer.

    Returns:
        np.ndarray: Denoised image copy.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(image).__name__}")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")

    method_clean = method.lower().strip()
    if method_clean == "gaussian":
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=0)
    elif method_clean == "median":
        return cv2.medianBlur(image, kernel_size)
    else:
        raise ValueError(f"Unknown denoising method '{method}'. Allowed: 'gaussian', 'median'.")


def normalize(
    image: np.ndarray,
    mode: str = "0-255",
) -> np.ndarray:
    """Perform min-max contrast stretching on an image.

    Args:
        image: 2D NumPy array.
        mode: Target intensity scale:
              - "0-255": output dtype uint8
              - "0-1": output dtype float32

    Returns:
        np.ndarray: Normalized image array.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(image).__name__}")

    mode_clean = mode.strip().lower()
    if mode_clean not in ("0-255", "0-1"):
        raise ValueError(f"Unknown normalization mode '{mode}'. Allowed: '0-255', '0-1'.")

    min_val = float(np.min(image))
    max_val = float(np.max(image))

    # Guard against division by zero on flat images
    if max_val == min_val:
        if mode_clean == "0-255":
            return np.zeros(image.shape, dtype=np.uint8)
        else:
            return np.zeros(image.shape, dtype=np.float32)

    stretched = (image.astype(np.float32) - min_val) / (max_val - min_val)

    if mode_clean == "0-255":
        return np.clip(np.round(stretched * 255.0), 0, 255).astype(np.uint8)
    else:
        return stretched.astype(np.float32)


def enhance_contrast(
    image: np.ndarray,
    method: str = "auto",
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
    low_contrast_std_threshold: float = 25.0,
    gradient_threshold: float = 30.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Enhance image contrast using adaptive histogram equalization (CLAHE) or global equalization.

    In "auto" mode, CLAHE is applied ONLY if:
    1. Global standard deviation is below `low_contrast_std_threshold` (low overall contrast), OR
    2. The absolute difference between the brightest and darkest quadrant mean intensities
       exceeds `gradient_threshold` (strong illumination gradient).
    Otherwise, enhancement is skipped to preserve natural sensor dynamics.

    Args:
        image: 2D uint8 NumPy array.
        method: Contrast technique: "clahe", "histeq", "none", or "auto".
        clip_limit: Threshold for contrast limiting in CLAHE.
        tile_grid_size: Grid size for CLAHE local histograms.
        low_contrast_std_threshold: Standard deviation threshold below which image is low contrast.
        gradient_threshold: Quadrant mean delta threshold indicating strong illumination gradient.

    Returns:
        Tuple[np.ndarray, Dict[str, Any]]: (enhanced_image, decision_metadata).
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(image).__name__}")
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("enhance_contrast expects a 2D uint8 image.")

    method_clean = method.strip().lower()
    if method_clean not in ("clahe", "histeq", "none", "auto"):
        raise ValueError(f"Unknown contrast method '{method}'. Allowed: 'clahe', 'histeq', 'none', 'auto'.")

    h, w = image.shape
    std_val = float(np.std(image))
    mean_val = float(np.mean(image))

    # Calculate quadrant illumination statistics
    mid_y, mid_x = h // 2, w // 2
    quadrants = [
        image[:mid_y, :mid_x],
        image[:mid_y, mid_x:],
        image[mid_y:, :mid_x],
        image[mid_y:, mid_x:],
    ]
    q_means = [float(np.mean(q)) for q in quadrants if q.size > 0]
    quadrant_delta = float(max(q_means) - min(q_means)) if q_means else 0.0

    decision_info: Dict[str, Any] = {
        "method": method_clean,
        "std": round(std_val, 2),
        "mean": round(mean_val, 2),
        "quadrant_delta": round(quadrant_delta, 2),
        "applied": False,
        "reason": "",
    }

    if method_clean == "none":
        decision_info["applied"] = False
        decision_info["reason"] = "Contrast enhancement explicitly disabled."
        return image.copy(), decision_info

    if method_clean == "histeq":
        enhanced = cv2.equalizeHist(image)
        decision_info["applied"] = True
        decision_info["reason"] = "Applied global histogram equalization."
        return enhanced, decision_info

    if method_clean == "clahe":
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced = clahe.apply(image)
        decision_info["applied"] = True
        decision_info["reason"] = f"Applied CLAHE with clip_limit={clip_limit}."
        return enhanced, decision_info

    # "auto" mode evaluation
    has_low_contrast = std_val < low_contrast_std_threshold
    has_strong_gradient = quadrant_delta > gradient_threshold

    if has_low_contrast or has_strong_gradient:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced = clahe.apply(image)
        decision_info["applied"] = True

        reasons = []
        if has_low_contrast:
            reasons.append(f"low global contrast (std={std_val:.2f} < {low_contrast_std_threshold})")
        if has_strong_gradient:
            reasons.append(f"illumination gradient (quadrant delta={quadrant_delta:.2f} > {gradient_threshold})")
        decision_info["reason"] = f"Applied CLAHE: {'; '.join(reasons)}."
        return enhanced, decision_info

    # Skipped in auto mode
    decision_info["applied"] = False
    decision_info["reason"] = (
        f"Skipped CLAHE: contrast normal (std={std_val:.2f} >= {low_contrast_std_threshold}) "
        f"and uniform illumination (quadrant delta={quadrant_delta:.2f} <= {gradient_threshold})."
    )
    return image.copy(), decision_info


class PreprocessingConfig(BaseModel):
    """Configuration holding validated parameters for the preprocessing pipeline."""

    to_grayscale_enabled: bool = Field(default=True, description="Convert input imagery to 8-bit grayscale")
    max_side: Optional[int] = Field(
        default=2048,
        description=(
            "Max side dimension for aspect-preserving downscaling (default 2048 px; leaves images "
            "<= 2048 px unchanged while keeping 4096x4096 self-reference under 4 s; see docs/performance.md)"
        ),
    )
    denoise_enabled: bool = Field(default=False, description="Apply denoising filter")
    denoise_method: str = Field(default="gaussian", description="Denoise technique: 'gaussian' or 'median'")
    denoise_kernel_size: int = Field(default=5, description="Filter kernel size (must be positive odd integer)")
    contrast_method: str = Field(default="auto", description="Contrast mode: 'auto', 'clahe', 'histeq', or 'none'")
    clahe_clip_limit: float = Field(default=2.0, description="CLAHE clip limit")
    clahe_tile_grid_size: Tuple[int, int] = Field(default=(8, 8), description="CLAHE tile grid dimensions")
    low_contrast_std_threshold: float = Field(default=25.0, description="Std dev threshold for low contrast trigger")
    gradient_threshold: float = Field(default=30.0, description="Quadrant delta threshold for illumination trigger")
    normalize_enabled: bool = Field(default=False, description="Apply min-max intensity stretching")
    normalize_mode: str = Field(default="0-255", description="Normalization range: '0-255' or '0-1'")
    tiling_enabled: bool = Field(default=False, description="Partition large images into tiles")
    tile_size: int = Field(default=512, description="Tile dimension in pixels")
    tile_overlap: int = Field(default=64, description="Overlap between adjacent tiles in pixels")

    @field_validator("denoise_kernel_size")
    @classmethod
    def validate_kernel_size(cls, v: int) -> int:
        if v <= 0 or v % 2 == 0:
            raise ValueError(f"denoise_kernel_size must be a positive odd integer, got {v}")
        return v

    @field_validator("denoise_method")
    @classmethod
    def validate_denoise_method(cls, v: str) -> str:
        clean = v.strip().lower()
        if clean not in ("gaussian", "median"):
            raise ValueError(f"Unsupported denoise_method '{v}'. Allowed: 'gaussian', 'median'")
        return clean

    @field_validator("contrast_method")
    @classmethod
    def validate_contrast_method(cls, v: str) -> str:
        clean = v.strip().lower()
        if clean not in ("auto", "clahe", "histeq", "none"):
            raise ValueError(f"Unsupported contrast_method '{v}'. Allowed: 'auto', 'clahe', 'histeq', 'none'")
        return clean

    @field_validator("normalize_mode")
    @classmethod
    def validate_normalize_mode(cls, v: str) -> str:
        clean = v.strip().lower()
        if clean not in ("0-255", "0-1"):
            raise ValueError(f"Unsupported normalize_mode '{v}'. Allowed: '0-255', '0-1'")
        return clean
