"""Computer vision inspection pipeline orchestration.

Implements the multi-stage image preprocessing workflow and step logging.
"""

import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from app.cv.preprocessing import (
    PreprocessingConfig,
    denoise,
    enhance_contrast,
    normalize,
    resize_preserving_aspect,
    tile_image,
    to_grayscale,
)


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

    # Helper to append record to step_log
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

    # -------------------------------------------------------------
    # Step 1: Grayscale Conversion
    # -------------------------------------------------------------
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

    # -------------------------------------------------------------
    # Step 2: Aspect-Preserving Resize
    # -------------------------------------------------------------
    t0 = time.perf_counter()
    if config.max_side is not None and config.max_side > 0:
        h, w = current_image.shape[:2]
        if max(h, w) > config.max_side:
            current_image, scale = resize_preserving_aspect(current_image, config.max_side)
            t_ms = (time.perf_counter() - t0) * 1000.0
            record_step(
                name="resize",
                parameters={"max_side": config.max_side, "scale_factor": round(scale, 4)},
                applied=True,
                reason=f"Downscaled image from {w}x{h} to {current_image.shape[1]}x{current_image.shape[0]} (scale={scale:.4f}).",
                duration_ms=t_ms,
                img=current_image,
            )
        else:
            t_ms = (time.perf_counter() - t0) * 1000.0
            record_step(
                name="resize",
                parameters={"max_side": config.max_side},
                applied=False,
                reason=f"Image dimensions ({w}x{h}) do not exceed max_side={config.max_side}; no resizing required.",
                duration_ms=t_ms,
                img=current_image,
            )
    else:
        t_ms = (time.perf_counter() - t0) * 1000.0
        record_step(
            name="resize",
            parameters={"max_side": config.max_side},
            applied=False,
            reason="Resize step not configured (max_side is None).",
            duration_ms=t_ms,
            img=current_image,
        )

    # -------------------------------------------------------------
    # Step 3: Denoising
    # -------------------------------------------------------------
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

    # -------------------------------------------------------------
    # Step 4: Contrast Enhancement
    # -------------------------------------------------------------
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
                "tile_grid_size": list(config.clahe_tile_grid_size),
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

    # -------------------------------------------------------------
    # Step 5: Normalization
    # -------------------------------------------------------------
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

    # -------------------------------------------------------------
    # Step 6: Tiling (large images only, controlled by config)
    # -------------------------------------------------------------
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
