"""Synthetic wafer image generator for testing and validation.

Generates deterministic, synthetic grayscale images simulating silicon wafer
imagery with circular geometry, die patterns, sensor noise, illumination gradients,
and ground-truth synthetic defects (bright blobs, dark blobs, scratches).

All data produced by this module is purely synthetic and intended solely
for computer vision inspection and algorithm accuracy validation.
"""

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np


def generate_synthetic_wafer(
    size: int = 512,
    seed: Optional[int] = 42,
    wafer_radius_ratio: float = 0.44,
    substrate_intensity: float = 140.0,
    background_intensity: float = 20.0,
    grid_pitch: int = 32,
    grid_line_width: int = 2,
    grid_intensity: float = 90.0,
    noise_std: float = 5.0,
    has_gradient: bool = False,
    gradient_strength: float = 35.0,
    defects: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Generate a deterministic synthetic wafer image with ground-truth defect annotations.

    Args:
        size: Width and height of the square synthetic canvas in pixels.
        seed: Random seed for numpy default_rng to ensure reproducibility.
        wafer_radius_ratio: Ratio of wafer radius to canvas dimension.
        substrate_intensity: Nominal pixel intensity for clean wafer silicon substrate.
        background_intensity: Nominal pixel intensity for outer background chuck.
        grid_pitch: Pitch in pixels between fabricated die grid lines.
        grid_line_width: Width in pixels of die streets/grid lines.
        grid_intensity: Pixel intensity along die grid streets.
        noise_std: Standard deviation of additive zero-mean Gaussian sensor noise.
        has_gradient: Whether to synthesize an uneven spatial illumination gradient.
        gradient_strength: Peak-to-peak amplitude of the illumination gradient.
        defects: Optional list of synthetic defect specifications. Each item is a dict:
                 - "type": "bright_blob" | "dark_blob" | "scratch"
                 - "center": (x, y) in canvas coordinates
                 - "radius": float (for blobs) or "size": float
                 - "contrast": float (positive intensity delta for bright, negative for dark)
                 - for "scratch": "start": (x1, y1), "end": (x2, y2), "width": int

    Returns:
        Tuple[np.ndarray, List[Dict[str, Any]]]:
            - image_uint8: 2D uint8 NumPy array of the synthetic wafer.
            - ground_truth: List of ground-truth defect metadata dicts with verified locations.
    """
    rng = np.random.default_rng(seed)

    # 1. Coordinate grids
    y, x = np.ogrid[:size, :size]
    center_y, center_x = size / 2.0, size / 2.0
    radius = size * wafer_radius_ratio
    wafer_mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2

    # 2. Base canvas (dark background and wafer substrate)
    canvas = np.full((size, size), background_intensity, dtype=np.float32)
    canvas[wafer_mask] = substrate_intensity

    # 3. Fabricated die grid texture
    if grid_pitch > 0 and grid_line_width > 0:
        grid_mask = ((x % grid_pitch < grid_line_width) | (y % grid_pitch < grid_line_width)) & wafer_mask
        canvas[grid_mask] = grid_intensity

    # 4. Optional spatial illumination gradient across wafer
    if has_gradient and gradient_strength > 0:
        # Diagonal linear gradient across wafer: bright at top-left, darker at bottom-right
        grad = ((size - x) + (size - y)) / (2.0 * size) * (gradient_strength * 2.0) - gradient_strength
        canvas[wafer_mask] += grad[wafer_mask]

    # 5. Additive Gaussian noise
    if noise_std > 0:
        noise = rng.normal(0.0, noise_std, (size, size)).astype(np.float32)
        canvas += noise

    # 6. Inject synthetic defects
    ground_truth: List[Dict[str, Any]] = []

    if defects:
        for defect in defects:
            d_type = defect.get("type", "bright_blob")
            contrast = float(defect.get("contrast", 40.0))

            if d_type in ("bright_blob", "dark_blob"):
                cx, cy = defect.get("center", (size / 2.0, size / 2.0))
                r = float(defect.get("radius", defect.get("size", 8.0) / 2.0))
                dist_sq = (x - cx) ** 2 + (y - cy) ** 2
                blob_mask = (dist_sq <= r ** 2) & wafer_mask

                if d_type == "bright_blob":
                    canvas[blob_mask] += abs(contrast)
                    contrast_val = abs(contrast)
                else:
                    canvas[blob_mask] -= abs(contrast)
                    contrast_val = -abs(contrast)

                ground_truth.append({
                    "type": d_type,
                    "center": (float(cx), float(cy)),
                    "radius": float(r),
                    "size": float(2.0 * r),
                    "contrast": contrast_val,
                })

            elif d_type == "scratch":
                start = defect.get("start", (int(center_x - 30), int(center_y - 30)))
                end = defect.get("end", (int(center_x + 30), int(center_y + 30)))
                line_width = int(defect.get("width", 2))

                scratch_layer = np.zeros((size, size), dtype=np.uint8)
                cv2.line(
                    scratch_layer,
                    (int(start[0]), int(start[1])),
                    (int(end[0]), int(end[1])),
                    color=255,
                    thickness=line_width,
                )
                scratch_mask = (scratch_layer > 0) & wafer_mask
                canvas[scratch_mask] += contrast

                mid_x = (start[0] + end[0]) / 2.0
                mid_y = (start[1] + end[1]) / 2.0
                length = float(np.hypot(end[0] - start[0], end[1] - start[1]))

                ground_truth.append({
                    "type": "scratch",
                    "center": (float(mid_x), float(mid_y)),
                    "start": (float(start[0]), float(start[1])),
                    "end": (float(end[0]), float(end[1])),
                    "size": length,
                    "radius": length / 2.0,
                    "width": line_width,
                    "contrast": contrast,
                })

    # 7. Clip to valid 8-bit intensity range and cast to uint8
    image_uint8 = np.clip(canvas, 0, 255).astype(np.uint8)
    return image_uint8, ground_truth


def create_standard_synthetic_pair(
    size: int = 512,
    seed: int = 42,
    shift: Tuple[float, float] = (0.0, 0.0),
    defects: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Create a matching synthetic golden reference and test image pair.

    The golden reference has no defects and is clean.
    The test image contains the injected defects and optional translational shift.

    Args:
        size: Dimensions of square image.
        seed: Random seed for deterministic reproducibility.
        shift: (dx, dy) translation shift to apply to the test image.
        defects: List of synthetic defects injected into the test image.

    Returns:
        Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
            (reference_image, test_image, ground_truth_defects)
    """
    # Clean reference image
    reference_image, _ = generate_synthetic_wafer(
        size=size,
        seed=seed,
        has_gradient=False,
        defects=None,
    )

    # Test image with defects
    test_image, ground_truth = generate_synthetic_wafer(
        size=size,
        seed=seed + 1,  # different noise realization
        has_gradient=False,
        defects=defects,
    )

    # Apply translation if requested
    dx, dy = shift
    if abs(dx) > 1e-4 or abs(dy) > 1e-4:
        M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        test_image = cv2.warpAffine(
            test_image,
            M,
            (size, size),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=20,
        )
        # Shift ground-truth coordinates
        shifted_gt = []
        for d in ground_truth:
            d_shifted = d.copy()
            cx, cy = d["center"]
            d_shifted["center"] = (cx + dx, cy + dy)
            if "start" in d:
                sx, sy = d["start"]
                ex, ey = d["end"]
                d_shifted["start"] = (sx + dx, sy + dy)
                d_shifted["end"] = (ex + dx, ey + dy)
            shifted_gt.append(d_shifted)
        ground_truth = shifted_gt

    return reference_image, test_image, ground_truth
