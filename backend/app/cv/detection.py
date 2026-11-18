"""Defect detection algorithms for wafer inspection.

Provides pure functions for classical computer vision wafer defect detection:
1. Wafer mask segmentation & geometry extraction (Otsu + convex enclosing circle).
2. Golden-image differential inspection (phaseCorrelate alignment + absdiff/signed diff).
3. Experimental no-reference residual inspection (multiscale Gaussian background subtraction).
4. Robust contrast-guarded thresholding (Otsu, Fixed, Adaptive).
5. Mathematical morphology filtering (opening, closing, erosion, dilation) with elliptical kernels.
6. Connected component analysis and candidate region tracking across pipeline stages.
"""

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Literal, Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger("waferinspect.cv.detection")


@dataclass
class DetectionResult:
    """Encapsulates the complete defect detection output and candidate diagnostics."""

    path: Literal["golden", "no_reference"]
    label: str
    threshold_method: str
    threshold_value: float
    threshold_reason: str
    difference_image: np.ndarray  # 2D uint8
    signed_difference: np.ndarray  # 2D int16
    wafer_mask: np.ndarray  # 2D uint8 (0 or 255)
    wafer_center: Tuple[float, float]
    wafer_radius: float
    is_full_frame: bool
    raw_mask: np.ndarray  # 2D uint8 (0 or 255)
    morphology_mask: np.ndarray  # 2D uint8 (0 or 255)
    alignment_info: Dict[str, Any] = field(default_factory=dict)
    stage_counts: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def estimate_wafer_mask(
    image: np.ndarray,
    edge_margin_ratio: float = 0.02,
) -> Tuple[np.ndarray, Tuple[float, float], float, bool]:
    """Segment the circular wafer substrate and extract geometric disk properties.

    Uses Otsu thresholding on a slightly smoothed image to separate the bright wafer
    substrate from the dark background chuck. Fits a minimum enclosing circle to the
    largest connected component and pulls the boundary inward by `edge_margin_ratio`
    to suppress false alarms along the bevel edge.

    If the segmented foreground covers more than 95% of the frame, the image is treated
    as full-frame (magnified die view or edge-less sample).

    Args:
        image: 2D uint8 grayscale image.
        edge_margin_ratio: Inward radial margin ratio (default 0.02, 2% of radius)
                           to guard against wafer edge bevel artifacts.

    Returns:
        Tuple containing:
            - mask: 2D uint8 array (255 inside wafer disk, 0 outside).
            - center: (cx, cy) disk center coordinates.
            - radius: Outer wafer radius in pixels.
            - is_full_frame: True if the wafer occupies > 95% of the frame.
    """
    if not isinstance(image, np.ndarray) or image.ndim != 2 or image.dtype != np.uint8:
        raise TypeError("estimate_wafer_mask requires a 2D uint8 NumPy array.")

    h, w = image.shape
    blurred = cv2.GaussianBlur(image, (7, 7), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    fg_pixels = int(np.count_nonzero(binary))
    total_pixels = binary.size

    # Full-frame guard (> 95% foreground)
    if total_pixels > 0 and (fg_pixels / total_pixels) > 0.95:
        center = (float(w / 2.0), float(h / 2.0))
        radius = float(min(w, h) / 2.0)
        mask = np.full((h, w), 255, dtype=np.uint8)
        logger.info("Wafer mask: image is full-frame (> 95%% foreground, radius=%.1f)", radius)
        return mask, center, radius, True

    # Find largest connected foreground contour
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        center = (float(w / 2.0), float(h / 2.0))
        radius = float(min(w, h) / 2.0)
        mask = np.full((h, w), 255, dtype=np.uint8)
        return mask, center, radius, True

    largest_cnt = max(contours, key=cv2.contourArea)
    (cx, cy), r = cv2.minEnclosingCircle(largest_cnt)

    # Apply radial edge margin to avoid false positive edge triggers
    effective_r = max(1.0, r * (1.0 - edge_margin_ratio))

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (int(round(cx)), int(round(cy))), int(round(effective_r)), 255, thickness=-1)

    logger.info("Wafer mask: estimated center=(%.1f, %.1f), radius=%.1f (effective=%.1f)", cx, cy, r, effective_r)
    return mask, (float(cx), float(cy)), float(r), False


def align_reference(
    test_image: np.ndarray,
    reference_image: np.ndarray,
    max_shift: float = 10.0,
    min_response: float = 0.05,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Estimate and compensate subpixel translational misregistration using Phase Correlation.

    Computes cross-power spectrum between the Hanning-windowed reference and test images.
    Applies translation only when the correlation response meets `min_response` and the
    shift magnitude is within `max_shift` pixels.

    Args:
        test_image: 2D uint8 preprocessed inspection image.
        reference_image: 2D uint8 preprocessed reference image.
        max_shift: Maximum allowable translation magnitude in pixels (default 10 px).
        min_response: Minimum phase correlation peak strength threshold.

    Returns:
        Tuple containing:
            - aligned_reference: Transformed reference image matching test coordinates.
            - alignment_info: Metadata with shift_x, shift_y, response, and applied flag.
    """
    h, w = test_image.shape
    hann = cv2.createHanningWindow((w, h), cv2.CV_32F)
    shift, response = cv2.phaseCorrelate(
        reference_image.astype(np.float32),
        test_image.astype(np.float32),
        hann,
    )

    dx, dy = float(shift[0]), float(shift[1])
    shift_magnitude = float(np.hypot(dx, dy))
    should_apply = bool((response >= min_response) and (shift_magnitude <= max_shift))

    alignment_info = {
        "shift_x": round(dx, 3),
        "shift_y": round(dy, 3),
        "shift_magnitude": round(shift_magnitude, 3),
        "response": round(float(response), 4),
        "applied": should_apply,
    }

    if should_apply and (abs(dx) > 1e-4 or abs(dy) > 1e-4):
        # Build affine translation matrix
        m = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        aligned_reference = cv2.warpAffine(
            reference_image,
            m,
            (w, h),
            borderMode=cv2.BORDER_REPLICATE,
        )
        logger.info(
            "Alignment applied: shift=(%.2f, %.2f) px, magnitude=%.2f px, response=%.4f",
            dx, dy, shift_magnitude, response,
        )
    else:
        aligned_reference = reference_image.copy()
        logger.info(
            "Alignment skipped: shift=(%.2f, %.2f) px, magnitude=%.2f px, response=%.4f",
            dx, dy, shift_magnitude, response,
        )

    return aligned_reference, alignment_info


def match_photometry(
    test_image: np.ndarray,
    reference_image: np.ndarray,
    wafer_mask: np.ndarray,
) -> Tuple[np.ndarray, bool]:
    """Photometrically adjust reference mean and standard deviation to match the test image.

    Normalizes global illumination discrepancies inside the wafer area to prevent
    spurious whole-wafer offset differentials.

    Args:
        test_image: 2D uint8 test image.
        reference_image: 2D uint8 reference image.
        wafer_mask: 2D uint8 mask denoting wafer domain.

    Returns:
        Tuple of (photometrically_matched_reference, applied_flag).
    """
    valid = wafer_mask > 0
    ref_vals = reference_image[valid].astype(np.float32)
    test_vals = test_image[valid].astype(np.float32)

    if ref_vals.size == 0 or test_vals.size == 0:
        return reference_image.copy(), False

    ref_std = float(np.std(ref_vals))
    if ref_std < 1e-3:
        return reference_image.copy(), False

    ref_mean = float(np.mean(ref_vals))
    test_mean = float(np.mean(test_vals))
    test_std = float(np.std(test_vals))

    matched = (reference_image.astype(np.float32) - ref_mean) * (test_std / ref_std) + test_mean
    matched_uint8 = np.clip(np.round(matched), 0, 255).astype(np.uint8)
    return matched_uint8, True


def compute_golden_difference(
    test_image: np.ndarray,
    reference_image: np.ndarray,
    wafer_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute absolute and signed pixel differences between aligned test and golden reference.

    Args:
        test_image: 2D uint8 test image.
        reference_image: 2D uint8 aligned reference image.
        wafer_mask: 2D uint8 mask.

    Returns:
        Tuple of:
            - diff_uint8: Absolute difference array (0 outside wafer mask).
            - signed_diff: Signed int16 difference array (test - reference).
    """
    diff_uint8 = cv2.absdiff(test_image, reference_image)
    diff_uint8 = cv2.bitwise_and(diff_uint8, diff_uint8, mask=wafer_mask)

    signed_diff = test_image.astype(np.int16) - reference_image.astype(np.int16)
    signed_diff = np.where(wafer_mask > 0, signed_diff, np.int16(0))

    return diff_uint8, signed_diff


def compute_no_reference_residual(
    test_image: np.ndarray,
    wafer_mask: np.ndarray,
    background_blur_sigma_ratio: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute high-frequency residual defect image via multiscale background estimation.

    In the absence of a golden reference, estimates the low-frequency illumination and
    background pattern with a large Gaussian blur kernel proportional to image dimension.
    Subtracting the background isolates anomalous local variations.

    Args:
        test_image: 2D uint8 test image.
        wafer_mask: 2D uint8 wafer mask.
        background_blur_sigma_ratio: Ratio of Gaussian sigma to maximum image dimension (default 0.05).

    Returns:
        Tuple of:
            - diff_uint8: Absolute residual magnitude (0 outside wafer mask).
            - signed_diff: Signed int16 residual (test - background).
    """
    h, w = test_image.shape
    sigma = max(1.0, float(max(h, w)) * background_blur_sigma_ratio)
    ksize = 2 * int(round(3.0 * sigma)) + 1

    # Normalized convolution strictly inside wafer_mask so dark outer chuck pixels (outside mask)
    # do not bleed inward across the wafer bevel and create false rim residuals.
    mask_f = (wafer_mask > 0).astype(np.float32)
    weighted_img = cv2.GaussianBlur(test_image.astype(np.float32) * mask_f, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    weight_sum = cv2.GaussianBlur(mask_f, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    background = np.divide(weighted_img, np.maximum(weight_sum, 1e-6))

    signed_diff = np.round(test_image.astype(np.float32) - background).astype(np.int16)
    signed_diff = np.where(wafer_mask > 0, signed_diff, np.int16(0))

    diff_uint8 = np.clip(np.abs(signed_diff), 0, 255).astype(np.uint8)
    diff_uint8 = cv2.bitwise_and(diff_uint8, diff_uint8, mask=wafer_mask)

    return diff_uint8, signed_diff


def estimate_noise_sigma(
    diff_image: np.ndarray,
    wafer_mask: np.ndarray,
) -> Tuple[float, float]:
    """Estimate robust background noise statistics from an absolute difference image via MAD.

    For a zero-mean Gaussian differential noise field, the absolute difference |D|
    follows a folded-normal distribution with median(|D|) = 0.6745 * sigma_diff.
    When median(|D|) is 0 (e.g. heavily quantized or noiseless synthetic images),
    falls back to the standard deviation of the masked difference pixels.

    Args:
        diff_image: 2D uint8 absolute difference image.
        wafer_mask: 2D uint8 wafer domain mask (non-zero inside wafer).

    Returns:
        Tuple[float, float]: (median_val, sigma_diff)
    """
    masked_diff = diff_image[wafer_mask > 0]
    if masked_diff.size == 0:
        return 0.0, 0.0

    med_val = float(np.median(masked_diff))
    sigma_diff = med_val / 0.6745 if med_val > 0 else float(np.std(masked_diff))
    return med_val, sigma_diff


def apply_threshold(
    diff_image: np.ndarray,
    wafer_mask: np.ndarray,
    method: str = "otsu",
    fixed_threshold: float = 30.0,
    adaptive_block_size: int = 31,
    adaptive_c: float = 5.0,
    min_robust_contrast: float = 8.0,
    noise_floor_k: float = 3.5,
) -> Tuple[np.ndarray, float, str]:
    """Threshold a difference image into a binary candidate defect mask with noise-floor guard.

    Guards against Otsu over-segmentation on difference images lacking actual defects:
    1. If the 99.9th percentile of the masked difference image is below `min_robust_contrast`,
       returns an empty mask with reason "no significant difference found".
    2. Lower-bounds Otsu's threshold by the folded-normal statistical noise ceiling derived
       from the Median Absolute Deviation (MAD) and single config constant k:
           sigma_diff = median(|D|) / 0.6745
           noise_ceiling = k * sigma_diff   (default k = 3.5)
       Because sparse defects occupy < 0.1% of the wafer, unconstrained Otsu bisects the
       zero-mean noise distribution; enforcing T >= k * sigma_diff prevents noise flooding
       while still detecting small high-contrast defects (< 0.1% area).

    Args:
        diff_image: 2D uint8 absolute difference image.
        wafer_mask: 2D uint8 wafer mask.
        method: Thresholding technique: "otsu", "fixed", or "adaptive".
        fixed_threshold: Intensity cutoff for "fixed" method.
        adaptive_block_size: Pixel neighborhood size for "adaptive" method (must be odd).
        adaptive_c: Subtracted constant for "adaptive" method.
        min_robust_contrast: Minimum 99.9th-percentile contrast required to run Otsu.
        noise_floor_k: Statistical sigma multiplier k for deriving the noise ceiling.

    Returns:
        Tuple containing:
            - raw_mask: 2D uint8 binary mask (255 for candidate defects, 0 elsewhere).
            - threshold_value: Effective numerical threshold applied.
            - reason: Descriptive diagnostic string.
    """
    masked_diff = diff_image[wafer_mask > 0]
    if masked_diff.size == 0:
        return np.zeros_like(diff_image), 0.0, "Empty wafer mask"

    # Robust noise standard deviation via Median Absolute Deviation (MAD)
    med_val, sigma_diff = estimate_noise_sigma(diff_image, wafer_mask)

    # Folded-normal statistical noise ceiling centered at 0: T_noise = k * sigma_diff
    noise_ceiling = noise_floor_k * sigma_diff

    # Robust contrast spread (99.9th percentile)
    p999 = float(np.percentile(masked_diff, 99.9))

    method_clean = method.strip().lower()

    # Otsu low-contrast guard
    if method_clean == "otsu" and p999 < min_robust_contrast:
        reason = (
            f"no significant difference found (p99.9={p999:.2f} < "
            f"min_contrast={min_robust_contrast:.1f})"
        )
        logger.info("Threshold guard triggered: %s", reason)
        return np.zeros_like(diff_image), 0.0, reason

    if method_clean == "fixed":
        thresh_val = float(fixed_threshold)
        _, raw_mask = cv2.threshold(diff_image, int(round(thresh_val)), 255, cv2.THRESH_BINARY)
        reason = f"Applied fixed threshold of {thresh_val:.1f}."

    elif method_clean == "otsu":
        # Compute Otsu optimal threshold on wafer foreground pixels
        thresh_int, _ = cv2.threshold(masked_diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Lower-bound Otsu by the statistical noise ceiling k * sigma_diff
        thresh_val = float(max(float(thresh_int), round(noise_ceiling)))
        _, raw_mask = cv2.threshold(diff_image, int(round(thresh_val)), 255, cv2.THRESH_BINARY)
        reason = (
            f"Applied Otsu threshold of {thresh_val:.1f} "
            f"(raw Otsu: {thresh_int}, noise ceiling: {noise_ceiling:.1f}, p99.9: {p999:.1f})."
        )

    elif method_clean == "adaptive":
        ksize = adaptive_block_size if (adaptive_block_size > 1 and adaptive_block_size % 2 == 1) else 31
        raw_mask = cv2.adaptiveThreshold(
            diff_image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            ksize,
            adaptive_c,
        )
        thresh_val = float(adaptive_c)
        reason = f"Applied adaptive Gaussian threshold (block={ksize}, C={adaptive_c:.1f})."

    else:
        raise ValueError(f"Unknown threshold method '{method}'. Allowed: 'otsu', 'fixed', 'adaptive'")

    # Constrain to wafer disk boundary
    raw_mask = cv2.bitwise_and(raw_mask, wafer_mask)
    return raw_mask, thresh_val, reason


def _validate_morphology_kernel(kernel_size: int, op_name: str) -> None:
    """Validate that mathematical morphology kernel sizes are positive and odd."""
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(f"Morphology op '{op_name}' requires an odd positive integer kernel size, got {kernel_size}")


def morph_open(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Perform mathematical opening (erosion followed by dilation) with an elliptical kernel.

    Removes isolated high-frequency noise spikes, false single-pixel dust specks, and
    thin filament artifacts without altering large defect geometries.
    """
    _validate_morphology_kernel(kernel_size, "open")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def morph_close(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Perform mathematical closing (dilation followed by erosion) with an elliptical kernel.

    Fills interior pinholes, bridges narrow fissures, and joins fragmented defect edges.
    """
    _validate_morphology_kernel(kernel_size, "close")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def morph_erode(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Perform mathematical erosion with an elliptical kernel."""
    _validate_morphology_kernel(kernel_size, "erode")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.erode(mask, kernel)


def morph_dilate(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Perform mathematical dilation with an elliptical kernel."""
    _validate_morphology_kernel(kernel_size, "dilate")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask, kernel)


def apply_morphology(
    mask: np.ndarray,
    ops: Optional[List[Tuple[str, int]]] = None,
) -> np.ndarray:
    """Apply an ordered sequence of mathematical morphology operations.

    Default sequence:
        1. Small opening (kernel 3x3) to eliminate single-pixel sensor noise.
        2. Small closing (kernel 3x3) to solidify defect cores and bridge fissures.
    Caution: excessively large structuring elements will erode and delete genuine
    low-contrast submicron physical defects.

    Args:
        mask: 2D uint8 binary candidate mask.
        ops: List of (operation_name, kernel_size) tuples.

    Returns:
        np.ndarray: Cleaned binary defect mask.
    """
    if ops is None:
        ops = [("open", 3), ("close", 3)]

    current_mask = mask.copy()
    for op_name, ksize in ops:
        clean_op = op_name.strip().lower()
        if clean_op == "open":
            current_mask = morph_open(current_mask, ksize)
        elif clean_op == "close":
            current_mask = morph_close(current_mask, ksize)
        elif clean_op == "erode":
            current_mask = morph_erode(current_mask, ksize)
        elif clean_op == "dilate":
            current_mask = morph_dilate(current_mask, ksize)
        else:
            raise ValueError(f"Unknown morphology operation '{op_name}'. Allowed: 'open', 'close', 'erode', 'dilate'")

    return current_mask


def count_connected_candidates(binary_mask: np.ndarray) -> int:
    """Count discrete candidate defect components using connected components labeling."""
    num_labels, _ = cv2.connectedComponents(binary_mask)
    # connectedComponents includes background label 0
    return max(0, int(num_labels) - 1)
