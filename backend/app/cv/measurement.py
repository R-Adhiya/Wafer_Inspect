"""Geometric and photometric defect measurement in original-image coordinates.

Pure functions operating on segmented Region objects and noise statistics.
No filesystem I/O, no global state, and never mutates input arrays or regions.

All spatial measurements (area, bbox, centroid, perimeter, equivalent diameter,
and oriented minimum-area bounding box length/thickness) are mapped back into
ORIGINAL image pixel coordinates using the Phase 3 resize scale factor:
  - lengths  *= 1 / scale_factor
  - areas    *= 1 / (scale_factor ** 2)
"""

from dataclasses import dataclass, field
import math
from typing import Dict, List, Literal, Optional, Tuple
import cv2
import numpy as np
from pydantic import BaseModel, Field

from app.cv.segmentation import Region, map_region_to_original_coords

DetectedByPath = Literal["golden", "self_reference", "background"]

UNCALIBRATED_STATUS = "uncalibrated: measurements are in image pixels"

_DETECTED_BY_PHRASE: Dict[str, str] = {
    "golden": "golden-image comparison",
    "self_reference": "die-to-die self-reference comparison",
    "background": "no-reference background subtraction",
}


class MeasurementConfig(BaseModel):
    """Configuration for defect measurement, half-max boundary refinement, and optional calibration."""

    pixel_size_um: Optional[float] = Field(
        default=None,
        gt=0.0,
        description=(
            "Optional spatial calibration in micrometres per original image pixel. "
            "When None, measurements are reported strictly in uncalibrated image pixels."
        ),
    )
    refine_measurements: bool = Field(
        default=True,
        description=(
            "Refine each accepted region's boundary using local half-maximum thresholding (0.5 * P) "
            "on the exact unshifted difference image before measuring area, perimeter, centroid, and minAreaRect."
        ),
    )
    refine_pad_px: int = Field(
        default=6,
        ge=1,
        le=32,
        description="Local window padding in pixels around each region bbox during half-max refinement.",
    )


def format_calibration_status(config: Optional[MeasurementConfig] = None) -> str:
    """Return a clear human-readable calibration status string."""
    if config is not None and config.pixel_size_um is not None:
        return f"calibrated: {config.pixel_size_um:g} um/pixel"
    return UNCALIBRATED_STATUS


@dataclass
class RegionMeasurement:
    """Full geometric, oriented, photometric, and diagnostic measurements in original pixels."""

    id: int
    area_px: int
    detected_area_px: int
    bbox: Tuple[int, int, int, int]  # (x, y, width, height) in original image pixels
    width_px: int
    height_px: int
    centroid: Tuple[float, float]  # (cx, cy) in original image pixels
    perimeter_px: float
    equivalent_diameter_px: float
    length_px: float
    thickness_px: float
    angle_deg: float
    elongation: float
    polarity: Literal["bright", "dark"]
    mean_contrast: float
    peak_contrast: float
    contrast_to_noise: float
    touches_edge: bool
    detected_by: DetectedByPath
    why_flagged: str
    severity: Literal["LOW", "MEDIUM", "HIGH"] = "LOW"
    area_um2: Optional[float] = None
    length_um: Optional[float] = None
    thickness_um: Optional[float] = None
    contour: List[Tuple[int, int]] = field(default_factory=list)  # Simplified polygon in original image px (<= 200 pts)


def refine_region_mask(
    region: Region,
    exact_difference: np.ndarray,
    wafer_mask: Optional[np.ndarray] = None,
    pad_px: int = 6,
) -> Region:
    """Refine a detected region's binary mask by local half-maximum edge localisation.

    WHY half-maximum (0.5 * P) edge localisation:
    Upstream denoising (Gaussian blur), morphological opening, and 3x3 shift-tolerant
    self-reference differencing systematically erode, notch, or shrink a defect's binary
    detection mask, causing large area underestimation (especially on small discs and
    scratches crossing die streets). For a step edge blurred by a symmetric PSF, the 50%
    intensity contour sits directly on the true geometric edge regardless of contrast.
    Evaluating `0.5 * P` (where `P` is the mean of the top 10% exact difference values
    inside the detection mask) on the EXACT unshifted difference image recovers the true
    spatial extent without inflating background noise.

    Residual error on very thin lines (thickness 1-3 px):
    When a scratch is narrower than or comparable to the 3x3 denoising PSF (~2-3 px FWHM),
    smoothing both attenuates the line's peak intensity `P` and spreads energy laterally.
    Half-maximum thresholding therefore still carries a systematic +1 to +2 px thickness
    bias (and proportional area overestimation) on 2-3 px thin lines. This is a physical
    resolution limit of pre-smoothed imagery, not a threshold bug.

    Does not mutate inputs.

    Args:
        region: Accepted Region from detection/segmentation (in processed coordinates).
        exact_difference: 2D uint8 exact absolute difference image (for self-reference,
                          the unshifted |test - ref| WITHOUT the 3x3 shift tolerance).
        wafer_mask: Optional 2D uint8 wafer domain mask for clipping and rim edge flags.
        pad_px: Padding around the region's bounding box in processed pixels (default 6).

    Returns:
        A new Region carrying the refined mask, bbox, area_px, centroid, contours, and
        touches_edge flag while preserving id, polarity, contrast stats, and scale_factor.
    """
    img_h, img_w = exact_difference.shape[:2]
    x, y, w, h = region.bbox

    wx0 = max(0, x - pad_px)
    wy0 = max(0, y - pad_px)
    wx1 = min(img_w, x + w + pad_px)
    wy1 = min(img_h, y + h + pad_px)
    win_h = wy1 - wy0
    win_w = wx1 - wx0

    if win_h <= 0 or win_w <= 0:
        return region

    # Place original detection mask into the local window
    det_local = np.zeros((win_h, win_w), dtype=bool)
    det_local[y - wy0 : y - wy0 + h, x - wx0 : x - wx0 + w] = region.mask

    diff_local = exact_difference[wy0:wy1, wx0:wx1]
    det_vals = diff_local[det_local].astype(np.float32)
    if det_vals.size == 0:
        return region

    # P = mean of the top 10% of exact difference values inside the detection mask
    k_top = max(1, int(math.ceil(0.10 * det_vals.size)))
    if k_top >= det_vals.size:
        peak_mean = float(np.mean(det_vals))
    else:
        top_vals = np.partition(det_vals, det_vals.size - k_top)[det_vals.size - k_top :]
        peak_mean = float(np.mean(top_vals))

    if peak_mean <= 1e-6:
        return region

    half_max = 0.5 * peak_mean
    cand_bool = diff_local.astype(np.float32) >= half_max

    wafer_local: Optional[np.ndarray] = None
    if wafer_mask is not None:
        wafer_local = wafer_mask[wy0:wy1, wx0:wx1] > 0
        cand_bool = cand_bool & wafer_local

    # Keep connected component(s) overlapping the original detection mask
    cand_u8 = cand_bool.astype(np.uint8) * 255
    num_labels, labels = cv2.connectedComponents(cand_u8, connectivity=8)
    overlapping_labels = set(np.unique(labels[det_local]).tolist())
    overlapping_labels.discard(0)

    if not overlapping_labels:
        return region

    kept_u8 = np.isin(labels, list(overlapping_labels)).astype(np.uint8) * 255

    # Fill interior holes cleanly via RETR_EXTERNAL contours, then re-clip to wafer_mask
    ext_contours, _ = cv2.findContours(kept_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_u8 = np.zeros_like(kept_u8)
    if ext_contours:
        cv2.drawContours(filled_u8, ext_contours, -1, 255, thickness=cv2.FILLED)
    else:
        filled_u8 = kept_u8

    if wafer_local is not None:
        filled_u8 = np.where(wafer_local, filled_u8, 0).astype(np.uint8)

    ys_loc, xs_loc = np.nonzero(filled_u8)
    if xs_loc.size == 0:
        return region

    rx_loc = int(xs_loc.min())
    ry_loc = int(ys_loc.min())
    rw = int(xs_loc.max()) - rx_loc + 1
    rh = int(ys_loc.max()) - ry_loc + 1
    rx = wx0 + rx_loc
    ry = wy0 + ry_loc

    refined_crop_bool = (filled_u8[ry_loc : ry_loc + rh, rx_loc : rx_loc + rw] > 0).copy()
    refined_area_px = int(xs_loc.size)
    refined_cx = round(float(np.mean(xs_loc)) + wx0, 2)
    refined_cy = round(float(np.mean(ys_loc)) + wy0, 2)

    # Extract external contours in global processed coordinates
    crop_u8 = refined_crop_bool.astype(np.uint8) * 255
    cnts, _ = cv2.findContours(crop_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_contours: List[List[Tuple[int, int]]] = []
    if cnts:
        for cnt in sorted(cnts, key=cv2.contourArea, reverse=True):
            pts = [(int(pt[0][0]) + rx, int(pt[0][1]) + ry) for pt in cnt]
            if pts:
                all_contours.append(pts)
    primary_contour = all_contours[0] if all_contours else []

    # Re-evaluate edge contact on the refined mask
    if wafer_mask is not None:
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        eroded_w = cv2.erode(wafer_mask[ry : ry + rh, rx : rx + rw], k3)
        rim_crop = (wafer_mask[ry : ry + rh, rx : rx + rw] > 0) & (eroded_w == 0)
        touches_edge = bool(
            np.any(refined_crop_bool & rim_crop)
            or rx == 0
            or ry == 0
            or (rx + rw) >= img_w
            or (ry + rh) >= img_h
        )
    else:
        touches_edge = bool(rx == 0 or ry == 0 or (rx + rw) >= img_w or (ry + rh) >= img_h)

    return Region(
        id=region.id,
        bbox=(rx, ry, rw, rh),
        area_px=refined_area_px,
        centroid=(refined_cx, refined_cy),
        mask=refined_crop_bool,
        contour=primary_contour,
        polarity=region.polarity,
        mean_contrast=region.mean_contrast,
        peak_contrast=region.peak_contrast,
        touches_edge=touches_edge,
        scale_factor=region.scale_factor,
        decision=region.decision,
        decision_reason=region.decision_reason,
        contours=all_contours,
    )


def refine_regions(
    regions: List[Region],
    exact_difference: np.ndarray,
    wafer_mask: Optional[np.ndarray] = None,
    pad_px: int = 6,
) -> List[Region]:
    """Apply local half-maximum mask refinement to all accepted defect regions."""
    return [
        refine_region_mask(r, exact_difference=exact_difference, wafer_mask=wafer_mask, pad_px=pad_px)
        for r in regions
    ]


def _compute_oriented_box_metrics(
    region: Region,
    scale_factor: float,
) -> Tuple[float, float, float, float]:
    """Compute oriented bounding box length, thickness, angle, and elongation.

    WHY oriented minAreaRect instead of axis-aligned bbox:
    The axis-aligned bounding box of a diagonal scratch (e.g. 45 degrees) severely
    overstates its thickness and understates or distorts its true length because the
    box spans sin/cos projections of both length and width. cv2.minAreaRect aligns
    directly to the defect's principal axis to recover true scratch length, thickness,
    orientation angle, and elongation ratio.

    Returns:
        Tuple of (length_px, thickness_px, angle_deg, elongation) in original pixels.
    """
    bx, by, _, _ = region.bbox
    ys, xs = np.nonzero(region.mask)
    if xs.size == 0:
        return 0.0, 0.0, 0.0, 1.0

    pts = np.column_stack((xs + bx, ys + by)).astype(np.float32)

    if pts.shape[0] == 1:
        unit = 1.0 / scale_factor
        return round(unit, 2), round(unit, 2), 0.0, 1.0

    rect = cv2.minAreaRect(pts)
    (_, _), (rw, rh), _ = rect

    # OpenCV minAreaRect measures center-to-center pixel span; adding 1.0 px gives
    # physical pixel coverage (so a 1-px-wide line has thickness 1.0 instead of 0.0).
    long_proc = max(float(rw), float(rh)) + 1.0
    short_proc = min(float(rw), float(rh)) + 1.0

    length_px = long_proc / scale_factor
    thickness_px = max(short_proc / scale_factor, 1e-6)
    elongation = length_px / thickness_px

    # Derive orientation unambiguously from the longer edge vector of boxPoints
    # to avoid OpenCV version-specific minAreaRect angle conventions.
    box = cv2.boxPoints(rect)
    edge0 = box[1] - box[0]
    edge1 = box[2] - box[1]
    len0 = float(np.hypot(edge0[0], edge0[1]))
    len1 = float(np.hypot(edge1[0], edge1[1]))

    major_vec = edge0 if len0 >= len1 else edge1
    if max(len0, len1) < 1e-6:
        angle_deg = 0.0
    else:
        angle_deg = math.degrees(math.atan2(float(major_vec[1]), float(major_vec[0]))) % 180.0
        if abs(angle_deg - 180.0) < 1e-6:
            angle_deg = 0.0

    return (
        round(length_px, 2),
        round(thickness_px, 2),
        round(angle_deg, 2),
        round(elongation, 2),
    )


def _compute_perimeter_px(region: Region, scale_factor: float) -> float:
    """Compute closed external contour perimeter in original image pixels via cv2.arcLength."""
    contour_list = region.contours if region.contours else ([region.contour] if region.contour else [])
    if not contour_list:
        return 0.0

    total_proc_perimeter = 0.0
    for cnt in contour_list:
        if len(cnt) < 2:
            continue
        cnt_np = np.array(cnt, dtype=np.int32).reshape(-1, 1, 2)
        total_proc_perimeter += float(cv2.arcLength(cnt_np, closed=True))

    return round(total_proc_perimeter / scale_factor, 2)


def build_why_flagged(
    polarity: Literal["bright", "dark"],
    area_px: int,
    mean_contrast: float,
    contrast_to_noise: float,
    noise_sigma: float,
    detected_by: DetectedByPath,
    touches_edge: bool = False,
) -> str:
    """Build a plain-language explanation sentence grounded in measured defect numbers."""
    polarity_title = "Bright" if polarity == "bright" else "Dark"
    path_phrase = _DETECTED_BY_PHRASE.get(detected_by, detected_by)
    edge_suffix = " (touches wafer rim)" if touches_edge else ""

    if noise_sigma > 1e-6:
        cnr_phrase = f"{contrast_to_noise:.1f}x the noise level"
    else:
        cnr_phrase = "above near-zero background noise"

    return (
        f"{polarity_title} region, {area_px} px, contrast {mean_contrast:.1f} gray levels = "
        f"{cnr_phrase}, found by {path_phrase}{edge_suffix}."
    )


def _compute_simplified_original_contour(
    region: Region,
    scale_factor: float,
    orig_image_size: Optional[Tuple[int, int]] = None,
    epsilon_px: float = 0.75,
    max_points: int = 200,
) -> List[Tuple[int, int]]:
    """Compute a simplified external contour polygon in ORIGINAL image coordinates.

    Uses `cv2.approxPolyDP` with epsilon ~0.75 px (in original coordinates) and caps
    vertex count at `max_points` (200) for fast SVG rendering in the browser.
    """
    src_cnt = region.contour
    if not src_cnt:
        return []

    scaled_pts: List[List[int]] = []
    for px, py in src_cnt:
        ox = int(round(float(px) / scale_factor))
        oy = int(round(float(py) / scale_factor))
        if orig_image_size is not None:
            max_w, max_h = int(orig_image_size[0]), int(orig_image_size[1])
            ox = max(0, min(max_w - 1, ox))
            oy = max(0, min(max_h - 1, oy))
        scaled_pts.append([ox, oy])

    if len(scaled_pts) <= 2:
        return [(pt[0], pt[1]) for pt in scaled_pts]

    cnt_np = np.array(scaled_pts, dtype=np.int32).reshape(-1, 1, 2)
    approx = cv2.approxPolyDP(cnt_np, epsilon=float(epsilon_px), closed=True)

    # Iteratively increase epsilon if vertex count still exceeds max_points
    curr_eps = float(epsilon_px)
    while approx.shape[0] > max_points and curr_eps < 50.0:
        curr_eps *= 1.5
        approx = cv2.approxPolyDP(cnt_np, epsilon=curr_eps, closed=True)

    pts_out = [(int(pt[0][0]), int(pt[0][1])) for pt in approx]
    if len(pts_out) > max_points:
        indices = np.linspace(0, len(pts_out) - 1, max_points, dtype=int)
        pts_out = [pts_out[int(i)] for i in indices]

    return pts_out


def measure_region(
    region: Region,
    noise_sigma: float,
    detected_by: DetectedByPath,
    config: Optional[MeasurementConfig] = None,
    detected_area_proc_px: Optional[int] = None,
    orig_image_size: Optional[Tuple[int, int]] = None,
) -> RegionMeasurement:
    """Compute full original-pixel measurements for a single accepted defect Region.

    Args:
        region: Segmented (or half-max refined) Region object in processed-image coordinates.
        noise_sigma: MAD-based difference-image noise sigma from detection.
        detected_by: Detection path name ("golden", "self_reference", or "background").
        config: Optional MeasurementConfig providing pixel_size_um for physical units.
        detected_area_proc_px: Optional pre-refinement detected pixel count in processed coordinates.

    Returns:
        RegionMeasurement with all geometric, oriented, photometric, and calibration metrics.
    """
    if config is None:
        config = MeasurementConfig()

    scale = float(region.scale_factor)
    if scale <= 0:
        raise ValueError(f"region.scale_factor must be positive, got {scale}")

    # Map area (x 1/scale^2) and bbox/centroid (x 1/scale) back to original image pixels
    area_px = int(round(float(region.area_px) / (scale * scale)))
    raw_det_area = region.area_px if detected_area_proc_px is None else detected_area_proc_px
    detected_area_px = int(round(float(raw_det_area) / (scale * scale)))
    orig_bbox, orig_centroid = map_region_to_original_coords(region.bbox, region.centroid, scale)
    orig_x, orig_y, orig_w, orig_h = orig_bbox

    perimeter_px = _compute_perimeter_px(region, scale)
    equivalent_diameter_px = round(math.sqrt(4.0 * float(area_px) / math.pi), 2) if area_px > 0 else 0.0
    length_px, thickness_px, angle_deg, elongation = _compute_oriented_box_metrics(region, scale)

    mean_contrast = round(float(region.mean_contrast), 2)
    peak_contrast = round(float(region.peak_contrast), 2)
    if noise_sigma > 1e-6:
        contrast_to_noise = round(mean_contrast / float(noise_sigma), 2)
    else:
        contrast_to_noise = round(mean_contrast, 2)

    why_flagged = build_why_flagged(
        polarity=region.polarity,
        area_px=area_px,
        mean_contrast=mean_contrast,
        contrast_to_noise=contrast_to_noise,
        noise_sigma=float(noise_sigma),
        detected_by=detected_by,
        touches_edge=region.touches_edge,
    )

    area_um2: Optional[float] = None
    length_um: Optional[float] = None
    thickness_um: Optional[float] = None
    if config.pixel_size_um is not None:
        px_um = float(config.pixel_size_um)
        area_um2 = round(float(area_px) * (px_um * px_um), 4)
        length_um = round(length_px * px_um, 4)
        thickness_um = round(thickness_px * px_um, 4)

    contour_orig = _compute_simplified_original_contour(
        region=region,
        scale_factor=scale,
        orig_image_size=orig_image_size,
        epsilon_px=0.75,
        max_points=200,
    )

    return RegionMeasurement(
        id=region.id,
        area_px=area_px,
        detected_area_px=detected_area_px,
        bbox=(orig_x, orig_y, orig_w, orig_h),
        width_px=orig_w,
        height_px=orig_h,
        centroid=orig_centroid,
        perimeter_px=perimeter_px,
        equivalent_diameter_px=equivalent_diameter_px,
        length_px=length_px,
        thickness_px=thickness_px,
        angle_deg=angle_deg,
        elongation=elongation,
        polarity=region.polarity,
        mean_contrast=mean_contrast,
        peak_contrast=peak_contrast,
        contrast_to_noise=contrast_to_noise,
        touches_edge=region.touches_edge,
        detected_by=detected_by,
        why_flagged=why_flagged,
        area_um2=area_um2,
        length_um=length_um,
        thickness_um=thickness_um,
        contour=contour_orig,
    )


def measure_regions(
    regions: List[Region],
    noise_sigma: float,
    detected_by: DetectedByPath,
    config: Optional[MeasurementConfig] = None,
    detected_regions: Optional[List[Region]] = None,
    orig_image_size: Optional[Tuple[int, int]] = None,
) -> List[RegionMeasurement]:
    """Measure all accepted defect regions in original-image pixel coordinates."""
    if config is None:
        config = MeasurementConfig()
    det_area_by_id = (
        {r.id: r.area_px for r in detected_regions}
        if detected_regions is not None
        else {}
    )
    return [
        measure_region(
            region=r,
            noise_sigma=noise_sigma,
            detected_by=detected_by,
            config=config,
            detected_area_proc_px=det_area_by_id.get(r.id),
            orig_image_size=orig_image_size,
        )
        for r in regions
    ]
