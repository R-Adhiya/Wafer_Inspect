"""Geometric and photometric defect measurement in original-image coordinates.

Pure functions operating on segmented Region objects and noise statistics.
No filesystem I/O, no global state, and never mutates input arrays or regions.

All spatial measurements (area, bbox, centroid, perimeter, equivalent diameter,
and oriented minimum-area bounding box length/thickness) are mapped back into
ORIGINAL image pixel coordinates using the Phase 3 resize scale factor:
  - lengths  *= 1 / scale_factor
  - areas    *= 1 / (scale_factor ** 2)
"""

from dataclasses import dataclass
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
    """Configuration for defect measurement and optional physical calibration."""

    pixel_size_um: Optional[float] = Field(
        default=None,
        gt=0.0,
        description=(
            "Optional spatial calibration in micrometres per original image pixel. "
            "When None, measurements are reported strictly in uncalibrated image pixels."
        ),
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


def measure_region(
    region: Region,
    noise_sigma: float,
    detected_by: DetectedByPath,
    config: Optional[MeasurementConfig] = None,
) -> RegionMeasurement:
    """Compute full original-pixel measurements for a single accepted defect Region.

    Args:
        region: Segmented Region object (in processed-image coordinates with scale_factor).
        noise_sigma: MAD-based difference-image noise sigma from detection.
        detected_by: Detection path name ("golden", "self_reference", or "background").
        config: Optional MeasurementConfig providing pixel_size_um for physical units.

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

    return RegionMeasurement(
        id=region.id,
        area_px=area_px,
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
    )


def measure_regions(
    regions: List[Region],
    noise_sigma: float,
    detected_by: DetectedByPath,
    config: Optional[MeasurementConfig] = None,
) -> List[RegionMeasurement]:
    """Measure all accepted defect regions in original-image pixel coordinates."""
    if config is None:
        config = MeasurementConfig()
    return [
        measure_region(region=r, noise_sigma=noise_sigma, detected_by=detected_by, config=config)
        for r in regions
    ]
