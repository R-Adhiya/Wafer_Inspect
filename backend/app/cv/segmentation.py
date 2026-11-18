"""Defect segmentation, connected-component analysis, fragment merging, and overlays.

Extracts discrete candidate defect regions from a binary morphology mask using
8-connected component analysis, optionally merges nearby fragments (e.g. a scratch
split where it crosses periodic die streets) while preserving exact original pixel
counts, and renders severity-styled bounding box, contour, and final overlays.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple
import cv2
import numpy as np

# Neutral fallback color (RGB) when no severity mapping is provided
NEUTRAL_OVERLAY_COLOR_RGB = (59, 130, 246)

# Severity overlay styles distinguishing LOW / MEDIUM / HIGH by BOTH colour and
# line thickness/style + text badge for colour-vision accessibility:
#   LOW    -> thin dashed blue   (thickness 1, tag [L])
#   MEDIUM -> medium solid orange (thickness 2, tag [M])
#   HIGH   -> thick double red    (thickness 3, tag [H])
SEVERITY_OVERLAY_STYLES: Dict[str, Dict[str, object]] = {
    "LOW": {
        "color": (37, 99, 235),
        "thickness": 1,
        "dashed": True,
        "double": False,
        "tag": "L",
    },
    "MEDIUM": {
        "color": (234, 88, 12),
        "thickness": 2,
        "dashed": False,
        "double": False,
        "tag": "M",
    },
    "HIGH": {
        "color": (220, 38, 38),
        "thickness": 3,
        "dashed": False,
        "double": True,
        "tag": "H",
    },
}


@dataclass
class Region:
    """Discrete segmented defect candidate region in processed-image coordinates."""

    id: int
    bbox: Tuple[int, int, int, int]  # (x, y, w, h) in processed-image pixels
    area_px: int
    centroid: Tuple[float, float]  # (cx, cy) in processed-image pixels
    mask: np.ndarray  # Cropped 2D boolean mask of shape (h, w) to save memory
    contour: List[Tuple[int, int]]  # Primary external contour points [(x, y), ...]
    polarity: Literal["bright", "dark"]
    mean_contrast: float
    peak_contrast: float
    touches_edge: bool
    scale_factor: float = 1.0
    decision: Literal["pending", "accepted", "rejected"] = "pending"
    decision_reason: str = ""
    contours: List[List[Tuple[int, int]]] = field(default_factory=list)

    def original_bbox_and_centroid(self) -> Tuple[Tuple[int, int, int, int], Tuple[float, float]]:
        """Map this region's bounding box and centroid back to original-image pixels."""
        return map_region_to_original_coords(self.bbox, self.centroid, self.scale_factor)


def map_region_to_original_coords(
    bbox: Tuple[int, int, int, int],
    centroid: Tuple[float, float],
    scale_factor: float,
) -> Tuple[Tuple[int, int, int, int], Tuple[float, float]]:
    """Map processed-image bounding box and centroid back to original-image pixel space.

    Args:
        bbox: (x, y, w, h) in processed-image coordinates.
        centroid: (cx, cy) in processed-image coordinates.
        scale_factor: Resize scale factor from preprocessing (processed / original).

    Returns:
        Tuple of ((orig_x, orig_y, orig_w, orig_h), (orig_cx, orig_cy)).
    """
    if scale_factor <= 0:
        raise ValueError(f"scale_factor must be positive, got {scale_factor}")

    x, y, w, h = bbox
    cx, cy = centroid

    orig_x = int(round(x / scale_factor))
    orig_y = int(round(y / scale_factor))
    orig_w = max(1, int(round(w / scale_factor)))
    orig_h = max(1, int(round(h / scale_factor)))
    orig_cx = round(float(cx / scale_factor), 2)
    orig_cy = round(float(cy / scale_factor), 2)

    return (orig_x, orig_y, orig_w, orig_h), (orig_cx, orig_cy)


def _compute_edge_boundary_mask(
    image_shape: Tuple[int, int],
    wafer_mask: Optional[np.ndarray],
) -> np.ndarray:
    """Build boolean mask of pixels lying on the wafer rim boundary or image border."""
    h, w = image_shape
    if wafer_mask is not None:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        eroded = cv2.erode(wafer_mask, kernel)
        boundary = (wafer_mask > 0) & (eroded == 0)
    else:
        boundary = np.zeros((h, w), dtype=bool)

    if h > 0 and w > 0:
        boundary[0, :] = True
        boundary[-1, :] = True
        boundary[:, 0] = True
        boundary[:, -1] = True

    return boundary


def segment_regions(
    morphology_mask: np.ndarray,
    signed_difference: np.ndarray,
    wafer_mask: Optional[np.ndarray] = None,
    scale_factor: float = 1.0,
    merge_gap_px: int = 0,
) -> List[Region]:
    """Segment 8-connected components from a binary mask into deterministic Region objects.

    Does not mutate input arrays. When `merge_gap_px > 0`, groups nearby fragments by
    labeling connected components on a copy of the mask dilated by an elliptical kernel
    of radius `merge_gap_px`, then assigns each group strictly the ORIGINAL mask pixels
    inside it so `area_px` stays the exact true foreground pixel count.

    External contours are extracted with cv2.RETR_EXTERNAL (interior holes are ignored
    in contours while `area_px` retains exact pixel count).

    Region IDs are assigned deterministically (1..N) after sorting by centroid (y, x).

    Args:
        morphology_mask: 2D uint8 binary mask (0 or 255) from Phase 4 detection.
        signed_difference: 2D int16 signed difference array (test - ref or test - bg).
        wafer_mask: Optional 2D uint8 wafer domain mask for detecting rim-touching defects.
        scale_factor: Scale factor from preprocessing resize step (default 1.0).
        merge_gap_px: Radius in pixels for elliptical fragment grouping (0 disables merging).

    Returns:
        List[Region]: Deterministically ordered list of candidate regions.
    """
    if not isinstance(morphology_mask, np.ndarray) or morphology_mask.ndim != 2:
        raise ValueError("morphology_mask must be a 2D NumPy array.")
    if not isinstance(signed_difference, np.ndarray) or signed_difference.shape != morphology_mask.shape:
        raise ValueError("signed_difference must be a 2D NumPy array matching morphology_mask shape.")
    if scale_factor <= 0:
        raise ValueError(f"scale_factor must be positive, got {scale_factor}")
    if merge_gap_px < 0:
        raise ValueError(f"merge_gap_px must be >= 0, got {merge_gap_px}")

    binary = (morphology_mask > 0).astype(np.uint8) * 255
    if not np.any(binary):
        return []

    edge_boundary = _compute_edge_boundary_mask(binary.shape[:2], wafer_mask)
    raw_regions: List[Region] = []

    if merge_gap_px > 0:
        # Label groups on a dilated copy, then restrict back to ORIGINAL foreground pixels
        ksize = 2 * int(merge_gap_px) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        dilated = cv2.dilate(binary, kernel)
        num_groups, group_labels = cv2.connectedComponents(dilated, connectivity=8)
        effective_labels = np.where(binary > 0, group_labels, 0)

        for gid in range(1, num_groups):
            ys, xs = np.nonzero(effective_labels == gid)
            if xs.size == 0:
                continue

            x = int(xs.min())
            y = int(ys.min())
            w = int(xs.max()) - x + 1
            h = int(ys.max()) - y + 1
            area_px = int(xs.size)
            cx = float(np.mean(xs))
            cy = float(np.mean(ys))

            cropped_bool = (effective_labels[y : y + h, x : x + w] == gid).copy()
            raw_regions.append(
                _build_region_from_crop(
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    area_px=area_px,
                    cx=cx,
                    cy=cy,
                    cropped_bool=cropped_bool,
                    signed_difference=signed_difference,
                    edge_boundary=edge_boundary,
                    scale_factor=scale_factor,
                )
            )
    else:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num_labels <= 1:
            return []

        for label_idx in range(1, num_labels):
            x = int(stats[label_idx, cv2.CC_STAT_LEFT])
            y = int(stats[label_idx, cv2.CC_STAT_TOP])
            w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
            h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
            area_px = int(stats[label_idx, cv2.CC_STAT_AREA])
            cx = float(centroids[label_idx][0])
            cy = float(centroids[label_idx][1])

            cropped_bool = (labels[y : y + h, x : x + w] == label_idx).copy()
            raw_regions.append(
                _build_region_from_crop(
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    area_px=area_px,
                    cx=cx,
                    cy=cy,
                    cropped_bool=cropped_bool,
                    signed_difference=signed_difference,
                    edge_boundary=edge_boundary,
                    scale_factor=scale_factor,
                )
            )

    # Deterministic ordering: sorted by centroid y, then centroid x (tie-broken by bbox)
    raw_regions.sort(key=lambda r: (r.centroid[1], r.centroid[0], r.bbox[1], r.bbox[0]))
    for idx, region in enumerate(raw_regions, start=1):
        region.id = idx

    return raw_regions


def _build_region_from_crop(
    x: int,
    y: int,
    w: int,
    h: int,
    area_px: int,
    cx: float,
    cy: float,
    cropped_bool: np.ndarray,
    signed_difference: np.ndarray,
    edge_boundary: np.ndarray,
    scale_factor: float,
) -> Region:
    """Build a Region dataclass from its cropped boolean mask and bounding box."""
    cropped_u8 = cropped_bool.astype(np.uint8) * 255
    contours, _ = cv2.findContours(cropped_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    all_contours: List[List[Tuple[int, int]]] = []
    if contours:
        sorted_cnts = sorted(contours, key=cv2.contourArea, reverse=True)
        for cnt in sorted_cnts:
            pts = [(int(pt[0][0]) + x, int(pt[0][1]) + y) for pt in cnt]
            if pts:
                all_contours.append(pts)
        contour_pts = all_contours[0] if all_contours else []
    else:
        contour_pts = []

    signed_crop = signed_difference[y : y + h, x : x + w][cropped_bool].astype(np.float32)
    mean_signed = float(np.mean(signed_crop)) if signed_crop.size > 0 else 0.0
    polarity: Literal["bright", "dark"] = "bright" if mean_signed >= 0.0 else "dark"

    abs_signed = np.abs(signed_crop)
    mean_contrast = float(np.mean(abs_signed)) if abs_signed.size > 0 else 0.0
    peak_contrast = float(np.max(abs_signed)) if abs_signed.size > 0 else 0.0

    touches_edge = bool(np.any(cropped_bool & edge_boundary[y : y + h, x : x + w]))

    return Region(
        id=0,
        bbox=(x, y, w, h),
        area_px=area_px,
        centroid=(round(cx, 2), round(cy, 2)),
        mask=cropped_bool,
        contour=contour_pts,
        polarity=polarity,
        mean_contrast=round(mean_contrast, 2),
        peak_contrast=round(peak_contrast, 2),
        touches_edge=touches_edge,
        scale_factor=scale_factor,
        contours=all_contours,
    )


def build_filtered_mask_and_labels(
    image_shape: Tuple[int, int],
    accepted_regions: List[Region],
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct full-frame filtered binary mask and integer label map for accepted regions.

    Args:
        image_shape: (height, width) of the processed image.
        accepted_regions: List of accepted Region objects.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - filtered_mask: 2D uint8 binary mask (255 for accepted regions, 0 elsewhere).
            - label_map: 2D int32 map with region.id at each region's pixels, 0 background.
    """
    h, w = image_shape[:2]
    filtered_mask = np.zeros((h, w), dtype=np.uint8)
    label_map = np.zeros((h, w), dtype=np.int32)

    for region in accepted_regions:
        x, y, rw, rh = region.bbox
        region_slice = filtered_mask[y : y + rh, x : x + rw]
        region_slice[region.mask] = 255
        label_slice = label_map[y : y + rh, x : x + rw]
        label_slice[region.mask] = region.id

    return filtered_mask, label_map


def _to_rgb_canvas(image: np.ndarray) -> np.ndarray:
    """Convert a 2D grayscale, 16-bit, RGB, or RGBA image to a 3-channel uint8 RGB copy."""
    if image.dtype == np.uint16:
        arr = np.clip(np.round(image.astype(np.float32) / 257.0), 0, 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        arr = np.clip(np.round(image), 0, 255).astype(np.uint8)
    else:
        arr = image.copy()

    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    if arr.ndim == 3 and arr.shape[2] == 4:
        return arr[:, :, :3].copy()
    if arr.ndim == 3 and arr.shape[2] == 3:
        return arr.copy()
    raise ValueError(f"Unsupported image shape for overlay drawing: {image.shape}")


def _resolve_style(severity: Optional[str]) -> Dict[str, object]:
    if severity and severity.upper() in SEVERITY_OVERLAY_STYLES:
        return SEVERITY_OVERLAY_STYLES[severity.upper()]
    return {
        "color": NEUTRAL_OVERLAY_COLOR_RGB,
        "thickness": 2,
        "dashed": False,
        "double": False,
        "tag": "",
    }


def _draw_dashed_line(
    canvas: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int,
    dash_len: int = 4,
    gap_len: int = 3,
) -> None:
    """Draw a dashed line segment for colour-vision-accessible LOW severity styling."""
    x1, y1 = pt1
    x2, y2 = pt2
    dist = float(np.hypot(x2 - x1, y2 - y1))
    if dist < 1.0:
        return
    step = dash_len + gap_len
    num_steps = max(1, int(dist // step) + 1)
    for i in range(num_steps):
        start_frac = (i * step) / dist
        end_frac = min(1.0, (i * step + dash_len) / dist)
        if start_frac >= 1.0:
            break
        sx = int(round(x1 + (x2 - x1) * start_frac))
        sy = int(round(y1 + (y2 - y1) * start_frac))
        ex = int(round(x1 + (x2 - x1) * end_frac))
        ey = int(round(y1 + (y2 - y1) * end_frac))
        cv2.line(canvas, (sx, sy), (ex, ey), color, thickness=thickness)


def _draw_styled_box(
    canvas: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    style: Dict[str, object],
) -> None:
    """Draw a bounding box using severity-specific colour, thickness, and dash/double style."""
    color = style["color"]  # type: ignore[assignment]
    thickness = int(style["thickness"])  # type: ignore[arg-type]
    dashed = bool(style["dashed"])
    double = bool(style["double"])
    img_h, img_w = canvas.shape[:2]

    if dashed:
        _draw_dashed_line(canvas, (x1, y1), (x2, y1), color, thickness)
        _draw_dashed_line(canvas, (x2, y1), (x2, y2), color, thickness)
        _draw_dashed_line(canvas, (x2, y2), (x1, y2), color, thickness)
        _draw_dashed_line(canvas, (x1, y2), (x1, y1), color, thickness)
    else:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness=thickness)
        if double:
            ox1 = max(0, x1 - 3)
            oy1 = max(0, y1 - 3)
            ox2 = min(img_w - 1, x2 + 3)
            oy2 = min(img_h - 1, y2 + 3)
            cv2.rectangle(canvas, (ox1, oy1), (ox2, oy2), color, thickness=1)


def _draw_region_contours_on_canvas(
    canvas: np.ndarray,
    region: Region,
    scale_factor: float,
    style: Dict[str, object],
) -> None:
    """Draw all external contours of a region (including merged segments) onto canvas."""
    img_h, img_w = canvas.shape[:2]
    color = style["color"]  # type: ignore[assignment]
    thickness = int(style["thickness"])  # type: ignore[arg-type]

    contour_list = region.contours if region.contours else ([region.contour] if region.contour else [])
    cv_contours = []
    for cnt in contour_list:
        if not cnt:
            continue
        scaled_pts = np.array(
            [
                [
                    [
                        max(0, min(img_w - 1, int(round(px / scale_factor)))),
                        max(0, min(img_h - 1, int(round(py / scale_factor)))),
                    ]
                ]
                for px, py in cnt
            ],
            dtype=np.int32,
        )
        cv_contours.append(scaled_pts)

    if cv_contours:
        cv2.drawContours(canvas, cv_contours, -1, color, thickness=thickness)


def draw_bounding_boxes(
    original_image: np.ndarray,
    regions: List[Region],
    scale_factor: float = 1.0,
    severities: Optional[Dict[int, str]] = None,
) -> np.ndarray:
    """Render accepted region bounding boxes and IDs on a copy of the original image.

    When `severities` ({region_id: "LOW" | "MEDIUM" | "HIGH"}) is provided, renders
    severity-distinct colour, line thickness, dash/double border style, and [L]/[M]/[H] tag.
    """
    canvas = _to_rgb_canvas(original_image)
    img_h, img_w = canvas.shape[:2]

    for region in regions:
        sev = severities.get(region.id) if severities else None
        style = _resolve_style(sev)
        color = style["color"]  # type: ignore[assignment]
        tag = str(style["tag"])

        (ox, oy, ow, oh), _ = map_region_to_original_coords(region.bbox, region.centroid, scale_factor)
        x1 = max(0, min(img_w - 1, ox))
        y1 = max(0, min(img_h - 1, oy))
        x2 = max(0, min(img_w - 1, ox + ow))
        y2 = max(0, min(img_h - 1, oy + oh))

        _draw_styled_box(canvas, x1, y1, x2, y2, style)
        label = f"#{region.id} [{tag}]" if tag else f"#{region.id}"
        label_y = max(12, y1 - 5)
        cv2.putText(
            canvas,
            label,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    return canvas


def draw_contours(
    original_image: np.ndarray,
    regions: List[Region],
    scale_factor: float = 1.0,
    severities: Optional[Dict[int, str]] = None,
) -> np.ndarray:
    """Render accepted region external contours on a copy of the original image."""
    canvas = _to_rgb_canvas(original_image)

    for region in regions:
        sev = severities.get(region.id) if severities else None
        style = _resolve_style(sev)
        _draw_region_contours_on_canvas(canvas, region, scale_factor, style)

    return canvas


def draw_final_overlay(
    original_image: np.ndarray,
    regions: List[Region],
    scale_factor: float = 1.0,
    severities: Optional[Dict[int, str]] = None,
) -> np.ndarray:
    """Render combined final overlay: original image + severity-styled contours, boxes, and IDs."""
    canvas = _to_rgb_canvas(original_image)
    img_h, img_w = canvas.shape[:2]

    for region in regions:
        sev = severities.get(region.id) if severities else None
        style = _resolve_style(sev)
        color = style["color"]  # type: ignore[assignment]
        tag = str(style["tag"])

        # 1. Contours first
        _draw_region_contours_on_canvas(canvas, region, scale_factor, style)

        # 2. Severity-styled bounding box + ID badge
        (ox, oy, ow, oh), _ = map_region_to_original_coords(region.bbox, region.centroid, scale_factor)
        x1 = max(0, min(img_w - 1, ox))
        y1 = max(0, min(img_h - 1, oy))
        x2 = max(0, min(img_w - 1, ox + ow))
        y2 = max(0, min(img_h - 1, oy + oh))

        _draw_styled_box(canvas, x1, y1, x2, y2, style)
        label = f"#{region.id} [{tag}]" if tag else f"#{region.id}"
        label_y = max(12, y1 - 5)
        cv2.putText(
            canvas,
            label,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    return canvas
