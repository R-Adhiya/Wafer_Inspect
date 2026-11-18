"""Defect segmentation, connected-component analysis, contour extraction, and overlays.

Extracts discrete candidate defect regions from a binary morphology mask using
8-connected component analysis, builds memory-efficient cropped boolean masks and
external contours, and provides coordinate mapping and neutral overlay rendering.
"""

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple
import cv2
import numpy as np

# Single neutral accent color (RGB) for Phase 5 overlays.
# Severity-based color coding is introduced in Phase 6.
NEUTRAL_OVERLAY_COLOR_RGB = (59, 130, 246)


@dataclass
class Region:
    """Discrete segmented defect candidate region in processed-image coordinates."""

    id: int
    bbox: Tuple[int, int, int, int]  # (x, y, w, h) in processed-image pixels
    area_px: int
    centroid: Tuple[float, float]  # (cx, cy) in processed-image pixels
    mask: np.ndarray  # Cropped 2D boolean mask of shape (h, w) to save memory
    contour: List[Tuple[int, int]]  # External contour points [(x, y), ...]
    polarity: Literal["bright", "dark"]
    mean_contrast: float
    peak_contrast: float
    touches_edge: bool
    scale_factor: float = 1.0
    decision: Literal["pending", "accepted", "rejected"] = "pending"
    decision_reason: str = ""

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
) -> List[Region]:
    """Segment 8-connected components from a binary mask into deterministic Region objects.

    Does not mutate input arrays. External contours are extracted with cv2.RETR_EXTERNAL
    (interior holes are ignored in the contour list, while area_px retains the exact
    foreground pixel count from connectedComponentsWithStats).

    Region IDs are assigned deterministically (1..N) after sorting by centroid (y, x).

    Args:
        morphology_mask: 2D uint8 binary mask (0 or 255) from Phase 4 detection.
        signed_difference: 2D int16 signed difference array (test - ref or test - bg).
        wafer_mask: Optional 2D uint8 wafer domain mask for detecting rim-touching defects.
        scale_factor: Scale factor from preprocessing resize step (default 1.0).

    Returns:
        List[Region]: Deterministically ordered list of candidate regions.
    """
    if not isinstance(morphology_mask, np.ndarray) or morphology_mask.ndim != 2:
        raise ValueError("morphology_mask must be a 2D NumPy array.")
    if not isinstance(signed_difference, np.ndarray) or signed_difference.shape != morphology_mask.shape:
        raise ValueError("signed_difference must be a 2D NumPy array matching morphology_mask shape.")
    if scale_factor <= 0:
        raise ValueError(f"scale_factor must be positive, got {scale_factor}")

    binary = (morphology_mask > 0).astype(np.uint8) * 255
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    if num_labels <= 1:
        return []

    edge_boundary = _compute_edge_boundary_mask(binary.shape[:2], wafer_mask)
    raw_regions: List[Region] = []

    for label_idx in range(1, num_labels):
        x = int(stats[label_idx, cv2.CC_STAT_LEFT])
        y = int(stats[label_idx, cv2.CC_STAT_TOP])
        w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        area_px = int(stats[label_idx, cv2.CC_STAT_AREA])
        cx = float(centroids[label_idx][0])
        cy = float(centroids[label_idx][1])

        # Cropped boolean mask (memory efficient vs storing full-frame masks)
        cropped_bool = (labels[y : y + h, x : x + w] == label_idx).copy()

        # External contour only (holes ignored; area_px stays true pixel count)
        cropped_u8 = cropped_bool.astype(np.uint8) * 255
        contours, _ = cv2.findContours(cropped_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            ext_cnt = max(contours, key=cv2.contourArea)
            contour_pts = [(int(pt[0][0]) + x, int(pt[0][1]) + y) for pt in ext_cnt]
        else:
            contour_pts = []

        # Signed contrast & polarity inside the component mask
        signed_crop = signed_difference[y : y + h, x : x + w][cropped_bool].astype(np.float32)
        mean_signed = float(np.mean(signed_crop)) if signed_crop.size > 0 else 0.0
        polarity: Literal["bright", "dark"] = "bright" if mean_signed >= 0.0 else "dark"

        abs_signed = np.abs(signed_crop)
        mean_contrast = float(np.mean(abs_signed)) if abs_signed.size > 0 else 0.0
        peak_contrast = float(np.max(abs_signed)) if abs_signed.size > 0 else 0.0

        # Flag regions touching wafer rim boundary or image border
        touches_edge = bool(np.any(cropped_bool & edge_boundary[y : y + h, x : x + w]))

        raw_regions.append(
            Region(
                id=0,  # Assigned deterministically after sorting
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
            )
        )

    # Deterministic ordering: sorted by centroid y, then centroid x (tie-broken by bbox)
    raw_regions.sort(key=lambda r: (r.centroid[1], r.centroid[0], r.bbox[1], r.bbox[0]))
    for idx, region in enumerate(raw_regions, start=1):
        region.id = idx

    return raw_regions


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


def draw_bounding_boxes(
    original_image: np.ndarray,
    regions: List[Region],
    scale_factor: float = 1.0,
) -> np.ndarray:
    """Render accepted region bounding boxes and IDs on a copy of the original image.

    Uses a single neutral color (severity coloring is deferred to Phase 6).
    Maps bounding boxes back to original-image pixel coordinates using scale_factor.
    """
    canvas = _to_rgb_canvas(original_image)
    img_h, img_w = canvas.shape[:2]
    thickness = max(1, int(round(min(img_h, img_w) / 300.0)))

    for region in regions:
        (ox, oy, ow, oh), _ = map_region_to_original_coords(region.bbox, region.centroid, scale_factor)
        x1 = max(0, min(img_w - 1, ox))
        y1 = max(0, min(img_h - 1, oy))
        x2 = max(0, min(img_w - 1, ox + ow))
        y2 = max(0, min(img_h - 1, oy + oh))

        cv2.rectangle(canvas, (x1, y1), (x2, y2), NEUTRAL_OVERLAY_COLOR_RGB, thickness=thickness)
        label_y = max(12, y1 - 4)
        cv2.putText(
            canvas,
            f"#{region.id}",
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            NEUTRAL_OVERLAY_COLOR_RGB,
            1,
            cv2.LINE_AA,
        )

    return canvas


def draw_contours(
    original_image: np.ndarray,
    regions: List[Region],
    scale_factor: float = 1.0,
) -> np.ndarray:
    """Render accepted region external contours on a copy of the original image.

    Uses a single neutral color (severity coloring is deferred to Phase 6).
    Scales contour coordinates back to original-image pixel coordinates.
    """
    canvas = _to_rgb_canvas(original_image)
    img_h, img_w = canvas.shape[:2]
    thickness = max(1, int(round(min(img_h, img_w) / 300.0)))

    cv_contours = []
    for region in regions:
        if not region.contour:
            continue
        scaled_pts = np.array(
            [
                [
                    [
                        max(0, min(img_w - 1, int(round(px / scale_factor)))),
                        max(0, min(img_h - 1, int(round(py / scale_factor)))),
                    ]
                ]
                for px, py in region.contour
            ],
            dtype=np.int32,
        )
        cv_contours.append(scaled_pts)

    if cv_contours:
        cv2.drawContours(canvas, cv_contours, -1, NEUTRAL_OVERLAY_COLOR_RGB, thickness=thickness)

    return canvas
