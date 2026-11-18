"""Noise filtering and false-positive reduction for segmented defect candidates.

Evaluates each candidate region against configurable, physics-grounded rules and
assigns every region an explicit accept or reject decision with a human-readable
reason string containing the measured values.
"""

from dataclasses import dataclass, field, replace
import logging
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
from pydantic import BaseModel, Field

from app.cv.segmentation import Region

logger = logging.getLogger("waferinspect.cv.noise_filter")


class NoiseFilterConfig(BaseModel):
    """Configurable rules for distinguishing genuine wafer defects from sensor noise."""

    min_area_px: int = Field(
        default=12,
        ge=1,
        description="Minimum candidate area in pixels; eliminates 1-3 px dust specks and residual noise.",
    )
    max_area_px: Optional[int] = Field(
        default=None,
        ge=1,
        description="Optional maximum candidate area in pixels (None disables upper area cutoff).",
    )
    contrast_sigma_k: float = Field(
        default=3.0,
        ge=0.0,
        description=(
            "Single statistical constant k multiplying the image's MAD-derived robust noise sigma "
            "to derive min_mean_contrast (min_mean_contrast = k * noise_sigma)."
        ),
    )
    min_solidity: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional minimum solidity (area / convex hull area). Default off (None).",
    )
    # WHY max_aspect_ratio is DEFAULT OFF:
    # Micro-scratches are critical real wafer defects caused by CMP slurry particles or handling
    # contact, and they naturally exhibit extreme aspect ratios (often > 20:1). Enabling an aspect
    # ratio filter by default would erroneously reject genuine scratch defects.
    max_aspect_ratio: Optional[float] = Field(
        default=None,
        gt=0.0,
        description=(
            "Optional maximum oriented aspect ratio. DEFAULT OFF (None) because real wafer "
            "scratches have extreme aspect ratios and would be deleted by a default aspect filter."
        ),
    )


@dataclass
class RejectedRegion:
    """Record of a candidate region rejected by noise filtering, with measured metrics and reason."""

    id: int
    area_px: int
    mean_contrast: float
    touches_edge: bool
    rule: str
    reason: str


@dataclass
class NoiseFilterResult:
    """Complete output of candidate region noise filtering."""

    accepted_regions: List[Region]
    rejected_regions: List[RejectedRegion]
    stage_counts: List[Dict[str, Any]] = field(default_factory=list)
    noise_sigma: float = 0.0
    derived_min_mean_contrast: float = 0.0


def compute_region_solidity(region: Region) -> float:
    """Compute pixel-exact solidity (region area / convex hull pixel area) in (0, 1]."""
    ys, xs = np.nonzero(region.mask)
    if xs.size == 0:
        return 0.0
    if xs.size < 3:
        return 1.0

    pts = np.column_stack((xs, ys)).astype(np.int32)
    hull = cv2.convexHull(pts)
    hull_canvas = np.zeros_like(region.mask, dtype=np.uint8)
    cv2.fillConvexPoly(hull_canvas, hull, 1)
    hull_area = int(np.count_nonzero(hull_canvas))
    if hull_area <= 0:
        return 1.0
    return min(1.0, float(region.area_px) / float(hull_area))


def compute_region_aspect_ratio(region: Region) -> float:
    """Compute oriented aspect ratio using minimum-area bounding rectangle of foreground pixels.

    Handles both axis-aligned and diagonal scratches accurately by using cv2.minAreaRect
    and adding 1.0 pixel to side extents to represent discrete pixel coverage.
    """
    ys, xs = np.nonzero(region.mask)
    if xs.size < 2:
        return 1.0

    pts = np.column_stack((xs, ys)).astype(np.float32)
    _, (rect_w, rect_h), _ = cv2.minAreaRect(pts)
    side_a = float(rect_w) + 1.0
    side_b = float(rect_h) + 1.0
    major = max(side_a, side_b)
    minor = max(1.0, min(side_a, side_b))
    return major / minor


def filter_regions(
    regions: List[Region],
    noise_sigma: float,
    config: Optional[NoiseFilterConfig] = None,
) -> NoiseFilterResult:
    """Filter segmented candidate regions into accepted defects and rejected noise with reasons.

    Does not mutate input Region objects. Regions touching the wafer boundary are preserved
    (not rejected for touching the edge) and retain touches_edge=True.

    Args:
        regions: Candidate Region list produced by segment_regions.
        noise_sigma: Image-level robust noise sigma derived from Phase 4 MAD estimator.
        config: NoiseFilterConfig parameters. Uses defaults if omitted.

    Returns:
        NoiseFilterResult containing accepted regions, rejected regions with reasons,
        and per-rule stage count entries for rules that removed at least one region.
    """
    if config is None:
        config = NoiseFilterConfig()

    safe_sigma = max(0.0, float(noise_sigma))
    derived_min_contrast = round(float(config.contrast_sigma_k) * safe_sigma, 4)

    surviving: List[Region] = [replace(r, mask=r.mask.copy(), contour=list(r.contour)) for r in regions]
    rejected: List[RejectedRegion] = []
    stage_counts: List[Dict[str, Any]] = []

    # 1. Size Filter (min_area_px and optional max_area_px)
    next_surviving: List[Region] = []
    size_removed = 0
    for r in surviving:
        if r.area_px < config.min_area_px:
            size_removed += 1
            rejected.append(
                RejectedRegion(
                    id=r.id,
                    area_px=r.area_px,
                    mean_contrast=r.mean_contrast,
                    touches_edge=r.touches_edge,
                    rule="size_filter",
                    reason=f"Rejected by size filter: area={r.area_px} px < min_area_px={config.min_area_px} px",
                )
            )
        elif config.max_area_px is not None and r.area_px > config.max_area_px:
            size_removed += 1
            rejected.append(
                RejectedRegion(
                    id=r.id,
                    area_px=r.area_px,
                    mean_contrast=r.mean_contrast,
                    touches_edge=r.touches_edge,
                    rule="size_filter",
                    reason=f"Rejected by size filter: area={r.area_px} px > max_area_px={config.max_area_px} px",
                )
            )
        else:
            next_surviving.append(r)

    surviving = next_surviving
    if size_removed > 0:
        stage_counts.append({"stage": "after_size_filter", "candidate_count": len(surviving)})

    # 2. Contrast Filter (mean absolute signed difference vs noise-derived limit)
    next_surviving = []
    contrast_removed = 0
    for r in surviving:
        if r.mean_contrast < derived_min_contrast:
            contrast_removed += 1
            rejected.append(
                RejectedRegion(
                    id=r.id,
                    area_px=r.area_px,
                    mean_contrast=r.mean_contrast,
                    touches_edge=r.touches_edge,
                    rule="contrast_filter",
                    reason=(
                        f"Rejected by contrast filter: mean_contrast={r.mean_contrast:.2f} < "
                        f"noise-derived limit {derived_min_contrast:.2f} "
                        f"(k={config.contrast_sigma_k} * sigma={safe_sigma:.2f})"
                    ),
                )
            )
        else:
            next_surviving.append(r)

    surviving = next_surviving
    if contrast_removed > 0:
        stage_counts.append({"stage": "after_contrast_filter", "candidate_count": len(surviving)})

    # 3. Optional Solidity Filter (default off)
    if config.min_solidity is not None:
        next_surviving = []
        solidity_removed = 0
        for r in surviving:
            solidity = compute_region_solidity(r)
            if solidity < config.min_solidity:
                solidity_removed += 1
                rejected.append(
                    RejectedRegion(
                        id=r.id,
                        area_px=r.area_px,
                        mean_contrast=r.mean_contrast,
                        touches_edge=r.touches_edge,
                        rule="solidity_filter",
                        reason=(
                            f"Rejected by solidity filter: solidity={solidity:.3f} < "
                            f"min_solidity={config.min_solidity:.3f}"
                        ),
                    )
                )
            else:
                next_surviving.append(r)

        surviving = next_surviving
        if solidity_removed > 0:
            stage_counts.append({"stage": "after_solidity_filter", "candidate_count": len(surviving)})

    # 4. Optional Aspect Ratio Filter (DEFAULT OFF to preserve real scratches)
    if config.max_aspect_ratio is not None:
        next_surviving = []
        aspect_removed = 0
        for r in surviving:
            aspect = compute_region_aspect_ratio(r)
            if aspect > config.max_aspect_ratio:
                aspect_removed += 1
                rejected.append(
                    RejectedRegion(
                        id=r.id,
                        area_px=r.area_px,
                        mean_contrast=r.mean_contrast,
                        touches_edge=r.touches_edge,
                        rule="aspect_ratio_filter",
                        reason=(
                            f"Rejected by aspect ratio filter: aspect_ratio={aspect:.2f} > "
                            f"max_aspect_ratio={config.max_aspect_ratio:.2f}"
                        ),
                    )
                )
            else:
                next_surviving.append(r)

        surviving = next_surviving
        if aspect_removed > 0:
            stage_counts.append({"stage": "after_aspect_ratio_filter", "candidate_count": len(surviving)})

    # Annotate accepted regions with explicit acceptance decision and measured metrics
    accepted: List[Region] = []
    for r in surviving:
        edge_note = " (touches wafer edge)" if r.touches_edge else ""
        r.decision = "accepted"
        r.decision_reason = (
            f"Accepted: area={r.area_px} px >= {config.min_area_px} px, "
            f"mean_contrast={r.mean_contrast:.2f} >= {derived_min_contrast:.2f}{edge_note}"
        )
        accepted.append(r)

    # Keep rejected list deterministically ordered by region id
    rejected.sort(key=lambda rr: rr.id)

    return NoiseFilterResult(
        accepted_regions=accepted,
        rejected_regions=rejected,
        stage_counts=stage_counts,
        noise_sigma=round(safe_sigma, 4),
        derived_min_mean_contrast=round(derived_min_contrast, 2),
    )
