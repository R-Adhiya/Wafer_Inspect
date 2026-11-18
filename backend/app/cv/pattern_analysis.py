"""Wafer-level spatial defect pattern diagnosis with Monte Carlo significance testing.

Pure functions operating on defect measurements and wafer geometry mapped to ORIGINAL
image pixels. No filesystem I/O, no global state, never mutates inputs.

IMPORTANT INTERPRETATION NOTE:
`confidence` is defined as `1 - p_value` of the winning pattern (or the Monte Carlo
evidence strength against uniform random spatial placement in the wafer disc).
It is **rule-based statistical evidence strength, NOT a calibrated machine-learning
posterior probability**. A non-random pattern is reported ONLY when both its geometric
rule passes AND its seeded Monte Carlo `p_value <= p_value_threshold` (default 0.01);
otherwise the wafer is classified as `RANDOM_SCATTER`.
"""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union
import cv2
import numpy as np
from pydantic import BaseModel, Field

from app.cv.measurement import RegionMeasurement
from app.cv.segmentation import SEVERITY_OVERLAY_STYLES, _to_rgb_canvas

PatternType = Literal[
    "INSUFFICIENT_DATA",
    "EDGE_RING",
    "CENTER_CLUSTER",
    "DONUT",
    "LOCAL_CLUSTER",
    "SCRATCH_LINE",
    "RANDOM_SCATTER",
]

# Deterministic tie-break priority when multiple pattern rules pass with p <= threshold
PATTERN_TIE_BREAK_ORDER: Tuple[str, ...] = (
    "SCRATCH_LINE",
    "EDGE_RING",
    "CENTER_CLUSTER",
    "DONUT",
    "LOCAL_CLUSTER",
)


class PatternConfig(BaseModel):
    """Configurable thresholds and Monte Carlo settings for wafer spatial pattern diagnosis."""

    min_defects: int = Field(
        default=3,
        ge=1,
        description="Minimum defect count required for multi-point spatial statistics (default 3).",
    )
    single_scratch_min_elongation: float = Field(
        default=8.0,
        gt=1.0,
        description="Minimum elongation (length/thickness) for single-region shape-based SCRATCH_LINE.",
    )
    single_scratch_min_diameter_ratio: float = Field(
        default=0.15,
        gt=0.0,
        le=1.0,
        description="Minimum defect length as fraction of wafer diameter for shape-based SCRATCH_LINE.",
    )
    band_min_fraction: float = Field(
        default=0.70,
        gt=0.0,
        le=1.0,
        description="Minimum fraction of defects inside target radial band for EDGE_RING / CENTER_CLUSTER / DONUT.",
    )
    edge_ring_r_min: float = Field(
        default=0.80,
        gt=0.0,
        lt=1.0,
        description="Minimum normalized radius r for the EDGE_RING outer band (default 0.80).",
    )
    center_cluster_r_max: float = Field(
        default=0.30,
        gt=0.0,
        lt=1.0,
        description="Maximum normalized radius r for CENTER_CLUSTER inner disk (default 0.30).",
    )
    donut_r_min: float = Field(
        default=0.40,
        gt=0.0,
        lt=1.0,
        description="Inner normalized radius r for DONUT annulus (default 0.40).",
    )
    donut_r_max: float = Field(
        default=0.70,
        gt=0.0,
        lt=1.0,
        description="Outer normalized radius r for DONUT annulus (default 0.70).",
    )
    max_angular_gap_deg: float = Field(
        default=150.0,
        gt=0.0,
        le=360.0,
        description="Maximum allowed empty angular gap (degrees) around the wafer for EDGE_RING and DONUT.",
    )
    scratch_min_explained_variance: float = Field(
        default=0.90,
        gt=0.5,
        le=1.0,
        description="Minimum TLS/PCA first-component explained variance ratio for SCRATCH_LINE.",
    )
    scratch_min_span_ratio: float = Field(
        default=0.40,
        gt=0.0,
        le=1.0,
        description="Minimum centroid span along fitted line as fraction of wafer diameter for SCRATCH_LINE.",
    )
    local_cluster_max_nn_ratio: float = Field(
        default=0.60,
        gt=0.0,
        lt=2.0,
        description="Maximum Clark-Evans nearest-neighbour ratio R for LOCAL_CLUSTER (default 0.60).",
    )
    p_value_threshold: float = Field(
        default=0.01,
        gt=0.0,
        le=0.5,
        description="Maximum Monte Carlo p-value required to report a non-random pattern (default 0.01).",
    )
    mc_trials: int = Field(
        default=2000,
        ge=100,
        description="Number of uniform-in-disc Monte Carlo simulation trials per diagnosis.",
    )
    mc_seed: int = Field(
        default=42,
        description="Deterministic RNG seed for Monte Carlo significance testing.",
    )
    radial_bins: int = Field(
        default=5,
        ge=2,
        le=20,
        description="Number of equal-width radial bins [0..1] reported for radial distribution charts.",
    )


@dataclass
class PatternResult:
    """Result of wafer-level spatial pattern diagnosis."""

    type: PatternType
    confidence: float  # Rule-based evidence strength (1 - p_value for detected patterns), NOT calibrated ML probability
    p_value: Optional[float]
    explanation: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    supporting_defect_ids: List[int] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def _extract_defect_record(
    item: Union[RegionMeasurement, Dict[str, Any]],
    fallback_id: int,
) -> Dict[str, Any]:
    """Normalize a RegionMeasurement or lightweight dict into a uniform measurement dict."""
    if isinstance(item, RegionMeasurement):
        return {
            "id": item.id,
            "centroid": (float(item.centroid[0]), float(item.centroid[1])),
            "area_px": int(item.area_px),
            "length_px": float(item.length_px),
            "elongation": float(item.elongation),
            "severity": item.severity,
        }
    return {
        "id": int(item.get("id", fallback_id)),
        "centroid": (float(item["centroid"][0]), float(item["centroid"][1])),
        "area_px": int(item.get("area_px", 0)),
        "length_px": float(item.get("length_px", 0.0)),
        "elongation": float(item.get("elongation", 1.0)),
        "severity": str(item.get("severity", "LOW")),
    }


def _largest_angular_gap_deg(angles_deg: np.ndarray) -> float:
    """Compute the largest empty angular gap in degrees on [0, 360) for a 1D array of angles."""
    if angles_deg.size < 2:
        return 360.0
    sorted_a = np.sort(angles_deg % 360.0)
    diffs = np.diff(sorted_a)
    wrap = (360.0 - float(sorted_a[-1])) + float(sorted_a[0])
    return float(max(float(np.max(diffs)), wrap))


def _compute_tls_line_metrics(
    pts_xy: np.ndarray,
    wafer_diameter: float,
) -> Tuple[float, float, Tuple[float, float], Tuple[float, float]]:
    """Total-least-squares (PCA) line fit through 2D centroids.

    Returns:
        (explained_variance, span_ratio, start_pt, end_pt)
    """
    n = pts_xy.shape[0]
    if n < 2:
        p = (float(pts_xy[0, 0]), float(pts_xy[0, 1])) if n == 1 else (0.0, 0.0)
        return 0.0, 0.0, p, p

    mean_pt = np.mean(pts_xy, axis=0)
    centered = pts_xy - mean_pt
    cov = centered.T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)
    # eigh returns ascending order: eigvals[1] >= eigvals[0]
    lam1 = max(0.0, float(eigvals[1]))
    lam2 = max(0.0, float(eigvals[0]))
    total_var = lam1 + lam2
    if total_var <= 1e-12:
        p = (float(mean_pt[0]), float(mean_pt[1]))
        return 0.0, 0.0, p, p

    explained_var = lam1 / total_var
    axis = eigvecs[:, 1]
    proj = centered @ axis
    t_min = float(np.min(proj))
    t_max = float(np.max(proj))
    span_px = t_max - t_min
    span_ratio = span_px / max(wafer_diameter, 1e-6)

    p_start = mean_pt + t_min * axis
    p_end = mean_pt + t_max * axis
    return (
        explained_var,
        span_ratio,
        (round(float(p_start[0]), 2), round(float(p_start[1]), 2)),
        (round(float(p_end[0]), 2), round(float(p_end[1]), 2)),
    )


def _compute_clark_evans_nn_ratio(pts_xy: np.ndarray, wafer_radius: float) -> float:
    """Compute Clark-Evans nearest-neighbour ratio R = mean_NN / (0.5 / sqrt(n / wafer_area))."""
    n = pts_xy.shape[0]
    if n < 2 or wafer_radius <= 0:
        return 1.0

    diff = pts_xy[:, None, :] - pts_xy[None, :, :]
    dist_mat = np.hypot(diff[:, :, 0], diff[:, :, 1])
    np.fill_diagonal(dist_mat, np.inf)
    nn_dists = np.min(dist_mat, axis=1)
    obs_mean_nn = float(np.mean(nn_dists))

    wafer_area = math.pi * (wafer_radius ** 2)
    expected_nn = 0.5 / math.sqrt(float(n) / wafer_area)
    return obs_mean_nn / max(expected_nn, 1e-9)


def _compute_radial_bins(r_norm: np.ndarray, num_bins: int = 5) -> List[Dict[str, Any]]:
    """Compute observed defect count vs expected uniform-random count per radial bin."""
    n = int(r_norm.size)
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    bins: List[Dict[str, Any]] = []
    for i in range(num_bins):
        r0 = float(edges[i])
        r1 = float(edges[i + 1])
        if i == num_bins - 1:
            # Include any slight r > 1.0 rim centroid in the outermost bin
            obs = int(np.count_nonzero(r_norm >= r0))
        else:
            obs = int(np.count_nonzero((r_norm >= r0) & (r_norm < r1)))
        area_frac = (r1 ** 2) - (r0 ** 2)
        expected = round(float(n) * area_frac, 2)
        bins.append(
            {
                "bin": f"{r0:.1f}-{r1:.1f}",
                "r_min": round(r0, 3),
                "r_max": round(r1, 3),
                "observed": obs,
                "expected_random": expected,
            }
        )
    return bins


def _mc_largest_angular_gaps_deg(
    theta_deg: np.ndarray,
    in_band_mask: np.ndarray,
) -> np.ndarray:
    """Vectorized largest empty angular gap (degrees) per MC trial for points inside `in_band_mask`."""
    num_trials = theta_deg.shape[0]
    counts = np.sum(in_band_mask, axis=1)
    gaps = np.full(num_trials, 360.0, dtype=np.float64)

    # Sort masked angles by pushing out-of-band entries to +inf
    masked_angles = np.where(in_band_mask, theta_deg, np.inf)
    sorted_angles = np.sort(masked_angles, axis=1)

    valid_trials = counts >= 2
    if not np.any(valid_trials):
        return gaps

    idx = np.nonzero(valid_trials)[0]
    s_valid = sorted_angles[idx]
    c_valid = counts[idx]

    # Adjacent differences for finite valid entries (avoids inf - inf RuntimeWarning)
    left = s_valid[:, :-1]
    right = s_valid[:, 1:]
    valid_pairs = np.isfinite(left) & np.isfinite(right)
    safe_left = np.where(valid_pairs, left, 0.0)
    safe_right = np.where(valid_pairs, right, 0.0)
    finite_diffs = np.where(valid_pairs, safe_right - safe_left, -1.0)
    max_internal = np.max(finite_diffs, axis=1)

    first_angle = s_valid[:, 0]
    last_angle = s_valid[np.arange(idx.size), c_valid - 1]
    wrap_gap = (360.0 - last_angle) + first_angle
    gaps[idx] = np.maximum(max_internal, wrap_gap)
    return gaps


def _run_monte_carlo_pvalues(
    n: int,
    obs_edge_frac: float,
    obs_center_frac: float,
    obs_donut_frac: float,
    obs_ev: float,
    obs_nn_ratio: float,
    config: PatternConfig,
) -> Dict[str, float]:
    """Estimate seeded Monte Carlo p-values for all 5 spatial patterns under uniform-in-disc H0.

    Draws `config.mc_trials` configurations of `n` points uniformly in the unit disc
    and computes Phipson-Smyth smoothed p-values `(hits + 1) / (T + 1)` so finite-sample
    p-values are strictly in (0, 1].
    """
    T = int(config.mc_trials)
    rng = np.random.default_rng(config.mc_seed)

    # Uniform random points in the unit disc (radius = 1, area = pi, diameter = 2)
    r_mc = np.sqrt(rng.uniform(0.0, 1.0, size=(T, n)))
    theta_rad = rng.uniform(0.0, 2.0 * math.pi, size=(T, n))
    theta_deg = np.degrees(theta_rad)
    x_mc = r_mc * np.cos(theta_rad)
    y_mc = r_mc * np.sin(theta_rad)

    # 1. EDGE_RING: fraction in r >= edge_ring_r_min AND largest angular gap <= max_angular_gap_deg
    edge_mask = r_mc >= config.edge_ring_r_min
    edge_frac_mc = np.mean(edge_mask, axis=1)
    edge_gap_mc = _mc_largest_angular_gaps_deg(theta_deg, edge_mask)
    edge_hits = int(
        np.count_nonzero(
            (edge_frac_mc >= obs_edge_frac - 1e-12)
            & (edge_gap_mc <= config.max_angular_gap_deg + 1e-12)
        )
    )
    p_edge = (edge_hits + 1.0) / (T + 1.0)

    # 2. CENTER_CLUSTER: fraction in r <= center_cluster_r_max
    center_frac_mc = np.mean(r_mc <= config.center_cluster_r_max, axis=1)
    center_hits = int(np.count_nonzero(center_frac_mc >= obs_center_frac - 1e-12))
    p_center = (center_hits + 1.0) / (T + 1.0)

    # 3. DONUT: fraction in [donut_r_min, donut_r_max] AND largest angular gap <= max_angular_gap_deg
    donut_mask = (r_mc >= config.donut_r_min) & (r_mc <= config.donut_r_max)
    donut_frac_mc = np.mean(donut_mask, axis=1)
    donut_gap_mc = _mc_largest_angular_gaps_deg(theta_deg, donut_mask)
    donut_hits = int(
        np.count_nonzero(
            (donut_frac_mc >= obs_donut_frac - 1e-12)
            & (donut_gap_mc <= config.max_angular_gap_deg + 1e-12)
        )
    )
    p_donut = (donut_hits + 1.0) / (T + 1.0)

    # 4. SCRATCH_LINE: closed-form 2x2 covariance eigenvalues + span along principal axis
    cx_mc = x_mc - np.mean(x_mc, axis=1, keepdims=True)
    cy_mc = y_mc - np.mean(y_mc, axis=1, keepdims=True)
    sxx = np.sum(cx_mc * cx_mc, axis=1)
    syy = np.sum(cy_mc * cy_mc, axis=1)
    sxy = np.sum(cx_mc * cy_mc, axis=1)

    trace = sxx + syy
    disc = np.sqrt(np.maximum(0.0, (sxx - syy) ** 2 + 4.0 * (sxy ** 2)))
    lam1 = 0.5 * (trace + disc)
    ev_mc = np.divide(lam1, np.maximum(trace, 1e-12))

    # Principal eigenvector angle = 0.5 * atan2(2*sxy, sxx - syy)
    phi = 0.5 * np.arctan2(2.0 * sxy, sxx - syy)
    ux = np.cos(phi)[:, None]
    uy = np.sin(phi)[:, None]
    proj = cx_mc * ux + cy_mc * uy
    span_ratio_mc = (np.max(proj, axis=1) - np.min(proj, axis=1)) / 2.0  # unit disc diameter == 2.0
    scratch_hits = int(
        np.count_nonzero(
            (ev_mc >= obs_ev - 1e-12)
            & (span_ratio_mc >= config.scratch_min_span_ratio - 1e-12)
        )
    )
    p_scratch = (scratch_hits + 1.0) / (T + 1.0)

    # 5. LOCAL_CLUSTER: Clark-Evans nearest-neighbour ratio in unit disc (area = pi)
    # Pairwise squared distances: (T, n, n)
    dx = x_mc[:, :, None] - x_mc[:, None, :]
    dy = y_mc[:, :, None] - y_mc[:, None, :]
    d2 = dx * dx + dy * dy
    eye = np.eye(n, dtype=bool)[None, :, :]
    d2 = np.where(eye, np.inf, d2)
    mean_nn_mc = np.mean(np.sqrt(np.min(d2, axis=2)), axis=1)
    expected_nn_unit = 0.5 / math.sqrt(float(n) / math.pi)
    nn_ratio_mc = mean_nn_mc / expected_nn_unit
    local_hits = int(np.count_nonzero(nn_ratio_mc <= obs_nn_ratio + 1e-12))
    p_local = (local_hits + 1.0) / (T + 1.0)

    return {
        "SCRATCH_LINE": round(p_scratch, 5),
        "EDGE_RING": round(p_edge, 5),
        "CENTER_CLUSTER": round(p_center, 5),
        "DONUT": round(p_donut, 5),
        "LOCAL_CLUSTER": round(p_local, 5),
    }


def analyze_spatial_pattern(
    measurements: Sequence[Union[RegionMeasurement, Dict[str, Any]]],
    wafer_center: Tuple[float, float],
    wafer_radius: float,
    image_size: Tuple[int, int],
    scale_factor: float = 1.0,
    is_full_frame: bool = False,
    config: Optional[PatternConfig] = None,
) -> PatternResult:
    """Diagnose wafer-level spatial defect pattern with Monte Carlo significance testing.

    Args:
        measurements: Accepted defect measurements (in ORIGINAL image pixel coordinates).
        wafer_center: (cx, cy) wafer center from detection (in PROCESSED coordinates).
        wafer_radius: Wafer radius from detection (in PROCESSED coordinates).
        image_size: (orig_width, orig_height) of the ORIGINAL image in pixels.
        scale_factor: Phase 3 resize scale factor (processed / original).
        is_full_frame: Whether detection flagged the image as full-frame (no visible wafer rim).
        config: Optional PatternConfig overriding default thresholds and MC parameters.

    Returns:
        PatternResult with type, rule-based evidence strength (`confidence`), `p_value`,
        plain-language `explanation`, detailed `metrics`, `supporting_defect_ids`, and `warnings`.
    """
    if config is None:
        config = PatternConfig()
    if scale_factor <= 0:
        raise ValueError(f"scale_factor must be positive, got {scale_factor}")

    orig_w, orig_h = int(image_size[0]), int(image_size[1])
    warnings: List[str] = []

    if is_full_frame:
        orig_cx = orig_w / 2.0
        orig_cy = orig_h / 2.0
        orig_r = min(orig_w, orig_h) / 2.0
        warnings.append(
            f"assumed wafer geometry: full-frame image without visible rim; "
            f"using inscribed circle center=({orig_cx:.1f}, {orig_cy:.1f}), radius={orig_r:.1f} px."
        )
    else:
        orig_cx = float(wafer_center[0]) / float(scale_factor)
        orig_cy = float(wafer_center[1]) / float(scale_factor)
        orig_r = float(wafer_radius) / float(scale_factor)

    wafer_diam = max(2.0 * orig_r, 1e-6)
    records = [_extract_defect_record(item, idx) for idx, item in enumerate(measurements, start=1)]
    n = len(records)

    pts_xy = (
        np.array([rec["centroid"] for rec in records], dtype=np.float64)
        if n > 0
        else np.zeros((0, 2), dtype=np.float64)
    )
    if n > 0:
        dxy = pts_xy - np.array([[orig_cx, orig_cy]], dtype=np.float64)
        r_norm = np.hypot(dxy[:, 0], dxy[:, 1]) / max(orig_r, 1e-6)
        theta_deg = np.degrees(np.arctan2(dxy[:, 1], dxy[:, 0])) % 360.0
    else:
        r_norm = np.zeros((0,), dtype=np.float64)
        theta_deg = np.zeros((0,), dtype=np.float64)

    radial_bins = _compute_radial_bins(r_norm, num_bins=config.radial_bins)

    exp_edge_frac = round(1.0 - (config.edge_ring_r_min ** 2), 4)
    exp_center_frac = round(config.center_cluster_r_max ** 2, 4)
    exp_donut_frac = round((config.donut_r_max ** 2) - (config.donut_r_min ** 2), 4)

    base_metrics: Dict[str, Any] = {
        "n": n,
        "wafer_center_orig": (round(orig_cx, 2), round(orig_cy, 2)),
        "wafer_radius_orig": round(orig_r, 2),
        "edge_fraction": round(float(np.mean(r_norm >= config.edge_ring_r_min)), 4) if n > 0 else 0.0,
        "center_fraction": round(float(np.mean(r_norm <= config.center_cluster_r_max)), 4) if n > 0 else 0.0,
        "donut_fraction": (
            round(float(np.mean((r_norm >= config.donut_r_min) & (r_norm <= config.donut_r_max))), 4)
            if n > 0
            else 0.0
        ),
        "expected_random_fraction": {
            "edge_ring": exp_edge_frac,
            "center_cluster": exp_center_frac,
            "donut": exp_donut_frac,
        },
        "largest_angular_gap_deg": round(_largest_angular_gap_deg(theta_deg), 2) if n >= 2 else 360.0,
        "edge_angular_gap_deg": (
            round(_largest_angular_gap_deg(theta_deg[r_norm >= config.edge_ring_r_min]), 2)
            if int(np.count_nonzero(r_norm >= config.edge_ring_r_min)) >= 2
            else 360.0
        ),
        "donut_angular_gap_deg": (
            round(
                _largest_angular_gap_deg(
                    theta_deg[(r_norm >= config.donut_r_min) & (r_norm <= config.donut_r_max)]
                ),
                2,
            )
            if int(np.count_nonzero((r_norm >= config.donut_r_min) & (r_norm <= config.donut_r_max))) >= 2
            else 360.0
        ),
        "line_explained_variance": 0.0,
        "line_span_ratio": 0.0,
        "nn_ratio": 1.0,
        "radial_bins": radial_bins,
    }

    # Case 1: Fewer than min_defects (default 3)
    if n < config.min_defects:
        # Shape-evidence exception: single (or <3) region with elongation >= 8 and length >= 15% of wafer diameter
        elongated_matches = [
            rec
            for rec in records
            if rec["elongation"] >= config.single_scratch_min_elongation
            and rec["length_px"] >= config.single_scratch_min_diameter_ratio * wafer_diam
        ]
        if elongated_matches:
            best = max(elongated_matches, key=lambda r: r["length_px"])
            diam_pct = (best["length_px"] / wafer_diam) * 100.0
            base_metrics["shape_scratch_length_px"] = round(best["length_px"], 2)
            base_metrics["shape_scratch_diameter_pct"] = round(diam_pct, 1)
            base_metrics["shape_scratch_elongation"] = round(best["elongation"], 2)
            return PatternResult(
                type="SCRATCH_LINE",
                confidence=0.85,
                p_value=None,
                explanation=(
                    f"Single-region shape evidence: defect #{best['id']} has elongation "
                    f"{best['elongation']:.1f} (>= {config.single_scratch_min_elongation:.1f}) and "
                    f"length {best['length_px']:.1f} px ({diam_pct:.1f}% of wafer diameter >= "
                    f"{config.single_scratch_min_diameter_ratio * 100.0:.0f}%), diagnosed as SCRATCH_LINE."
                ),
                metrics=base_metrics,
                supporting_defect_ids=[rec["id"] for rec in elongated_matches],
                warnings=warnings,
            )

        if n == 0:
            expl = "No defects detected (0 regions); insufficient data for wafer-level spatial pattern diagnosis."
        else:
            expl = (
                f"Only {n} non-elongated defect(s) detected (minimum {config.min_defects} required "
                f"for spatial pattern diagnosis)."
            )
        return PatternResult(
            type="INSUFFICIENT_DATA",
            confidence=0.0,
            p_value=None,
            explanation=expl,
            metrics=base_metrics,
            supporting_defect_ids=[rec["id"] for rec in records],
            warnings=warnings,
        )

    # Case 2: n >= min_defects -> compute full spatial statistics
    edge_mask = r_norm >= config.edge_ring_r_min
    center_mask = r_norm <= config.center_cluster_r_max
    donut_mask = (r_norm >= config.donut_r_min) & (r_norm <= config.donut_r_max)

    edge_count = int(np.count_nonzero(edge_mask))
    center_count = int(np.count_nonzero(center_mask))
    donut_count = int(np.count_nonzero(donut_mask))

    edge_frac = edge_count / float(n)
    center_frac = center_count / float(n)
    donut_frac = donut_count / float(n)

    edge_gap = _largest_angular_gap_deg(theta_deg[edge_mask])
    donut_gap = _largest_angular_gap_deg(theta_deg[donut_mask])

    ev, span_ratio, line_start, line_end = _compute_tls_line_metrics(pts_xy, wafer_diam)
    nn_ratio = _compute_clark_evans_nn_ratio(pts_xy, orig_r)

    base_metrics["line_explained_variance"] = round(ev, 4)
    base_metrics["line_span_ratio"] = round(span_ratio, 4)
    base_metrics["line_endpoints_orig"] = [line_start, line_end]
    base_metrics["nn_ratio"] = round(nn_ratio, 4)

    # Geometric rule checks
    scratch_rule = (
        ev >= config.scratch_min_explained_variance
        and span_ratio >= config.scratch_min_span_ratio
    )
    edge_rule = (
        edge_frac >= config.band_min_fraction
        and edge_gap <= config.max_angular_gap_deg
    )
    center_rule = center_frac >= config.band_min_fraction
    donut_rule = (
        donut_frac >= config.band_min_fraction
        and donut_gap <= config.max_angular_gap_deg
    )
    local_rule = (nn_ratio < config.local_cluster_max_nn_ratio) and (not center_rule)

    rules_passed: Dict[str, bool] = {
        "SCRATCH_LINE": scratch_rule,
        "EDGE_RING": edge_rule,
        "CENTER_CLUSTER": center_rule,
        "DONUT": donut_rule,
        "LOCAL_CLUSTER": local_rule,
    }

    # Seeded Monte Carlo significance testing for all 5 pattern statistics
    p_values = _run_monte_carlo_pvalues(
        n=n,
        obs_edge_frac=edge_frac,
        obs_center_frac=center_frac,
        obs_donut_frac=donut_frac,
        obs_ev=ev,
        obs_nn_ratio=nn_ratio,
        config=config,
    )
    base_metrics["p_values"] = p_values
    base_metrics["rules_passed"] = rules_passed

    # Deterministic tie-break order: SCRATCH_LINE, EDGE_RING, CENTER_CLUSTER, DONUT, LOCAL_CLUSTER
    winner: Optional[str] = None
    for pat in PATTERN_TIE_BREAK_ORDER:
        if rules_passed[pat] and p_values[pat] <= config.p_value_threshold:
            winner = pat
            break

    if winner == "SCRATCH_LINE":
        p_val = p_values["SCRATCH_LINE"]
        return PatternResult(
            type="SCRATCH_LINE",
            confidence=round(1.0 - p_val, 4),
            p_value=p_val,
            explanation=(
                f"{n} defect centroids align along a line with explained variance "
                f"{ev * 100.0:.1f}% spanning {span_ratio * 100.0:.1f}% of the wafer diameter, "
                f"p = {p_val:.4f}."
            ),
            metrics=base_metrics,
            supporting_defect_ids=[rec["id"] for rec in records],
            warnings=warnings,
        )

    if winner == "EDGE_RING":
        p_val = p_values["EDGE_RING"]
        outer_band_pct = round((1.0 - config.edge_ring_r_min) * 100.0)
        return PatternResult(
            type="EDGE_RING",
            confidence=round(1.0 - p_val, 4),
            p_value=p_val,
            explanation=(
                f"{edge_count} of {n} defects lie in the outer {outer_band_pct}% radial band "
                f"(random placement would put {exp_edge_frac * 100.0:.0f}% there), spread around "
                f"the wafer (max gap {edge_gap:.0f} deg), p = {p_val:.4f}."
            ),
            metrics=base_metrics,
            supporting_defect_ids=[records[i]["id"] for i in range(n) if edge_mask[i]],
            warnings=warnings,
        )

    if winner == "CENTER_CLUSTER":
        p_val = p_values["CENTER_CLUSTER"]
        return PatternResult(
            type="CENTER_CLUSTER",
            confidence=round(1.0 - p_val, 4),
            p_value=p_val,
            explanation=(
                f"{center_count} of {n} defects concentrate within r <= {config.center_cluster_r_max:.1f} "
                f"of the wafer center (random placement would put {exp_center_frac * 100.0:.0f}% there), "
                f"p = {p_val:.4f}."
            ),
            metrics=base_metrics,
            supporting_defect_ids=[records[i]["id"] for i in range(n) if center_mask[i]],
            warnings=warnings,
        )

    if winner == "DONUT":
        p_val = p_values["DONUT"]
        return PatternResult(
            type="DONUT",
            confidence=round(1.0 - p_val, 4),
            p_value=p_val,
            explanation=(
                f"{donut_count} of {n} defects lie in the mid-radius annulus "
                f"{config.donut_r_min:.1f} <= r <= {config.donut_r_max:.1f} "
                f"(random placement would put {exp_donut_frac * 100.0:.0f}% there), "
                f"max angular gap {donut_gap:.0f} deg, p = {p_val:.4f}."
            ),
            metrics=base_metrics,
            supporting_defect_ids=[records[i]["id"] for i in range(n) if donut_mask[i]],
            warnings=warnings,
        )

    if winner == "LOCAL_CLUSTER":
        p_val = p_values["LOCAL_CLUSTER"]
        return PatternResult(
            type="LOCAL_CLUSTER",
            confidence=round(1.0 - p_val, 4),
            p_value=p_val,
            explanation=(
                f"{n} defects form a localized spatial cluster with Clark-Evans nearest-neighbour "
                f"ratio R = {nn_ratio:.2f} (< {config.local_cluster_max_nn_ratio:.2f}), "
                f"p = {p_val:.4f}."
            ),
            metrics=base_metrics,
            supporting_defect_ids=[rec["id"] for rec in records],
            warnings=warnings,
        )

    # Otherwise: RANDOM_SCATTER (report closest candidate pattern and its p-value)
    closest_pattern = min(PATTERN_TIE_BREAK_ORDER, key=lambda k: p_values[k])
    closest_p = p_values[closest_pattern]
    rule_note = (
        f"passed geometric rule but did not meet p <= {config.p_value_threshold}"
        if rules_passed[closest_pattern]
        else "did not meet geometric threshold"
    )
    return PatternResult(
        type="RANDOM_SCATTER",
        confidence=round(closest_p, 4),
        p_value=closest_p,
        explanation=(
            f"No spatial pattern met both geometric rules and p <= {config.p_value_threshold} across "
            f"{n} defects (closest pattern was {closest_pattern}, p = {closest_p:.4f}, {rule_note})."
        ),
        metrics=base_metrics,
        supporting_defect_ids=[rec["id"] for rec in records],
        warnings=warnings,
    )


def draw_pattern_overlay(
    original_image: np.ndarray,
    measurements: Sequence[RegionMeasurement],
    pattern: PatternResult,
) -> np.ndarray:
    """Render wafer-level spatial pattern overlay on a copy of the original image.

    Draws:
      - Wafer outer circle and faint radial guide rings (0.3R, 0.4R, 0.7R, 0.8R)
      - Pattern-specific highlight (band tint for EDGE_RING / CENTER_CLUSTER / DONUT,
        fitted line for SCRATCH_LINE, or cluster circle for LOCAL_CLUSTER)
      - Defect centroids coloured by severity (LOW blue, MEDIUM orange, HIGH red) + #ID
      - Top banner showing diagnosed pattern type, confidence, and p-value.
    """
    canvas = _to_rgb_canvas(original_image)
    img_h, img_w = canvas.shape[:2]

    center_orig = pattern.metrics.get("wafer_center_orig", (img_w / 2.0, img_h / 2.0))
    radius_orig = float(pattern.metrics.get("wafer_radius_orig", min(img_w, img_h) / 2.0))
    cx_i = int(round(float(center_orig[0])))
    cy_i = int(round(float(center_orig[1])))
    r_i = max(1, int(round(radius_orig)))

    # 1. Translucent pattern-specific band/cluster highlight
    tint_layer = canvas.copy()
    if pattern.type == "EDGE_RING":
        ring_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.circle(ring_mask, (cx_i, cy_i), r_i, 255, thickness=-1)
        cv2.circle(ring_mask, (cx_i, cy_i), int(round(0.80 * r_i)), 0, thickness=-1)
        tint_layer[ring_mask > 0] = (239, 68, 68)
        canvas = cv2.addWeighted(tint_layer, 0.22, canvas, 0.78, 0.0)
    elif pattern.type == "CENTER_CLUSTER":
        cv2.circle(tint_layer, (cx_i, cy_i), int(round(0.30 * r_i)), (234, 88, 12), thickness=-1)
        canvas = cv2.addWeighted(tint_layer, 0.22, canvas, 0.78, 0.0)
    elif pattern.type == "DONUT":
        donut_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.circle(donut_mask, (cx_i, cy_i), int(round(0.70 * r_i)), 255, thickness=-1)
        cv2.circle(donut_mask, (cx_i, cy_i), int(round(0.40 * r_i)), 0, thickness=-1)
        tint_layer[donut_mask > 0] = (147, 51, 234)
        canvas = cv2.addWeighted(tint_layer, 0.22, canvas, 0.78, 0.0)
    elif pattern.type == "LOCAL_CLUSTER" and measurements:
        supp_set = set(pattern.supporting_defect_ids)
        pts = np.array(
            [m.centroid for m in measurements if (not supp_set or m.id in supp_set)],
            dtype=np.float32,
        )
        if pts.shape[0] >= 2:
            (cl_x, cl_y), cl_r = cv2.minEnclosingCircle(pts)
            cv2.circle(
                tint_layer,
                (int(round(cl_x)), int(round(cl_y))),
                max(10, int(round(cl_r + 12))),
                (234, 88, 12),
                thickness=-1,
            )
            canvas = cv2.addWeighted(tint_layer, 0.20, canvas, 0.80, 0.0)
            cv2.circle(
                canvas,
                (int(round(cl_x)), int(round(cl_y))),
                max(10, int(round(cl_r + 12))),
                (234, 88, 12),
                thickness=2,
                lineType=cv2.LINE_AA,
            )

    # 2. Faint radial reference rings & wafer outline
    guide_layer = canvas.copy()
    for frac in (0.30, 0.40, 0.70, 0.80):
        cv2.circle(
            guide_layer,
            (cx_i, cy_i),
            max(1, int(round(frac * r_i))),
            (148, 163, 184),
            thickness=1,
            lineType=cv2.LINE_AA,
        )
    canvas = cv2.addWeighted(guide_layer, 0.45, canvas, 0.55, 0.0)
    cv2.circle(canvas, (cx_i, cy_i), r_i, (56, 189, 248), thickness=2, lineType=cv2.LINE_AA)

    # 3. Fitted TLS line for SCRATCH_LINE
    if pattern.type == "SCRATCH_LINE":
        endpoints = pattern.metrics.get("line_endpoints_orig")
        if endpoints and len(endpoints) == 2:
            p1 = (int(round(float(endpoints[0][0]))), int(round(float(endpoints[0][1]))))
            p2 = (int(round(float(endpoints[1][0]))), int(round(float(endpoints[1][1]))))
            cv2.line(canvas, p1, p2, (220, 38, 38), thickness=2, lineType=cv2.LINE_AA)
        elif measurements:
            # Single-region elongated scratch highlight via its bbox diagonal / oriented axis
            supp_set = set(pattern.supporting_defect_ids)
            for m in measurements:
                if m.id in supp_set:
                    rad = math.radians(m.angle_deg)
                    half_l = m.length_px / 2.0
                    p1 = (int(round(m.centroid[0] - half_l * math.cos(rad))), int(round(m.centroid[1] - half_l * math.sin(rad))))
                    p2 = (int(round(m.centroid[0] + half_l * math.cos(rad))), int(round(m.centroid[1] + half_l * math.sin(rad))))
                    cv2.line(canvas, p1, p2, (220, 38, 38), thickness=2, lineType=cv2.LINE_AA)

    # 4. Defect centroids coloured by severity
    for m in measurements:
        style = SEVERITY_OVERLAY_STYLES.get(m.severity, SEVERITY_OVERLAY_STYLES["LOW"])
        color = style["color"]  # type: ignore[assignment]
        mx = max(0, min(img_w - 1, int(round(m.centroid[0]))))
        my = max(0, min(img_h - 1, int(round(m.centroid[1]))))
        cv2.circle(canvas, (mx, my), 6, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (mx, my), 4, color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"#{m.id}",
            (min(img_w - 28, mx + 7), max(14, my - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )

    # 5. Pattern diagnosis label banner at top-left
    p_str = f"p={pattern.p_value:.4f}" if pattern.p_value is not None else "shape/count rule"
    banner_text = f"PATTERN: {pattern.type} ({pattern.confidence * 100.0:.1f}%, {p_str})"
    (tw, th), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(canvas, (8, 8), (min(img_w - 8, 22 + tw), 18 + th), (15, 23, 42), thickness=-1)
    cv2.putText(
        canvas,
        banner_text,
        (14, 13 + th),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (248, 250, 252),
        1,
        cv2.LINE_AA,
    )

    return canvas
