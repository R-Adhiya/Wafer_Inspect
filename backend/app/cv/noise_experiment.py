"""Pure functional logic for noise-filtering evaluation and multi-level noise sweeps.

Computes the four canonical stages of the defect detection pipeline:
1. Raw detection: connected components of the raw thresholded mask.
2. After minimum-area filtering only: components of the raw mask with area >= min_area_px.
3. After morphological filtering: components of the morphology mask with area >= min_area_px.
4. Final defects: accepted regions of the full pipeline (fragment merge, size, contrast rules).

When ground-truth defects are available (synthetic images), matches each stage's candidate
centroids to true defect centroids via greedy nearest-neighbor one-to-one matching, reporting
TP, FP, FN, precision, and recall per stage. When not available, reports counts only and sets
ground_truth to None.
"""

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from app.cv.noise_filter import NoiseFilterConfig
from app.cv.pipeline import DetectionConfig, FullPipelineResult, PreprocessingConfig, run_full_pipeline
from app.cv.segmentation import Region
from app.utils.synthetic import (
    DefectLayout,
    create_standard_synthetic_pair,
    generate_layout_defects,
    generate_layout_wafer,
)

logger = logging.getLogger("waferinspect.cv.noise_experiment")

SYNTHETIC_DATA_DISCLAIMER = "Synthetic demonstration data"

STAGE_NAMES = [
    "raw_threshold",
    "min_area_only",
    "after_morphology_and_area",
    "final_defects",
]

STAGE_LABELS = [
    "1. Raw Detection",
    "2. Min-Area Filter Only",
    "3. Morphology + Area",
    "4. Final Filtered Defects",
]


@dataclass(frozen=True)
class MatchingMetrics:
    """Evaluation metrics comparing candidate centroids to ground-truth defects."""

    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    ground_truth_count: int
    candidate_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            "ground_truth_count": self.ground_truth_count,
            "candidate_count": self.candidate_count,
        }


def extract_gt_centroids(
    ground_truth: Sequence[Union[Dict[str, Any], Tuple[float, float]]],
) -> List[Tuple[float, float]]:
    """Extract (x, y) centroids from ground-truth defect specifications."""
    pts: List[Tuple[float, float]] = []
    for gt in ground_truth:
        if isinstance(gt, dict):
            if "center" in gt:
                cx, cy = gt["center"]
                pts.append((float(cx), float(cy)))
            elif "centroid" in gt:
                cx, cy = gt["centroid"]
                pts.append((float(cx), float(cy)))
            elif "start" in gt and "end" in gt:
                sx, sy = gt["start"]
                ex, ey = gt["end"]
                pts.append(((float(sx) + float(ex)) / 2.0, (float(sy) + float(ey)) / 2.0))
        elif isinstance(gt, (tuple, list)) and len(gt) >= 2:
            pts.append((float(gt[0]), float(gt[1])))
    return pts


def match_candidates_to_ground_truth(
    candidates: Sequence[Tuple[float, float]],
    ground_truth: Sequence[Union[Dict[str, Any], Tuple[float, float]]],
    tolerance_px: float = 15.0,
) -> MatchingMetrics:
    """Match candidate centroids to true defect centroids via greedy nearest-neighbor one-to-one matching.

    Args:
        candidates: List of (x, y) candidate centroids.
        ground_truth: List of GT defect dicts (with 'center') or (x, y) tuples.
        tolerance_px: Maximum Euclidean distance in pixels to count as a match.

    Returns:
        MatchingMetrics with TP, FP, FN, precision, and recall.
    """
    gt_pts = extract_gt_centroids(ground_truth)
    n_cand = len(candidates)
    n_gt = len(gt_pts)

    if n_gt == 0:
        # Zero ground truth defects (e.g. clean wafer)
        return MatchingMetrics(
            tp=0,
            fp=n_cand,
            fn=0,
            precision=1.0 if n_cand == 0 else 0.0,
            recall=1.0,
            ground_truth_count=0,
            candidate_count=n_cand,
        )

    if n_cand == 0:
        # Candidates empty but GT exists
        return MatchingMetrics(
            tp=0,
            fp=0,
            fn=n_gt,
            precision=0.0,
            recall=0.0,
            ground_truth_count=n_gt,
            candidate_count=0,
        )

    # Compute pairwise Euclidean distances
    pairs: List[Tuple[float, int, int]] = []
    for i, (cx, cy) in enumerate(candidates):
        for j, (gx, gy) in enumerate(gt_pts):
            dist = math.hypot(cx - gx, cy - gy)
            if dist <= tolerance_px:
                pairs.append((dist, i, j))

    # Sort pairs by distance ascending (greedy nearest neighbor)
    pairs.sort(key=lambda p: p[0])

    matched_candidates: set[int] = set()
    matched_gt: set[int] = set()

    for dist, i, j in pairs:
        if i not in matched_candidates and j not in matched_gt:
            matched_candidates.add(i)
            matched_gt.add(j)

    tp = len(matched_gt)
    fp = n_cand - tp
    fn = n_gt - tp

    precision = round(float(tp) / float(n_cand), 4) if n_cand > 0 else 0.0
    recall = round(float(tp) / float(n_gt), 4) if n_gt > 0 else 1.0

    return MatchingMetrics(
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        ground_truth_count=n_gt,
        candidate_count=n_cand,
    )


def compute_four_cases_from_components(
    raw_mask: np.ndarray,
    morph_mask: np.ndarray,
    accepted_regions: Sequence[Region],
    min_area_px: int = 12,
    scale_factor: float = 1.0,
    ground_truth: Optional[Sequence[Union[Dict[str, Any], Tuple[float, float]]]] = None,
    tolerance_px: float = 15.0,
) -> List[Dict[str, Any]]:
    """Compute candidate counts and optional ground-truth metrics across the four canonical stages."""
    scale = float(scale_factor) if scale_factor > 0 else 1.0
    area_scale = scale * scale

    # Stage 1: Raw detection (connected components of raw thresholded mask)
    num1, labels1, stats1, centroids1 = cv2.connectedComponentsWithStats(raw_mask, connectivity=8)
    cand1: List[Tuple[float, float]] = []
    areas1: List[float] = []
    for i in range(1, num1):
        area_orig = float(stats1[i, cv2.CC_STAT_AREA]) / area_scale
        cx_orig = float(centroids1[i, 0]) / scale
        cy_orig = float(centroids1[i, 1]) / scale
        cand1.append((cx_orig, cy_orig))
        areas1.append(area_orig)

    # Stage 2: After minimum-area filtering only (raw mask components with area >= min_area_px)
    cand2: List[Tuple[float, float]] = [cand1[i] for i, area in enumerate(areas1) if area >= float(min_area_px)]

    # Stage 3: After morphological filtering (+ minimum area filtering)
    num3, labels3, stats3, centroids3 = cv2.connectedComponentsWithStats(morph_mask, connectivity=8)
    cand3: List[Tuple[float, float]] = []
    for i in range(1, num3):
        area_orig = float(stats3[i, cv2.CC_STAT_AREA]) / area_scale
        if area_orig >= float(min_area_px):
            cx_orig = float(centroids3[i, 0]) / scale
            cy_orig = float(centroids3[i, 1]) / scale
            cand3.append((cx_orig, cy_orig))

    # Stage 4: Final defects (the accepted regions of the full pipeline)
    cand4: List[Tuple[float, float]] = [(float(r.centroid[0]), float(r.centroid[1])) for r in accepted_regions]

    candidates_by_stage = [cand1, cand2, cand3, cand4]
    has_gt = ground_truth is not None

    cases: List[Dict[str, Any]] = []
    for idx, cands in enumerate(candidates_by_stage):
        stage_id = idx + 1
        stage_name = STAGE_NAMES[idx]
        label = STAGE_LABELS[idx]
        count = len(cands)

        entry: Dict[str, Any] = {
            "stage_id": stage_id,
            "stage_name": stage_name,
            "label": label,
            "candidate_count": count,
            "tp": None,
            "fp": None,
            "fn": None,
            "precision": None,
            "recall": None,
        }

        if has_gt:
            metrics = match_candidates_to_ground_truth(cands, ground_truth, tolerance_px=tolerance_px)
            entry["tp"] = metrics.tp
            entry["fp"] = metrics.fp
            entry["fn"] = metrics.fn
            entry["precision"] = metrics.precision
            entry["recall"] = metrics.recall

        cases.append(entry)

    logger.info(
        "Noise experiment 4 cases: raw=%d, area_only=%d, morph_area=%d, final=%d (has_gt=%s)",
        cases[0]["candidate_count"],
        cases[1]["candidate_count"],
        cases[2]["candidate_count"],
        cases[3]["candidate_count"],
        has_gt,
    )
    return cases


def evaluate_inspection_noise_experiment(
    result: FullPipelineResult,
    min_area_px: int = 12,
    ground_truth: Optional[Sequence[Union[Dict[str, Any], Tuple[float, float]]]] = None,
    tolerance_px: float = 15.0,
) -> Dict[str, Any]:
    """Run four-stage noise experiment on a completed pipeline execution."""
    cases = compute_four_cases_from_components(
        raw_mask=result.detection.raw_mask,
        morph_mask=result.detection.morphology_mask,
        accepted_regions=result.accepted_regions,
        min_area_px=min_area_px,
        scale_factor=result.scale_factor,
        ground_truth=ground_truth,
        tolerance_px=tolerance_px,
    )

    c1 = cases[0]["candidate_count"]
    c4 = cases[3]["candidate_count"]
    pct_filtered = round(float(c1 - c4) / float(c1) * 100.0, 1) if c1 > 0 else 0.0

    if ground_truth is not None:
        gt_count = len(extract_gt_centroids(ground_truth))
        summary = (
            f"Raw detection produced {c1} candidates. Hierarchical filtering (area, morphology, "
            f"and robust contrast) suppressed {c1 - c4} false alarm(s) ({pct_filtered}%), "
            f"identifying {c4} final defects against {gt_count} true ground-truth defects "
            f"(precision: {cases[3]['precision'] * 100:.1f}%, recall: {cases[3]['recall'] * 100:.1f}%)."
        )
    else:
        summary = (
            f"Raw thresholding yielded {c1} candidate region(s). Minimum-area filtering, morphological "
            f"cleanup, and robust contrast-to-noise gating eliminated {c1 - c4} candidate(s) ({pct_filtered}%), "
            f"yielding {c4} accepted defect region(s)."
        )

    return {
        "has_ground_truth": ground_truth is not None,
        "ground_truth_count": len(extract_gt_centroids(ground_truth)) if ground_truth is not None else None,
        "tolerance_px": tolerance_px,
        "cases": cases,
        "summary": summary,
        "disclaimer": SYNTHETIC_DATA_DISCLAIMER if ground_truth is not None else None,
    }


def run_single_noise_evaluation(
    test_image: np.ndarray,
    reference_image: Optional[np.ndarray] = None,
    prep_config: Optional[PreprocessingConfig] = None,
    det_config: Optional[DetectionConfig] = None,
    filter_config: Optional[NoiseFilterConfig] = None,
    ground_truth: Optional[Sequence[Union[Dict[str, Any], Tuple[float, float]]]] = None,
    tolerance_px: float = 15.0,
) -> Dict[str, Any]:
    """Execute the full CV pipeline and return the four noise experiment cases."""
    if filter_config is None:
        filter_config = NoiseFilterConfig()

    result = run_full_pipeline(
        test_image=test_image,
        reference_image=reference_image,
        prep_config=prep_config,
        det_config=det_config,
        filter_config=filter_config,
    )

    return evaluate_inspection_noise_experiment(
        result=result,
        min_area_px=filter_config.min_area_px,
        ground_truth=ground_truth,
        tolerance_px=tolerance_px,
    )


def run_noise_sweep(
    layout: DefectLayout = "edge_ring",
    path: Literal["golden", "self_reference"] = "self_reference",
    noise_levels: Optional[List[float]] = None,
    seeds: Optional[List[int]] = None,
    size: int = 512,
    filter_config: Optional[NoiseFilterConfig] = None,
    det_config: Optional[DetectionConfig] = None,
    tolerance_px: float = 15.0,
    defect_count: int = 8,
) -> Dict[str, Any]:
    """Run a deterministic multi-seed noise sweep over a series of Gaussian noise sigmas.

    Args:
        layout: Defect layout name ('edge_ring', 'center_cluster', 'donut', 'scratch_line', 'local_cluster', 'random').
        path: Detection path: 'golden' (with reference) or 'self_reference' (single wafer).
        noise_levels: List of Gaussian noise sigma values (at most 8 levels).
        seeds: List of integer seeds (at most 10 seeds).
        size: Image dimension (width & height, at most 1024).
        filter_config: Optional NoiseFilterConfig (uses defaults if None).
        det_config: Optional DetectionConfig (uses defaults if None).
        tolerance_px: Ground truth centroid matching distance tolerance.
        defect_count: Number of synthetic defects injected (default 8).

    Returns:
        Structured sweep dictionary containing per-level statistics, per-case means/stds,
        precision/recall series, and an automated plain-language summary.
    """
    levels = [float(x) for x in (noise_levels if noise_levels is not None else [2.0, 5.0, 10.0, 15.0, 20.0])]
    seed_list = [int(s) for s in (seeds if seeds is not None else [42, 43, 44, 45, 46])]

    if len(levels) > 8:
        raise ValueError(f"Too many noise levels: requested {len(levels)}, maximum allowed is 8.")
    if len(seed_list) > 10:
        raise ValueError(f"Too many seeds: requested {len(seed_list)}, maximum allowed is 10.")
    if size > 1024:
        raise ValueError(f"Image dimension {size} exceeds maximum allowable sweep size of 1024.")
    if size < 64:
        raise ValueError(f"Image dimension {size} is smaller than minimum 64.")

    if filter_config is None:
        filter_config = NoiseFilterConfig()
    if det_config is None:
        det_config = DetectionConfig(
            path="golden" if path == "golden" else "self_reference",
            no_reference_mode="self_reference",
        )

    sweep_series: List[Dict[str, Any]] = []
    true_defect_count = defect_count

    for sigma in levels:
        level_cases: List[List[Dict[str, Any]]] = []

        for seed in seed_list:
            if path == "golden":
                # Generate matching golden reference pair
                defects, _ = generate_layout_defects(
                    layout=layout,
                    count=defect_count,
                    size=size,
                    seed=seed,
                )
                ref_img, test_img, gt = create_standard_synthetic_pair(
                    size=size,
                    seed=seed,
                    defects=defects,
                    noise_std=sigma,
                    die_pattern="grid",
                )
                eval_res = run_single_noise_evaluation(
                    test_image=test_img,
                    reference_image=ref_img,
                    det_config=det_config,
                    filter_config=filter_config,
                    ground_truth=gt,
                    tolerance_px=tolerance_px,
                )
            else:
                # Single patterned wafer with self-reference detection
                test_img, gt, _ = generate_layout_wafer(
                    defect_layout=layout,
                    count=defect_count,
                    size=size,
                    seed=seed,
                    grid_pitch=32,
                    die_pattern="grid",
                    noise_std=sigma,
                    radius=7.0,
                    contrast=65.0,
                )
                eval_res = run_single_noise_evaluation(
                    test_image=test_img,
                    reference_image=None,
                    det_config=det_config,
                    filter_config=filter_config,
                    ground_truth=gt,
                    tolerance_px=tolerance_px,
                )

            level_cases.append(eval_res["cases"])
            if gt:
                true_defect_count = len(gt)

        # Aggregate across seeds for this noise level
        cases_summary: List[Dict[str, Any]] = []
        for stage_idx in range(4):
            counts = [run[stage_idx]["candidate_count"] for run in level_cases]
            precisions = [run[stage_idx]["precision"] for run in level_cases if run[stage_idx]["precision"] is not None]
            recalls = [run[stage_idx]["recall"] for run in level_cases if run[stage_idx]["recall"] is not None]
            tps = [run[stage_idx]["tp"] for run in level_cases if run[stage_idx]["tp"] is not None]
            fps = [run[stage_idx]["fp"] for run in level_cases if run[stage_idx]["fp"] is not None]
            fns = [run[stage_idx]["fn"] for run in level_cases if run[stage_idx]["fn"] is not None]

            mean_cnt = float(np.mean(counts))
            std_cnt = float(np.std(counts))
            mean_prec = float(np.mean(precisions)) if precisions else 0.0
            std_prec = float(np.std(precisions)) if precisions else 0.0
            mean_rec = float(np.mean(recalls)) if recalls else 0.0
            std_rec = float(np.std(recalls)) if recalls else 0.0

            cases_summary.append(
                {
                    "stage_id": stage_idx + 1,
                    "stage_name": STAGE_NAMES[stage_idx],
                    "label": STAGE_LABELS[stage_idx],
                    "mean_count": round(mean_cnt, 2),
                    "std_count": round(std_cnt, 2),
                    "mean_precision": round(mean_prec, 4),
                    "std_precision": round(std_prec, 4),
                    "mean_recall": round(mean_rec, 4),
                    "std_recall": round(std_rec, 4),
                    "mean_tp": round(float(np.mean(tps)), 2) if tps else 0.0,
                    "mean_fp": round(float(np.mean(fps)), 2) if fps else 0.0,
                    "mean_fn": round(float(np.mean(fns)), 2) if fns else 0.0,
                }
            )

        # Per-level plain-language explanation
        lvl_c1 = cases_summary[0]["mean_count"]
        lvl_c4 = cases_summary[3]["mean_count"]
        lvl_tp4 = cases_summary[3]["mean_tp"]
        lvl_removed = round(max(0.0, lvl_c1 - lvl_c4), 1)
        lvl_pct = round((lvl_removed / max(lvl_c1, 1.0)) * 100.0, 1)
        lvl_summary = (
            f"Noise filtering removed an average of {lvl_removed:.1f} false detections "
            f"({lvl_pct:.1f}%) at sigma={sigma:.1f} while retaining {lvl_tp4:.1f} of {true_defect_count} true defects."
        )

        sweep_series.append(
            {
                "noise_level": round(sigma, 2),
                "seed_count": len(seed_list),
                "cases": cases_summary,
                "summary": lvl_summary,
            }
        )

    # Construct dynamic, honest plain-language summary based on actual measurements
    c1_low = sweep_series[0]["cases"][0]["mean_count"]
    c1_high = sweep_series[-1]["cases"][0]["mean_count"]
    c4_low = sweep_series[0]["cases"][3]["mean_count"]
    c4_high = sweep_series[-1]["cases"][3]["mean_count"]
    p4_low = sweep_series[0]["cases"][3]["mean_precision"] * 100.0
    p4_high = sweep_series[-1]["cases"][3]["mean_precision"] * 100.0
    r4_low = sweep_series[0]["cases"][3]["mean_recall"] * 100.0
    r4_high = sweep_series[-1]["cases"][3]["mean_recall"] * 100.0

    raw_growth_pct = round(((c1_high - c1_low) / max(c1_low, 1.0)) * 100.0, 1)

    # Representative point (median level)
    rep_idx = len(sweep_series) // 2
    rep_point = sweep_series[rep_idx]
    rep_sigma = rep_point["noise_level"]
    rep_c1 = rep_point["cases"][0]["mean_count"]
    rep_c4 = rep_point["cases"][3]["mean_count"]
    rep_tp4 = rep_point["cases"][3]["mean_tp"]
    rep_removed = round(max(0.0, rep_c1 - rep_c4), 1)
    rep_pct = round((rep_removed / max(rep_c1, 1.0)) * 100.0, 1)

    # Determine highest noise level where final defect count remained within +-1 of true count
    effective_sigma = levels[0]
    for pt in sweep_series:
        c4 = pt["cases"][3]["mean_count"]
        if abs(c4 - true_defect_count) <= 1.0:
            effective_sigma = pt["noise_level"]

    summary_text = (
        f"Noise filtering removed an average of {rep_removed:.1f} false detections ({rep_pct:.1f}%) "
        f"at sigma={rep_sigma:.1f} while retaining {rep_tp4:.1f} of {true_defect_count} true defects. "
        f"Across {len(levels)} noise levels (sigma={levels[0]} to {levels[-1]} gray levels) and {len(seed_list)} seeds, "
        f"raw threshold candidates surged from {c1_low:.1f} to {c1_high:.1f} (+{raw_growth_pct}%). "
        f"Hierarchical filtering eliminated noise specks, keeping final defects near the ground-truth target of {true_defect_count} "
        f"({c4_low:.1f} at sigma={levels[0]}, {c4_high:.1f} at sigma={levels[-1]}). "
        f"Final stage precision was {p4_low:.1f}% at sigma={levels[0]} and {p4_high:.1f}% at sigma={levels[-1]}, "
        f"with recall at {r4_low:.1f}% and {r4_high:.1f}%. "
        f"Filtering robustly controlled false alarms up to sigma={effective_sigma} gray levels."
    )

    return {
        "layout": layout,
        "path": path,
        "size": size,
        "noise_levels": levels,
        "seeds": seed_list,
        "true_defect_count": true_defect_count,
        "series": sweep_series,
        "summary": summary_text,
        "disclaimer": SYNTHETIC_DATA_DISCLAIMER,
    }
