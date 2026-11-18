"""Full-pipeline accuracy evaluation benchmark for WaferInspect (Phase 5B).

Sweeps defect diameter (4, 8, 12, 16 px) and contrast (15, 30, 60 gray levels)
over 30 deterministic seeded synthetic wafers per cell across:
  1. golden           (grid pattern, golden reference comparison)
  2. self_ref_grid    (grid pattern, die-to-die self-reference, 3x3 shift tolerance ON)
  3. self_ref_circuit (circuit pattern, die-to-die self-reference, 3x3 shift tolerance ON)
  4. self_ref_notol   (grid pattern, die-to-die self-reference, 3x3 shift tolerance OFF - trade-off check)
  5. background_grid  (grid pattern, Gaussian background no-reference fallback)

Reports genuine post-filter Recall (%), False Positives per Image (FP/img),
mean funnel counts (raw_threshold, after_morphology, final), and mean verifiable coverage (%).
"""

from pathlib import Path
import sys
from typing import Dict, List, Literal, Tuple
import numpy as np

# Ensure backend root is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.cv.noise_filter import NoiseFilterConfig
from app.cv.pipeline import DetectionConfig, PreprocessingConfig, run_full_pipeline
from app.utils.synthetic import create_standard_synthetic_pair, generate_synthetic_wafer


def _stage_count_for(stage_counts: List[Dict[str, object]], stage_name: str) -> int:
    for item in stage_counts:
        if item.get("stage") == stage_name:
            return int(item.get("candidate_count", 0))
    return 0


def evaluate_combination(
    mode_name: str,
    die_pattern: Literal["grid", "circuit"],
    diameter: int,
    contrast: float,
    num_images: int = 30,
    base_seed: int = 1000,
    shift_tolerance: bool = True,
) -> Tuple[float, float, float, float, float, float]:
    """Evaluate post-filter recall, FP/image, mean funnel counts, and mean coverage."""
    true_positives = 0
    total_injected = 0
    total_false_positives = 0

    sum_raw = 0
    sum_morph = 0
    sum_final = 0
    sum_cov = 0.0

    radius = diameter / 2.0
    prep_config = PreprocessingConfig()
    filter_config = NoiseFilterConfig()

    if mode_name == "golden":
        det_config = DetectionConfig()
    elif mode_name == "background":
        det_config = DetectionConfig(no_reference_mode="background")
    else:
        det_config = DetectionConfig(
            no_reference_mode="self_reference",
            self_ref_shift_tolerance=shift_tolerance,
        )

    for i in range(num_images):
        seed = base_seed + i
        center_x = 256.0 + 40.0 * float(np.cos(i * 0.4))
        center_y = 256.0 + 40.0 * float(np.sin(i * 0.4))

        defect = {
            "type": "bright_blob",
            "center": (center_x, center_y),
            "radius": radius,
            "contrast": contrast,
        }

        if mode_name == "golden":
            ref_img, test_img, _ = create_standard_synthetic_pair(
                size=512,
                seed=seed,
                shift=(0.0, 0.0),
                defects=[defect],
                die_pattern=die_pattern,
            )
            result = run_full_pipeline(
                test_image=test_img,
                reference_image=ref_img,
                prep_config=prep_config,
                det_config=det_config,
                filter_config=filter_config,
            )
            cov = 1.0
        else:
            test_img, _ = generate_synthetic_wafer(
                size=512,
                seed=seed,
                noise_std=5.0,
                die_pattern=die_pattern,
                defects=[defect],
            )
            result = run_full_pipeline(
                test_image=test_img,
                reference_image=None,
                prep_config=prep_config,
                det_config=det_config,
                filter_config=filter_config,
            )
            cov = result.detection.coverage if result.detection.coverage is not None else 1.0

        total_injected += 1
        sum_raw += _stage_count_for(result.stage_counts, "raw_threshold")
        sum_morph += _stage_count_for(result.stage_counts, "after_morphology")
        sum_final += _stage_count_for(result.stage_counts, "final")
        sum_cov += cov

        match_found = False
        fp_count = 0
        match_tolerance = max(float(diameter), 10.0)

        for region in result.accepted_regions:
            cx, cy = region.centroid
            dist = float(np.hypot(cx - center_x, cy - center_y))
            if dist <= match_tolerance and not match_found:
                match_found = True
            else:
                fp_count += 1

        if match_found:
            true_positives += 1
        total_false_positives += fp_count

    recall = (true_positives / total_injected) * 100.0 if total_injected > 0 else 0.0
    fp_per_img = total_false_positives / float(num_images)
    mean_raw = sum_raw / float(num_images)
    mean_morph = sum_morph / float(num_images)
    mean_final = sum_final / float(num_images)
    mean_cov_pct = (sum_cov / float(num_images)) * 100.0

    return recall, fp_per_img, mean_raw, mean_morph, mean_final, mean_cov_pct


def run_benchmark() -> List[Dict[str, object]]:
    diameters = [4, 8, 12, 16]
    contrasts = [15.0, 30.0, 60.0]
    num_runs = 30

    configs: List[Tuple[str, str, Literal["grid", "circuit"], int, bool]] = [
        ("golden (grid)", "golden", "grid", 1000, True),
        ("self_ref (grid)", "self_reference", "grid", 2000, True),
        ("self_ref (circuit)", "self_reference", "circuit", 3000, True),
        ("self_ref_notol(grd)", "self_reference", "grid", 2000, False),
        ("background (grid)", "background", "grid", 4000, True),
    ]

    header = (
        f"{'Path / Pattern':<19} | {'Diam':<4} | {'Contr':<5} | {'Runs':<4} | "
        f"{'Recall(%)':<9} | {'FP/Img':<7} | {'Mean Raw':<8} | {'Mean Morph':<10} | "
        f"{'Mean Final':<10} | {'Cov(%)':<6}"
    )
    sep = "=" * len(header)
    print(sep)
    print("WaferInspect Phase 5B Full-Pipeline Accuracy Benchmark (30 Seeded Runs per Cell)")
    print(sep)
    print(header)
    print("-" * len(header))

    results_table: List[Dict[str, object]] = []

    for label, mode_name, pattern, base_seed, shift_tol in configs:
        for d in diameters:
            for c in contrasts:
                recall, fp, m_raw, m_morph, m_final, m_cov = evaluate_combination(
                    mode_name=mode_name,
                    die_pattern=pattern,
                    diameter=d,
                    contrast=c,
                    num_images=num_runs,
                    base_seed=base_seed,
                    shift_tolerance=shift_tol,
                )
                row: Dict[str, object] = {
                    "label": label,
                    "diameter": d,
                    "contrast": c,
                    "runs": num_runs,
                    "recall": recall,
                    "fp": fp,
                    "mean_raw": m_raw,
                    "mean_morph": m_morph,
                    "mean_final": m_final,
                    "mean_cov_pct": m_cov,
                }
                results_table.append(row)
                print(
                    f"{label:<19} | {d:<4} | {c:<5.1f} | {num_runs:<4} | "
                    f"{recall:<9.1f} | {fp:<7.3f} | {m_raw:<8.2f} | {m_morph:<10.2f} | "
                    f"{m_final:<10.2f} | {m_cov:<6.1f}"
                )
        print("-" * len(header))

    print(sep)
    return results_table


if __name__ == "__main__":
    run_benchmark()
