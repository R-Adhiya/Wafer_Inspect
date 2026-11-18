"""Full-pipeline accuracy evaluation benchmark for WaferInspect (Phase 5).

Runs the complete classical CV pipeline (preprocessing -> detection -> segmentation ->
noise filtering) over 30 deterministic seeded synthetic images per (path, diameter, contrast)
combination. Sweeps defect diameter (4, 8, 12, 16 px) and contrast (15, 30, 60 gray levels)
for both Golden Reference and Experimental No-Reference paths.

Reports genuine Recall (%) and False Positives per Image (FP/img) AFTER filtering,
alongside the mean candidate funnel counts (raw_threshold, after_morphology, final).
"""

from pathlib import Path
import sys
from typing import Dict, List, Tuple
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
    path: str,
    diameter: int,
    contrast: float,
    num_images: int = 30,
    base_seed: int = 1000,
) -> Tuple[float, float, float, float, float]:
    """Evaluate post-filter recall, FP/image, and mean funnel counts for one sweep cell."""
    true_positives = 0
    total_injected = 0
    total_false_positives = 0

    sum_raw = 0
    sum_morph = 0
    sum_final = 0

    radius = diameter / 2.0
    prep_config = PreprocessingConfig()
    det_config = DetectionConfig()
    filter_config = NoiseFilterConfig()

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

        if path == "golden":
            ref_img, test_img, _ = create_standard_synthetic_pair(
                size=512,
                seed=seed,
                shift=(0.0, 0.0),
                defects=[defect],
            )
            result = run_full_pipeline(
                test_image=test_img,
                reference_image=ref_img,
                prep_config=prep_config,
                det_config=det_config,
                filter_config=filter_config,
            )
        else:
            test_img, _ = generate_synthetic_wafer(
                size=512,
                seed=seed,
                noise_std=5.0,
                defects=[defect],
            )
            result = run_full_pipeline(
                test_image=test_img,
                reference_image=None,
                prep_config=prep_config,
                det_config=det_config,
                filter_config=filter_config,
            )

        total_injected += 1

        sum_raw += _stage_count_for(result.stage_counts, "raw_threshold")
        sum_morph += _stage_count_for(result.stage_counts, "after_morphology")
        sum_final += _stage_count_for(result.stage_counts, "final")

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

    return recall, fp_per_img, mean_raw, mean_morph, mean_final


def run_benchmark() -> List[Dict[str, object]]:
    diameters = [4, 8, 12, 16]
    contrasts = [15.0, 30.0, 60.0]
    num_runs = 30

    header = (
        f"{'Path':<13} | {'Diam':<4} | {'Contr':<5} | {'Runs':<4} | "
        f"{'Recall(%)':<9} | {'FP/Img':<7} | {'Mean Raw':<8} | {'Mean Morph':<10} | {'Mean Final':<10}"
    )
    sep = "=" * len(header)
    print(sep)
    print("WaferInspect Full-Pipeline Accuracy Benchmark (Post-Filter, 30 Seeded Runs per Cell)")
    print(sep)
    print(header)
    print("-" * len(header))

    results_table: List[Dict[str, object]] = []

    for path in ["golden", "no_reference"]:
        base_seed = 1000 if path == "golden" else 2000
        for d in diameters:
            for c in contrasts:
                recall, fp, m_raw, m_morph, m_final = evaluate_combination(
                    path, d, c, num_images=num_runs, base_seed=base_seed
                )
                row: Dict[str, object] = {
                    "path": path,
                    "diameter": d,
                    "contrast": c,
                    "runs": num_runs,
                    "recall": recall,
                    "fp": fp,
                    "mean_raw": m_raw,
                    "mean_morph": m_morph,
                    "mean_final": m_final,
                }
                results_table.append(row)
                print(
                    f"{path:<13} | {d:<4} | {c:<5.1f} | {num_runs:<4} | "
                    f"{recall:<9.1f} | {fp:<7.3f} | {m_raw:<8.2f} | {m_morph:<10.2f} | {m_final:<10.2f}"
                )

    print(sep)
    return results_table


if __name__ == "__main__":
    run_benchmark()
