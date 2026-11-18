"""Spatial pattern diagnosis benchmark for WaferInspect (Phase 6B).

Evaluates `analyze_spatial_pattern` directly on seeded synthetic defect centroids across:
  - Layouts: random, edge_ring, center_cluster, donut, scratch_line, local_cluster
  - Defect counts: n = 3, 5, 8, 15
  - 100 deterministic seeds per (layout, n) cell (2,400 total evaluations)

Reports confusion matrices, per-class recall and precision, and false-pattern rate
on random layouts (including weak cells such as n = 3), and writes `docs/pattern_accuracy.md`.
"""

from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Dict, List
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.cv.pattern_analysis import PatternConfig, analyze_spatial_pattern  # noqa: E402
from app.utils.synthetic import (  # noqa: E402
    DefectLayout,
    LAYOUT_TO_PATTERN_LABEL,
    generate_layout_centroids,
)

LAYOUTS: List[DefectLayout] = [
    "random",
    "edge_ring",
    "center_cluster",
    "donut",
    "scratch_line",
    "local_cluster",
]

CLASSES: List[str] = [LAYOUT_TO_PATTERN_LABEL[lay] for lay in LAYOUTS]


def evaluate_for_n(
    n: int,
    num_seeds: int = 100,
    base_seed: int = 20000,
) -> Dict[str, object]:
    """Run 100 seeds per layout for defect count `n` and compute confusion matrix + metrics."""
    cfg = PatternConfig(mc_trials=2000, mc_seed=42)
    wafer_center = (256.0, 256.0)
    wafer_radius = 225.28
    image_size = (512, 512)

    conf: Dict[str, Dict[str, int]] = {
        true_cls: {pred_cls: 0 for pred_cls in CLASSES} for true_cls in CLASSES
    }

    for lay_idx, layout in enumerate(LAYOUTS):
        true_cls = LAYOUT_TO_PATTERN_LABEL[layout]
        for i in range(num_seeds):
            seed = base_seed + n * 1000 + lay_idx * 200 + i
            centroids, _ = generate_layout_centroids(
                layout=layout,
                count=n,
                center=wafer_center,
                radius=wafer_radius,
                seed=seed,
                min_separation_px=0.0,
            )
            records = [
                {
                    "id": idx + 1,
                    "centroid": pt,
                    "area_px": 150,
                    "length_px": 14.0,
                    "elongation": 1.0,
                    "severity": "MEDIUM",
                }
                for idx, pt in enumerate(centroids)
            ]
            res = analyze_spatial_pattern(
                measurements=records,
                wafer_center=wafer_center,
                wafer_radius=wafer_radius,
                image_size=image_size,
                scale_factor=1.0,
                is_full_frame=False,
                config=cfg,
            )
            pred_cls = res.type if res.type in CLASSES else "RANDOM_SCATTER"
            conf[true_cls][pred_cls] += 1

    per_class: Dict[str, Dict[str, float]] = {}
    for cls in CLASSES:
        tp = conf[cls][cls]
        total_true = sum(conf[cls].values())
        total_pred = sum(conf[t][cls] for t in CLASSES)
        recall = (tp / total_true * 100.0) if total_true > 0 else 0.0
        precision = (tp / total_pred * 100.0) if total_pred > 0 else 0.0
        per_class[cls] = {
            "recall": round(recall, 1),
            "precision": round(precision, 1),
            "tp": tp,
            "total_true": total_true,
            "total_pred": total_pred,
        }

    random_false_patterns = num_seeds - conf["RANDOM_SCATTER"]["RANDOM_SCATTER"]
    false_pattern_rate = (random_false_patterns / float(num_seeds)) * 100.0

    return {
        "n": n,
        "num_seeds": num_seeds,
        "confusion": conf,
        "per_class": per_class,
        "false_pattern_rate": round(false_pattern_rate, 1),
    }


def run_pattern_benchmark() -> List[Dict[str, object]]:
    counts = [3, 5, 8, 15]
    all_results: List[Dict[str, object]] = []

    print("=" * 98)
    print("WaferInspect Phase 6B Spatial Pattern Diagnosis Benchmark (100 Seeds per Layout x n)")
    print("=" * 98)

    for n in counts:
        res = evaluate_for_n(n=n, num_seeds=100)
        all_results.append(res)
        print(f"\n--- n = {n} defects (Random false-pattern rate: {res['false_pattern_rate']:.1f}%) ---")
        print(f"{'Class':<18} | {'Recall (%)':<10} | {'Precision (%)':<13} | {'TP/True':<8} | {'Pred Total':<10}")
        print("-" * 70)
        per_class = res["per_class"]  # type: ignore[assignment]
        for cls in CLASSES:
            m = per_class[cls]
            print(
                f"{cls:<18} | {m['recall']:<10.1f} | {m['precision']:<13.1f} | "
                f"{int(m['tp'])}/{int(m['total_true']):<5} | {int(m['total_pred']):<10}"
            )

        # Print confusion matrix
        conf = res["confusion"]  # type: ignore[assignment]
        short = {
            "RANDOM_SCATTER": "RAND",
            "EDGE_RING": "EDGE",
            "CENTER_CLUSTER": "CNTR",
            "DONUT": "DONUT",
            "SCRATCH_LINE": "SCRAT",
            "LOCAL_CLUSTER": "LOCAL",
        }
        print("\nConfusion Matrix (rows=True, cols=Predicted):")
        hdr = f"{'True \\ Pred':<16} | " + " | ".join(f"{short[c]:>5}" for c in CLASSES)
        print(hdr)
        print("-" * len(hdr))
        for t in CLASSES:
            row_str = f"{t:<16} | " + " | ".join(f"{conf[t][p]:>5}" for p in CLASSES)
            print(row_str)

    return all_results


def write_pattern_markdown(all_results: List[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generated_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    short = {
        "RANDOM_SCATTER": "RAND",
        "EDGE_RING": "EDGE",
        "CENTER_CLUSTER": "CNTR",
        "DONUT": "DONUT",
        "SCRATCH_LINE": "SCRAT",
        "LOCAL_CLUSTER": "LOCAL",
    }

    lines = [
        "# WaferInspect Spatial Pattern Diagnosis Accuracy Benchmark",
        "",
        f"Generated by `scripts/evaluate_patterns.py` on {generated_utc}.",
        "",
        "## Benchmark Setup",
        "",
        "- **Method**: Rule-based geometric spatial tests gated by seeded Monte Carlo significance (`mc_trials=2000`, `p_value <= 0.01`) against uniform-in-disc random placement (`backend/app/cv/pattern_analysis.py`).",
        "- **Sweep**: All 6 layouts (`random`, `edge_ring`, `center_cluster`, `donut`, `scratch_line`, `local_cluster`) × defect counts `n ∈ {3, 5, 8, 15}` × `100` deterministic seeds per cell (`2,400` total runs).",
        "- **Note on small n (e.g. n = 3)**: Under uniform random placement of only 3 points in a disc, even extreme configurations (e.g. 3/3 in the outer 20% ring has probability `0.36^3 = 4.7%`, and 3/3 in the donut band has probability `0.33^3 = 3.6%`) cannot achieve `p <= 0.01`, so the strict `p <= 0.01` gate deliberately classifies weak low-n patterns as `RANDOM_SCATTER` rather than hallucinating high-confidence diagnoses.",
        "",
        "## Summary of Random-Layout False-Pattern Rate",
        "",
        "| Defects (n) | Random Runs | False Patterns Reported | False-Pattern Rate (%) |",
        "| ---: | ---: | ---: | ---: |",
    ]

    for res in all_results:
        n = int(res["n"])  # type: ignore[arg-type]
        num = int(res["num_seeds"])  # type: ignore[arg-type]
        fpr = float(res["false_pattern_rate"])  # type: ignore[arg-type]
        conf = res["confusion"]  # type: ignore[assignment]
        fp = num - conf["RANDOM_SCATTER"]["RANDOM_SCATTER"]
        lines.append(f"| {n} | {num} | {fp} | {fpr:.1f}% |")

    for res in all_results:
        n = int(res["n"])  # type: ignore[arg-type]
        fpr = float(res["false_pattern_rate"])  # type: ignore[arg-type]
        per_class = res["per_class"]  # type: ignore[assignment]
        conf = res["confusion"]  # type: ignore[assignment]

        lines.extend(
            [
                "",
                f"## Results for n = {n} Defects (Random False-Pattern Rate: {fpr:.1f}%)",
                "",
                "### Per-Class Recall & Precision",
                "",
                "| Class | Recall (%) | Precision (%) | TP / True | Total Predicted |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for cls in CLASSES:
            m = per_class[cls]
            lines.append(
                f"| `{cls}` | {m['recall']:.1f} | {m['precision']:.1f} | "
                f"{int(m['tp'])} / {int(m['total_true'])} | {int(m['total_pred'])} |"
            )

        lines.extend(
            [
                "",
                "### Confusion Matrix (Rows = Ground Truth, Columns = Predicted)",
                "",
                "| Ground Truth \\ Predicted | " + " | ".join(short[c] for c in CLASSES) + " |",
                "| --- | " + " | ".join("---:" for _ in CLASSES) + " |",
            ]
        )
        for t in CLASSES:
            lines.append(
                f"| `{t}` | " + " | ".join(str(conf[t][p]) for p in CLASSES) + " |"
            )

    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote pattern accuracy report to {output_path}")


if __name__ == "__main__":
    results = run_pattern_benchmark()
    write_pattern_markdown(results, PROJECT_ROOT / "docs" / "pattern_accuracy.md")
