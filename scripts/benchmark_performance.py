"""Performance benchmark for WaferInspect full CV pipeline (Phase 8 Step 0b).

Measures wall time (median, min, max) and peak Python memory (tracemalloc) on
synthetic patterned wafers of sizes 512, 1024, 2048, and 4096 px for BOTH `golden`
and `self_reference` detection paths, 5 runs each.

Writes `docs/performance.md` and prints whether a default `max_side=2048` cap is
recommended based on the 4096 px runtime (> 15 s threshold) along with any
measurement-accuracy impact on a 4096 px wafer when downscaled to 2048 px.
"""

from datetime import datetime, timezone
import gc
from pathlib import Path
import sys
import time
import tracemalloc
from typing import Any, Dict, List
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.cv.pipeline import DetectionConfig, PreprocessingConfig, run_full_pipeline  # noqa: E402
from app.utils.synthetic import create_standard_synthetic_pair, generate_synthetic_wafer  # noqa: E402


def _make_defects_for_size(size: int) -> List[Dict[str, Any]]:
    scale = size / 512.0
    return [
        {"type": "bright_blob", "center": (165.0 * scale, 175.0 * scale), "radius": 8.0 * scale, "contrast": 65.0},
        {"type": "dark_blob", "center": (345.0 * scale, 185.0 * scale), "radius": 9.0 * scale, "contrast": -65.0},
        {
            "type": "scratch",
            "start": (190.0 * scale, 315.0 * scale),
            "end": (260.0 * scale, 345.0 * scale),
            "width": max(3, int(round(3 * scale))),
            "contrast": 70.0,
        },
    ]


def benchmark_size_and_path(
    size: int,
    path_name: str,
    runs: int = 5,
    max_side: int | None = None,
) -> Dict[str, Any]:
    defects = _make_defects_for_size(size)
    pitch = max(32, int(round(32 * (size / 512.0))))
    prep_cfg = PreprocessingConfig(max_side=max_side)
    det_cfg = DetectionConfig() if path_name == "golden" else DetectionConfig(no_reference_mode="self_reference")

    if path_name == "golden":
        ref_img, test_img, gt = create_standard_synthetic_pair(
            size=size, seed=8800, defects=defects, grid_pitch=pitch, die_pattern="grid"
        )
    else:
        test_img, gt = generate_synthetic_wafer(
            size=size, seed=8800, grid_pitch=pitch, die_pattern="grid", noise_std=5.0, defects=defects
        )
        ref_img = None

    times_s: List[float] = []
    peak_mb_list: List[float] = []
    last_res = None

    for _ in range(runs):
        gc.collect()
        tracemalloc.start()
        t0 = time.perf_counter()
        last_res = run_full_pipeline(
            test_image=test_img,
            reference_image=ref_img,
            prep_config=prep_cfg,
            det_config=det_cfg,
        )
        elapsed = time.perf_counter() - t0
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        times_s.append(elapsed)
        peak_mb_list.append(peak_bytes / (1024.0 * 1024.0))

    return {
        "size": size,
        "path": path_name,
        "max_side": max_side,
        "runs": runs,
        "median_s": float(np.median(times_s)),
        "min_s": float(np.min(times_s)),
        "max_s": float(np.max(times_s)),
        "peak_mb": float(np.max(peak_mb_list)),
        "defects_found": len(last_res.measurements) if last_res else 0,
        "scale_factor": last_res.scale_factor if last_res else 1.0,
        "gt": gt,
        "measurements": last_res.measurements if last_res else [],
    }


def main() -> None:
    sizes = [512, 1024, 2048, 4096]
    rows: List[Dict[str, Any]] = []

    print("=" * 92)
    print("WaferInspect Full-Pipeline Performance Benchmark (5 Runs per Cell, No Resize Cap)")
    print("=" * 92)
    print(f"{'Size':<8} | {'Path':<14} | {'Median (s)':<10} | {'Min (s)':<8} | {'Max (s)':<8} | {'Peak Mem (MB)':<13} | {'Found':<5}")
    print("-" * 92)

    for size in sizes:
        for path_name in ("golden", "self_reference"):
            rec = benchmark_size_and_path(size=size, path_name=path_name, runs=5, max_side=None)
            rows.append(rec)
            print(
                f"{size}x{size:<4} | {path_name:<14} | {rec['median_s']:<10.3f} | "
                f"{rec['min_s']:<8.3f} | {rec['max_s']:<8.3f} | {rec['peak_mb']:<13.1f} | {rec['defects_found']:<5}"
            )

    max_4096_s = max(r["median_s"] for r in rows if r["size"] == 4096)
    needs_cap = max_4096_s > 15.0
    print("=" * 92)
    print(f"Max 4096x4096 median runtime (uncapped): {max_4096_s:.3f} s -> max_side=2048 cap needed (>15s): {needs_cap}")

    # Also measure 4096 with max_side=2048 to document speed & measurement-accuracy impact
    capped_rows: List[Dict[str, Any]] = []
    accuracy_impact_lines: List[str] = []
    for path_name in ("golden", "self_reference"):
        uncapped = next(r for r in rows if r["size"] == 4096 and r["path"] == path_name)
        capped = benchmark_size_and_path(size=4096, path_name=path_name, runs=5, max_side=2048)
        capped_rows.append(capped)
        print(
            f"4096 cap=2048 | {path_name:<14} | {capped['median_s']:<10.3f} | "
            f"{capped['min_s']:<8.3f} | {capped['max_s']:<8.3f} | {capped['peak_mb']:<13.1f} | {capped['defects_found']:<5}"
        )

        # Compute mean relative area error vs true_area_visible_px for uncapped vs capped
        def _mean_area_err(rec: Dict[str, Any]) -> float:
            errs = []
            for d in rec["gt"]:
                gx, gy = d["center"]
                true_a = float(d["true_area_visible_px"])
                if not rec["measurements"]:
                    continue
                best = min(rec["measurements"], key=lambda m: np.hypot(m.centroid[0] - gx, m.centroid[1] - gy))
                errs.append(abs(best.area_px - true_a) / max(true_a, 1.0) * 100.0)
            return float(np.mean(errs)) if errs else 0.0

        err_uncapped = _mean_area_err(uncapped)
        err_capped = _mean_area_err(capped)
        accuracy_impact_lines.append(
            f"- **{path_name} (4096×4096)**: uncapped median `{uncapped['median_s']:.2f} s` (peak `{uncapped['peak_mb']:.1f} MB`, mean area error `{err_uncapped:.2f}%`) "
            f"vs `max_side=2048` (`scale=0.5`) median `{capped['median_s']:.2f} s` (peak `{capped['peak_mb']:.1f} MB`, mean area error `{err_capped:.2f}%`)."
        )

    # Write docs/performance.md
    out_path = PROJECT_ROOT / "docs" / "performance.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    decision_note = (
        "Because the uncapped 4096×4096 self-reference run exceeded 15 s, `PreprocessingConfig.max_side` "
        "defaults to `2048` px (downscaling images larger than 2048 px via `cv2.INTER_AREA` while leaving "
        "all images `<= 2048` px completely unchanged, and mapping all measurements back to original pixels via `scale_factor`)."
        if needs_cap
        else "All uncapped runs up to 4096×4096 completed well under 15 s, so `PreprocessingConfig.max_side` "
        "remains `None` by default (users can still set `max_side=2048` explicitly if desired; impact measured below)."
    )

    lines = [
        "# WaferInspect Full-Pipeline Performance Benchmark",
        "",
        f"Generated by `scripts/benchmark_performance.py` on {ts} (`5` runs per cell, Python `tracemalloc` peak memory).",
        "",
        "## Uncapped Full-Pipeline Timings & Peak Memory",
        "",
        "| Image Size | Path | Runs | Median Time (s) | Min (s) | Max (s) | Peak Python Memory (MB) | Defects Found |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['size']}×{r['size']} | `{r['path']}` | {r['runs']} | "
            f"{r['median_s']:.3f} | {r['min_s']:.3f} | {r['max_s']:.3f} | {r['peak_mb']:.1f} | {r['defects_found']} |"
        )

    lines.extend(
        [
            "",
            "## 4096×4096 `max_side=2048` Evaluation & Default Policy",
            "",
            decision_note,
            "",
            "| Image Size | Path | `max_side` | Scale | Median Time (s) | Peak Memory (MB) | Defects Found |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for r in capped_rows:
        lines.append(
            f"| 4096×4096 | `{r['path']}` | {r['max_side']} | {r['scale_factor']:.2f} | "
            f"{r['median_s']:.3f} | {r['peak_mb']:.1f} | {r['defects_found']} |"
        )

    lines.extend(["", "### Measurement-Accuracy Impact on 4096×4096", ""] + accuracy_impact_lines + [""])
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
