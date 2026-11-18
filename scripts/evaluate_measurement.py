"""Measurement accuracy benchmark for WaferInspect (Step 0c).

Evaluates half-maximum measurement refinement (OFF vs ON) across BOTH `golden` and
`self_reference` detection paths on:
  - Discs of diameter 8, 12, 16, 24 px
  - Scratches of length 40 and 80 px with thickness 2, 3, 5 px
  - Contrasts 30 and 60 gray levels
  - 20 deterministic seeds per cell

Reports median and 90th-percentile (p90) relative error (%) in area (against
`true_area_visible_px`), length, and thickness, plus centroid error (px), and writes
the full table to `docs/measurement_accuracy.md`.
"""

from datetime import datetime, timezone
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Literal, Tuple
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.cv.measurement import MeasurementConfig  # noqa: E402
from app.cv.noise_filter import NoiseFilterConfig  # noqa: E402
from app.cv.pipeline import DetectionConfig, PreprocessingConfig, run_full_pipeline  # noqa: E402
from app.utils.synthetic import create_standard_synthetic_pair, generate_synthetic_wafer  # noqa: E402


def _pct_stats(values: List[float]) -> Tuple[str, str]:
    if not values:
        return "N/A", "N/A"
    arr = np.array(values, dtype=np.float64)
    return f"{float(np.median(arr)):.1f}", f"{float(np.percentile(arr, 90)):.1f}"


def _px_stats(values: List[float]) -> Tuple[str, str]:
    if not values:
        return "N/A", "N/A"
    arr = np.array(values, dtype=np.float64)
    return f"{float(np.median(arr)):.2f}", f"{float(np.percentile(arr, 90)):.2f}"


def evaluate_cell(
    path_name: Literal["golden", "self_reference"],
    shape_label: str,
    defect_builder,
    contrast: float,
    refine: bool,
    num_seeds: int = 20,
    base_seed: int = 5000,
) -> Dict[str, Any]:
    """Run one (path, shape, contrast, refine) cell over `num_seeds` deterministic wafers."""
    area_errs: List[float] = []
    len_errs: List[float] = []
    thick_errs: List[float] = []
    centroid_errs: List[float] = []
    detected_count = 0

    prep_cfg = PreprocessingConfig()
    det_cfg = DetectionConfig() if path_name == "golden" else DetectionConfig(no_reference_mode="self_reference")
    filt_cfg = NoiseFilterConfig()
    meas_cfg = MeasurementConfig(refine_measurements=refine)

    for i in range(num_seeds):
        seed = base_seed + i
        defect_spec = defect_builder(i, contrast)

        if path_name == "golden":
            ref_img, test_img, gt = create_standard_synthetic_pair(
                size=512,
                seed=seed,
                defects=[defect_spec],
                grid_pitch=32,
                die_pattern="grid",
            )
            res = run_full_pipeline(
                test_image=test_img,
                reference_image=ref_img,
                prep_config=prep_cfg,
                det_config=det_cfg,
                filter_config=filt_cfg,
                measurement_config=meas_cfg,
            )
        else:
            test_img, gt = generate_synthetic_wafer(
                size=512,
                seed=seed,
                grid_pitch=32,
                die_pattern="grid",
                noise_std=5.0,
                defects=[defect_spec],
            )
            res = run_full_pipeline(
                test_image=test_img,
                reference_image=None,
                prep_config=prep_cfg,
                det_config=det_cfg,
                filter_config=filt_cfg,
                measurement_config=meas_cfg,
            )

        if not res.measurements or not gt:
            continue

        d = gt[0]
        gx, gy = d["center"]
        best_m = min(res.measurements, key=lambda m: math.hypot(m.centroid[0] - gx, m.centroid[1] - gy))
        c_err = math.hypot(best_m.centroid[0] - gx, best_m.centroid[1] - gy)

        # Match within reasonable radius of defect extent
        max_match = max(20.0, float(d.get("true_length_px", d.get("size", 16.0))) * 0.6)
        if c_err > max_match:
            continue

        detected_count += 1
        true_area = max(1.0, float(d["true_area_visible_px"]))
        if d["type"] == "scratch":
            # Use true rendered mask minAreaRect length/thickness (Step 0a) rather than nominal cv2.line args
            true_len = float(d.get("true_mask_length_px", d["true_length_px"]))
            true_thick = float(d.get("true_mask_thickness_px", d["true_thickness_px"]))
        else:
            true_len = float(d["size"])
            true_thick = float(d["size"])

        area_errs.append(abs(best_m.area_px - true_area) / true_area * 100.0)
        len_errs.append(abs(best_m.length_px - true_len) / max(true_len, 1e-6) * 100.0)
        thick_errs.append(abs(best_m.thickness_px - true_thick) / max(true_thick, 1e-6) * 100.0)
        centroid_errs.append(c_err)

    area_med, area_p90 = _pct_stats(area_errs)
    len_med, len_p90 = _pct_stats(len_errs)
    thick_med, thick_p90 = _pct_stats(thick_errs)
    cent_med, cent_p90 = _px_stats(centroid_errs)

    return {
        "path": path_name,
        "shape": shape_label,
        "contrast": contrast,
        "refine": "ON" if refine else "OFF",
        "det_rate": f"{detected_count}/{num_seeds}",
        "area_med": area_med,
        "area_p90": area_p90,
        "len_med": len_med,
        "len_p90": len_p90,
        "thick_med": thick_med,
        "thick_p90": thick_p90,
        "cent_med": cent_med,
        "cent_p90": cent_p90,
    }


def _make_disc_builder(diameter: int):
    r = diameter / 2.0

    def _build(i: int, contrast: float) -> Dict[str, Any]:
        cx = 256.0 + 48.0 * math.cos(i * 0.55)
        cy = 256.0 + 48.0 * math.sin(i * 0.55)
        return {"type": "bright_blob", "center": (cx, cy), "radius": r, "contrast": contrast}

    return _build


def _make_scratch_builder(length: int, thickness: int):
    half_l = length / 2.0

    def _build(i: int, contrast: float) -> Dict[str, Any]:
        angle = (i * 0.43) % math.pi
        cx = 256.0 + 30.0 * math.cos(i * 0.7)
        cy = 256.0 + 30.0 * math.sin(i * 0.7)
        dx = half_l * math.cos(angle)
        dy = half_l * math.sin(angle)
        return {
            "type": "scratch",
            "start": (cx - dx, cy - dy),
            "end": (cx + dx, cy + dy),
            "width": thickness,
            "contrast": contrast,
        }

    return _build


def run_measurement_benchmark(num_seeds: int = 20) -> List[Dict[str, Any]]:
    shapes = [
        (f"disc d={d}", _make_disc_builder(d)) for d in (8, 12, 16, 24)
    ] + [
        (f"scratch L={L} T={T}", _make_scratch_builder(L, T))
        for L in (40, 80)
        for T in (2, 3, 5)
    ]

    rows: List[Dict[str, Any]] = []
    header = (
        f"{'Path':<14} | {'Shape':<16} | {'Contr':<5} | {'Ref':<3} | {'Det':<5} | "
        f"{'Area Med/P90%':<13} | {'Len Med/P90%':<13} | {'Thk Med/P90%':<13} | {'Cent Med/P90px':<14}"
    )
    sep = "=" * len(header)
    print(sep)
    print(f"WaferInspect Measurement Accuracy Benchmark (Step 0c, {num_seeds} Seeds per Cell)")
    print(sep)
    print(header)
    print("-" * len(header))

    for path_name in ("golden", "self_reference"):
        for shape_label, builder in shapes:
            for contrast in (30.0, 60.0):
                for refine in (False, True):
                    row = evaluate_cell(
                        path_name=path_name,  # type: ignore[arg-type]
                        shape_label=shape_label,
                        defect_builder=builder,
                        contrast=contrast,
                        refine=refine,
                        num_seeds=num_seeds,
                    )
                    rows.append(row)
                    print(
                        f"{row['path']:<14} | {row['shape']:<16} | {row['contrast']:<5.0f} | "
                        f"{row['refine']:<3} | {row['det_rate']:<5} | "
                        f"{row['area_med'] + '/' + row['area_p90']:<13} | "
                        f"{row['len_med'] + '/' + row['len_p90']:<13} | "
                        f"{row['thick_med'] + '/' + row['thick_p90']:<13} | "
                        f"{row['cent_med'] + '/' + row['cent_p90']:<14}"
                    )
        print("-" * len(header))

    print(sep)
    return rows


def write_measurement_markdown(rows: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generated_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        "# WaferInspect Measurement Accuracy Benchmark (Refinement OFF vs ON)",
        "",
        f"Generated by `scripts/evaluate_measurement.py` on {generated_utc}.",
        "",
        "## Benchmark Setup",
        "",
        "- **Paths**: `golden` (reference difference) and `self_reference` (die-to-die median self-reference on periodic 32 px grid, `noise_std=5.0`).",
        "- **Shapes**: Discs of diameter `{8, 12, 16, 24}` px; scratches of length `{40, 80}` px × thickness `{2, 3, 5}` px.",
        "- **Contrasts**: `{30, 60}` gray levels, `20` seeded runs per cell.",
        "- **Refinement**: Half-maximum edge localisation (`0.5 * P`, where `P` is the mean of top 10% exact unshifted difference values inside the detection mask, local window padded 6 px, clipped to wafer mask, holes filled).",
        "- **Ground truth (Step 0a)**: Area relative error is measured against exact `true_area_visible_px` (clipped to wafer disk). For discs, length/thickness are compared against true diameter; for scratches, ground-truth length and thickness are computed via `cv2.minAreaRect` on the **generator's true rendered defect mask** (`true_mask_length_px`, `true_mask_thickness_px`) rather than nominal `cv2.line` parameters because OpenCV rasterises angled `cv2.line(..., thickness=T)` strokes with wider rounded pixel coverage than nominal `T`.",
        "- **Residual error on very thin low-contrast lines (T=2 px, contrast=30)**: Default 3×3 Gaussian denoising attenuates narrow line peaks and blurs sub-kernel-width strokes, causing partial detection and larger length/area error on thin low-contrast scratches.",
        "",
        "## Full Results Table (Median / P90 Errors)",
        "",
        "| Path | Shape | Contrast | Refine | Detected | Area Med (%) | Area P90 (%) | Length Med (%) | Length P90 (%) | Thick Med (%) | Thick P90 (%) | Centroid Med (px) | Centroid P90 (px) |",
        "| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for r in rows:
        lines.append(
            f"| {r['path']} | {r['shape']} | {r['contrast']:.0f} | {r['refine']} | {r['det_rate']} | "
            f"{r['area_med']} | {r['area_p90']} | {r['len_med']} | {r['len_p90']} | "
            f"{r['thick_med']} | {r['thick_p90']} | {r['cent_med']} | {r['cent_p90']} |"
        )

    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote measurement accuracy report to {output_path}")


if __name__ == "__main__":
    results = run_measurement_benchmark(num_seeds=20)
    write_measurement_markdown(results, PROJECT_ROOT / "docs" / "measurement_accuracy.md")
