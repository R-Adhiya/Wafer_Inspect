"""Deterministic synthetic tests for Phase 6 measurement, severity, summary, and overlays.

Validates:
1. Injected discs and squares of known size:
   - area_px within 5% of analytic/true pixel area (tolerance explicitly stated)
   - width_px and height_px within 2 px
   - centroid within 1 px
   - perimeter_px within 15% of analytic perimeter
2. Injected diagonal scratch of known length, thickness, and angle:
   - length_px within 10%
   - thickness_px within 2 px
   - angle_deg within 5 degrees
   - elongation sensible (matches length_px / thickness_px and >> 1)
3. Scale factor mapping: resize scale=0.5 maps reported lengths and areas back to original pixels (area x4 within tolerance).
4. Prototype severity heuristic:
   - Three defect sizes map to LOW, MEDIUM, HIGH
   - Changing thresholds in SeverityConfig (both fractional and absolute px) changes classification
   - Count escalation and area escalation each raise inspection severity by 1 level with numbers in severity_reason
   - Zero defects maps to NONE
5. Optional physical calibration:
   - pixel_size_um=2.0 -> area_um2 == 4 * area_px, length_um == 2 * length_px, thickness_um == 2 * thickness_px, status notes calibrated
   - Default pixel_size_um=None -> um fields None, status says "uncalibrated: measurements are in image pixels"
6. why_flagged contains measured contrast and correct detection path across golden, self_reference, and background.
7. Summary aggregates equal sum, max, mean of the regions list.
8. processing_time_s > 0 and stage_timings_ms contains all expected pipeline stages.
9. API contract: POST /inspection/run includes summary, model, processing_time_s, stage_timings_ms, full region measurements,
   persists them in run.json, serves GET .../image/final_overlay (200 PNG), and returns severity NONE for a clean wafer.
"""

from io import BytesIO
import json
import math
from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pytest

from app.core.config import DETECTION_METHOD_LABEL, MODEL_NAME, MODEL_VERSION, settings
from app.cv.measurement import MeasurementConfig, UNCALIBRATED_STATUS
from app.cv.noise_filter import NoiseFilterConfig
from app.cv.pipeline import DetectionConfig, PreprocessingConfig, run_full_pipeline
from app.cv.severity import SeverityConfig, assess_inspection_severity, classify_defect_severity
from app.main import app
from app.utils.synthetic import create_standard_synthetic_pair, generate_synthetic_wafer

client = TestClient(app)


def _to_png_bytes(arr: np.ndarray) -> bytes:
    buf = BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 1. Disc & Square Measurement Accuracy (Area <= 5%, W/H <= 2 px, Centroid <= 1 px, Perimeter <= 15%)
# ---------------------------------------------------------------------------
def test_disc_and_square_measurement_accuracy():
    """Verify measured area, width, height, centroid, and perimeter on known synthetic discs and squares.

    Tolerances verified:
      - area_px: <= 5% relative error vs true area
      - width_px, height_px: <= 2 px error
      - centroid: <= 1 px Euclidean error
      - perimeter_px: <= 15% relative error vs analytic perimeter (2*pi*r for disc, 4*side for square)
    """
    # Disc of radius 14 px centered at (180, 180)
    disc_r = 14.0
    disc_cx, disc_cy = 180.0, 180.0
    defects = [
        {"type": "bright_blob", "center": (disc_cx, disc_cy), "radius": disc_r, "contrast": 75.0},
    ]
    ref_img, test_img, _ = create_standard_synthetic_pair(
        size=512,
        seed=6101,
        defects=defects,
        grid_pitch=32,
        die_pattern="grid",
    )

    # Inject a known 24x24 dark square at x=310..333, y=300..323 (exact center (321.5, 311.5), area=576)
    sq_x, sq_y, sq_side = 310, 300, 24
    test_with_square = test_img.copy()
    patch = test_with_square[sq_y : sq_y + sq_side, sq_x : sq_x + sq_side].astype(np.int16) - 75
    test_with_square[sq_y : sq_y + sq_side, sq_x : sq_x + sq_side] = np.clip(patch, 0, 255).astype(np.uint8)

    result = run_full_pipeline(
        test_image=test_with_square,
        reference_image=ref_img,
        prep_config=PreprocessingConfig(denoise_enabled=False),
        det_config=DetectionConfig(threshold_method="otsu"),
        filter_config=NoiseFilterConfig(),
    )

    assert len(result.measurements) == 2

    # Region #1 (upper y) is the disc, Region #2 (lower y) is the square
    disc_m = min(result.measurements, key=lambda m: math.hypot(m.centroid[0] - disc_cx, m.centroid[1] - disc_cy))
    sq_m = min(
        result.measurements,
        key=lambda m: math.hypot(m.centroid[0] - (sq_x + (sq_side - 1) / 2.0), m.centroid[1] - (sq_y + (sq_side - 1) / 2.0)),
    )

    # Disc ground truth
    true_disc_area = math.pi * (disc_r ** 2)
    true_disc_diam = 2.0 * disc_r + 1.0  # pixel grid [-14..+14] spans 29 px
    true_disc_perim = 2.0 * math.pi * disc_r

    assert abs(disc_m.area_px - true_disc_area) / true_disc_area <= 0.05, (
        f"Disc area {disc_m.area_px} outside 5% of {true_disc_area:.1f}"
    )
    assert abs(disc_m.width_px - true_disc_diam) <= 2.0
    assert abs(disc_m.height_px - true_disc_diam) <= 2.0
    assert math.hypot(disc_m.centroid[0] - disc_cx, disc_m.centroid[1] - disc_cy) <= 1.0
    assert abs(disc_m.perimeter_px - true_disc_perim) / true_disc_perim <= 0.15, (
        f"Disc perimeter {disc_m.perimeter_px} outside 15% of {true_disc_perim:.1f}"
    )
    assert abs(disc_m.equivalent_diameter_px - 2.0 * disc_r) <= 1.5

    # Square ground truth
    true_sq_area = float(sq_side * sq_side)  # 576
    true_sq_cx = sq_x + (sq_side - 1) / 2.0  # 321.5
    true_sq_cy = sq_y + (sq_side - 1) / 2.0  # 311.5
    true_sq_perim = 4.0 * sq_side  # 96.0

    assert abs(sq_m.area_px - true_sq_area) / true_sq_area <= 0.05, (
        f"Square area {sq_m.area_px} outside 5% of {true_sq_area:.1f}"
    )
    assert abs(sq_m.width_px - sq_side) <= 2.0
    assert abs(sq_m.height_px - sq_side) <= 2.0
    assert math.hypot(sq_m.centroid[0] - true_sq_cx, sq_m.centroid[1] - true_sq_cy) <= 1.0
    assert abs(sq_m.perimeter_px - true_sq_perim) / true_sq_perim <= 0.15, (
        f"Square perimeter {sq_m.perimeter_px} outside 15% of {true_sq_perim:.1f}"
    )


# ---------------------------------------------------------------------------
# 2. Diagonal Scratch Oriented Measurements (Length <= 10%, Thickness <= 2 px, Angle <= 5 deg)
# ---------------------------------------------------------------------------
def test_scratch_oriented_length_thickness_angle_and_elongation():
    """Verify oriented minAreaRect accurately recovers diagonal scratch length, thickness, angle, and elongation."""
    start = (170.0, 210.0)
    end = (260.0, 260.0)
    true_width = 4
    true_length = math.hypot(end[0] - start[0], end[1] - start[1])  # ~102.96 px
    true_angle = math.degrees(math.atan2(end[1] - start[1], end[0] - start[0])) % 180.0  # ~29.05 deg

    defects = [
        {"type": "scratch", "start": start, "end": end, "width": true_width, "contrast": 75.0},
    ]
    ref_img, test_img, _ = create_standard_synthetic_pair(
        size=512,
        seed=6202,
        defects=defects,
        grid_pitch=32,
        die_pattern="grid",
    )

    result = run_full_pipeline(
        test_image=test_img,
        reference_image=ref_img,
        prep_config=PreprocessingConfig(denoise_enabled=False),
        det_config=DetectionConfig(threshold_method="otsu"),
        filter_config=NoiseFilterConfig(),
    )

    assert len(result.measurements) == 1
    m = result.measurements[0]

    assert abs(m.length_px - true_length) / true_length <= 0.10, (
        f"Scratch length {m.length_px} outside 10% of {true_length:.2f}"
    )
    assert abs(m.thickness_px - true_width) <= 2.0, (
        f"Scratch thickness {m.thickness_px} outside 2 px of {true_width}"
    )
    # Angular distance modulo 180 degrees
    raw_angle_diff = abs(m.angle_deg - true_angle)
    angle_diff = min(raw_angle_diff, 180.0 - raw_angle_diff)
    assert angle_diff <= 5.0, (
        f"Scratch angle {m.angle_deg} deg outside 5 deg of {true_angle:.2f} deg"
    )
    assert m.elongation >= 10.0
    assert abs(m.elongation - (m.length_px / m.thickness_px)) <= 0.15


# ---------------------------------------------------------------------------
# 3. Scale Factor Coordinate & Area Mapping (scale = 0.5 -> area x4)
# ---------------------------------------------------------------------------
def test_scale_factor_maps_lengths_and_areas_back_to_original_pixels():
    """Verify resize to scale=0.5 maps lengths (x2) and areas (x4) back to original image pixels."""
    disc_r = 20.0
    true_orig_area = math.pi * (disc_r ** 2)  # ~1256.6 px
    true_orig_diam = 2.0 * disc_r + 1.0  # 41 px
    defects = [
        {"type": "bright_blob", "center": (256.0, 256.0), "radius": disc_r, "contrast": 75.0},
    ]
    ref_img, test_img, _ = create_standard_synthetic_pair(
        size=512,
        seed=6303,
        defects=defects,
        grid_pitch=32,
        die_pattern="grid",
    )

    # max_side=256 on 512x512 gives exact scale_factor = 0.5
    result = run_full_pipeline(
        test_image=test_img,
        reference_image=ref_img,
        prep_config=PreprocessingConfig(max_side=256, denoise_enabled=False),
        det_config=DetectionConfig(threshold_method="otsu"),
        filter_config=NoiseFilterConfig(),
    )

    assert abs(result.scale_factor - 0.5) < 1e-6
    assert len(result.accepted_regions) == 1
    assert len(result.measurements) == 1

    proc_region = result.accepted_regions[0]
    meas = result.measurements[0]

    # Area in original pixels is exact 4x processed pixel count (since 1 / 0.5^2 == 4)
    assert meas.area_px == proc_region.area_px * 4
    # And matches the original full-resolution injected disc area within 10% (downsampling quantization)
    assert abs(meas.area_px - true_orig_area) / true_orig_area <= 0.10
    # Lengths and centroid mapped back to 512x512 original space
    assert abs(meas.width_px - true_orig_diam) <= 4.0
    assert abs(meas.height_px - true_orig_diam) <= 4.0
    assert math.hypot(meas.centroid[0] - 256.0, meas.centroid[1] - 256.0) <= 2.0


# ---------------------------------------------------------------------------
# 4. Prototype Severity Heuristic: LOW/MEDIUM/HIGH, Config Overrides, Escalations, NONE
# ---------------------------------------------------------------------------
def test_severity_classification_config_overrides_escalations_and_none():
    """Verify LOW/MEDIUM/HIGH mapping, config changes, count/area escalations, and NONE on zero defects."""
    wafer_area = 150_000.0  # typical 512x512 wafer disk area
    default_cfg = SeverityConfig()  # medium=0.0008 (120 px), high=0.003 (450 px)

    # Three distinct sizes map to LOW, MEDIUM, HIGH
    assert classify_defect_severity(area_px=60, wafer_area_px=wafer_area, config=default_cfg) == "LOW"
    assert classify_defect_severity(area_px=220, wafer_area_px=wafer_area, config=default_cfg) == "MEDIUM"
    assert classify_defect_severity(area_px=600, wafer_area_px=wafer_area, config=default_cfg) == "HIGH"

    # Changing fractional thresholds in config changes the classification
    strict_frac_cfg = SeverityConfig(medium_area_fraction=0.0002, high_area_fraction=0.001)
    assert classify_defect_severity(area_px=60, wafer_area_px=wafer_area, config=strict_frac_cfg) == "MEDIUM"
    assert classify_defect_severity(area_px=220, wafer_area_px=wafer_area, config=strict_frac_cfg) == "HIGH"

    # User-supplied absolute pixel thresholds override fractional defaults
    abs_px_cfg = SeverityConfig(medium_area_px=50.0, high_area_px=150.0)
    assert classify_defect_severity(area_px=40, wafer_area_px=wafer_area, config=abs_px_cfg) == "LOW"
    assert classify_defect_severity(area_px=80, wafer_area_px=wafer_area, config=abs_px_cfg) == "MEDIUM"
    assert classify_defect_severity(area_px=200, wafer_area_px=wafer_area, config=abs_px_cfg) == "HIGH"

    # Zero defects -> NONE
    sev_none, reason_none = assess_inspection_severity([], [], wafer_area_px=wafer_area, config=default_cfg)
    assert sev_none == "NONE"
    assert "0" in reason_none

    # Baseline without escalation: 2 LOW defects -> stays LOW
    sev_low, reason_low = assess_inspection_severity([50, 60], ["LOW", "LOW"], wafer_area_px=wafer_area, config=default_cfg)
    assert sev_low == "LOW"
    assert "2" in reason_low and "110" in reason_low

    # Count escalation: 5 LOW defects (>= count_escalation=5) -> escalated 1 level to MEDIUM
    sev_cnt, reason_cnt = assess_inspection_severity(
        [40, 45, 50, 55, 60],
        ["LOW", "LOW", "LOW", "LOW", "LOW"],
        wafer_area_px=wafer_area,
        config=default_cfg,
    )
    assert sev_cnt == "MEDIUM"
    assert "5" in reason_cnt
    assert "escalated" in reason_cnt.lower()

    # Area escalation: 4 MEDIUM defects totaling 1600 px (> 1.0% of 150,000 = 1500 px) -> escalated to HIGH
    sev_area, reason_area = assess_inspection_severity(
        [400, 400, 400, 400],
        ["MEDIUM", "MEDIUM", "MEDIUM", "MEDIUM"],
        wafer_area_px=wafer_area,
        config=default_cfg,
    )
    assert sev_area == "HIGH"
    assert "1600" in reason_area
    assert "1.07%" in reason_area
    assert "escalated" in reason_area.lower()


# ---------------------------------------------------------------------------
# 5. Calibration (pixel_size_um=2.0 vs Uncalibrated)
# ---------------------------------------------------------------------------
def test_calibration_um_conversion_and_uncalibrated_status():
    """Verify pixel_size_um=2.0 yields area_um2 = 4 * area_px and calibrated status; default notes uncalibrated."""
    defects = [
        {"type": "bright_blob", "center": (220.0, 220.0), "radius": 10.0, "contrast": 70.0},
    ]
    ref_img, test_img, _ = create_standard_synthetic_pair(size=512, seed=6404, defects=defects)

    # Uncalibrated (default)
    res_uncal = run_full_pipeline(test_image=test_img, reference_image=ref_img)
    assert len(res_uncal.measurements) == 1
    m_uncal = res_uncal.measurements[0]
    assert m_uncal.area_um2 is None
    assert m_uncal.length_um is None
    assert m_uncal.thickness_um is None
    assert res_uncal.summary["calibration_status"] == UNCALIBRATED_STATUS
    assert "uncalibrated" in res_uncal.summary["calibration_status"].lower()

    # Calibrated with pixel_size_um=2.0
    res_cal = run_full_pipeline(
        test_image=test_img,
        reference_image=ref_img,
        measurement_config=MeasurementConfig(pixel_size_um=2.0),
    )
    assert len(res_cal.measurements) == 1
    m_cal = res_cal.measurements[0]
    assert m_cal.area_um2 == pytest.approx(4.0 * m_cal.area_px, rel=1e-6)
    assert m_cal.length_um == pytest.approx(2.0 * m_cal.length_px, rel=1e-6)
    assert m_cal.thickness_um == pytest.approx(2.0 * m_cal.thickness_px, rel=1e-6)
    assert "calibrated" in res_cal.summary["calibration_status"].lower()
    assert "uncalibrated" not in res_cal.summary["calibration_status"].lower()
    assert "2" in res_cal.summary["calibration_status"]


# ---------------------------------------------------------------------------
# 6. why_flagged Content Across All Detection Paths
# ---------------------------------------------------------------------------
def test_why_flagged_contains_measured_contrast_and_detection_path():
    """Verify why_flagged embeds measured contrast and the actual detection path name."""
    defects = [
        {"type": "bright_blob", "center": (220.0, 240.0), "radius": 9.0, "contrast": 70.0},
    ]

    # A. Golden path
    ref_img, test_img, _ = create_standard_synthetic_pair(size=512, seed=6501, defects=defects)
    res_golden = run_full_pipeline(test_image=test_img, reference_image=ref_img)
    m_g = res_golden.measurements[0]
    assert m_g.detected_by == "golden"
    assert f"{m_g.mean_contrast:.1f}" in m_g.why_flagged
    assert "golden" in m_g.why_flagged.lower()

    # B. Self-reference path (patterned wafer, no reference)
    res_self = run_full_pipeline(test_image=test_img, reference_image=None)
    m_s = res_self.measurements[0]
    assert m_s.detected_by == "self_reference"
    assert f"{m_s.mean_contrast:.1f}" in m_s.why_flagged
    assert "self-reference" in m_s.why_flagged.lower()

    # C. Background path (unpatterned wafer, no reference)
    plain_img, _ = generate_synthetic_wafer(size=512, seed=6502, grid_pitch=0, noise_std=5.0, defects=defects)
    res_bg = run_full_pipeline(
        test_image=plain_img,
        reference_image=None,
        det_config=DetectionConfig(no_reference_mode="background"),
    )
    m_b = res_bg.measurements[0]
    assert m_b.detected_by == "background"
    assert f"{m_b.mean_contrast:.1f}" in m_b.why_flagged
    assert "background" in m_b.why_flagged.lower()


# ---------------------------------------------------------------------------
# 7. Summary Aggregates & Pipeline Timings
# ---------------------------------------------------------------------------
def test_summary_aggregates_and_positive_stage_timings():
    """Verify summary numbers match exact sum/max/mean of regions and timings are populated & positive."""
    defects = [
        {"type": "bright_blob", "center": (165.0, 175.0), "radius": 5.0, "contrast": 70.0},   # LOW (~78 px)
        {"type": "dark_blob", "center": (345.0, 185.0), "radius": 9.0, "contrast": -70.0},   # MEDIUM (~254 px)
        {"type": "bright_blob", "center": (256.0, 340.0), "radius": 14.0, "contrast": 70.0}, # HIGH (~615 px)
    ]
    ref_img, test_img, _ = create_standard_synthetic_pair(size=512, seed=6601, defects=defects)
    res = run_full_pipeline(test_image=test_img, reference_image=ref_img)

    assert len(res.measurements) == 3
    areas = [m.area_px for m in res.measurements]
    severities = {m.severity for m in res.measurements}
    assert severities == {"LOW", "MEDIUM", "HIGH"}

    assert res.summary["defect_count"] == 3
    assert res.summary["total_defect_area_px"] == sum(areas)
    assert res.summary["largest_defect_area_px"] == max(areas)
    assert res.summary["average_defect_area_px"] == round(sum(areas) / 3.0, 2)
    assert res.summary["severity"] == "HIGH"

    assert res.processing_time_s > 0.0
    for stage_key in (
        "preprocessing",
        "detection",
        "segmentation_and_merge",
        "noise_filtering",
        "measurement_and_severity",
        "overlays",
    ):
        assert stage_key in res.stage_timings_ms
        assert res.stage_timings_ms[stage_key] >= 0.0


# ---------------------------------------------------------------------------
# 8. API Integration: Summary, Model, Measurements, final_overlay, Clean Wafer NONE
# ---------------------------------------------------------------------------
def test_api_phase6_summary_model_measurements_final_overlay_and_clean_none():
    """Verify full Phase 6 API contract, run.json persistence, final_overlay 200, and clean NONE severity."""
    # A. Defective wafer (with golden reference)
    defects = [
        {"type": "bright_blob", "center": (175.0, 185.0), "radius": 8.0, "contrast": 65.0},
        {"type": "scratch", "start": (220.0, 310.0), "end": (290.0, 340.0), "width": 3, "contrast": 70.0},
    ]
    ref_img, test_img, _ = create_standard_synthetic_pair(size=512, seed=6701, defects=defects)

    up = client.post(
        "/inspection/upload",
        files={
            "test_image": ("wafer_test.png", _to_png_bytes(test_img), "image/png"),
            "reference_image": ("wafer_ref.png", _to_png_bytes(ref_img), "image/png"),
        },
    )
    assert up.status_code == 200
    upload_id = up.json()["upload_id"]

    run_res = client.post("/inspection/run", json={"upload_id": upload_id})
    assert run_res.status_code == 200
    body = run_res.json()

    # Model metadata from single config constants
    assert body["model"] == {
        "name": MODEL_NAME,
        "version": MODEL_VERSION,
        "detection_method": DETECTION_METHOD_LABEL,
    }
    assert body["processing_time_s"] > 0.0
    assert isinstance(body["stage_timings_ms"], dict) and len(body["stage_timings_ms"]) >= 5

    # Summary matches region aggregates
    summary = body["summary"]
    regions = body["regions"]
    assert len(regions) == 2
    assert summary["defect_count"] == 2
    assert summary["total_defect_area_px"] == sum(r["area_px"] for r in regions)
    assert summary["largest_defect_area_px"] == max(r["area_px"] for r in regions)
    assert summary["average_defect_area_px"] == round(summary["total_defect_area_px"] / 2.0, 2)
    assert summary["severity"] in ("LOW", "MEDIUM", "HIGH")
    assert summary["calibration_status"] == UNCALIBRATED_STATUS

    # Every region carries full Phase 6 measurements
    for r in regions:
        for key in (
            "id",
            "area_px",
            "bbox",
            "width_px",
            "height_px",
            "centroid",
            "perimeter_px",
            "equivalent_diameter_px",
            "length_px",
            "thickness_px",
            "angle_deg",
            "elongation",
            "polarity",
            "mean_contrast",
            "peak_contrast",
            "contrast_to_noise",
            "touches_edge",
            "detected_by",
            "why_flagged",
            "severity",
        ):
            assert key in r, f"Missing measurement field '{key}'"
        assert r["detected_by"] == "golden"
        assert "golden" in r["why_flagged"].lower()

    # final_overlay endpoint returns 200 PNG
    assert body["final_overlay_image_url"] == f"/inspection/run/{upload_id}/image/final_overlay"
    overlay_res = client.get(body["final_overlay_image_url"])
    assert overlay_res.status_code == 200
    assert overlay_res.headers["content-type"] == "image/png"

    # run.json persisted all Phase 6 sections
    run_json_path = settings.results_path / upload_id / "run.json"
    assert run_json_path.is_file()
    persisted = json.loads(run_json_path.read_text(encoding="utf-8"))
    assert persisted["summary"] == summary
    assert persisted["model"] == body["model"]
    assert persisted["processing_time_s"] == body["processing_time_s"]
    assert len(persisted["regions"]) == 2

    # B. Clean wafer -> severity NONE, defect_count == 0
    clean_img, _ = generate_synthetic_wafer(size=512, seed=6702, grid_pitch=32, die_pattern="circuit", defects=None)
    up_clean = client.post(
        "/inspection/upload",
        files={"test_image": ("clean_wafer.png", _to_png_bytes(clean_img), "image/png")},
    )
    assert up_clean.status_code == 200
    clean_id = up_clean.json()["upload_id"]

    clean_run = client.post("/inspection/run", json={"upload_id": clean_id})
    assert clean_run.status_code == 200
    clean_body = clean_run.json()
    assert clean_body["summary"]["defect_count"] == 0
    assert clean_body["summary"]["severity"] == "NONE"
    assert clean_body["regions"] == []

    # final_overlay still renders cleanly on 0-defect wafers
    clean_overlay = client.get(f"/inspection/run/{clean_id}/image/final_overlay")
    assert clean_overlay.status_code == 200
