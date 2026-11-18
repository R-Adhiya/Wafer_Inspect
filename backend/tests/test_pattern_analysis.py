"""Deterministic seeded tests for Phase 6B wafer-level spatial pattern diagnosis.

Validates:
1. Unit tests on synthetic centroids (8 points per pattern): EDGE_RING, CENTER_CLUSTER,
   DONUT, SCRATCH_LINE, LOCAL_CLUSTER each classified accurately with p <= 0.01.
2. Random layouts over 100 seeds: false-pattern rate bounded <= 5% (stated bound).
3. n = 2 non-elongated gives INSUFFICIENT_DATA; single elongated scratch gives SCRATCH_LINE
   with shape-based explanation.
4. Determinism across identical runs.
5. Explanation changes with input and embeds measured numbers.
6. Scale factor mapping (scale = 0.5) correctly transforms wafer center/radius to original pixels.
7. Full-frame fallback uses inscribed circle and adds "assumed wafer geometry" warning.
8. End-to-end: all 6 layouts on patterned wafers WITHOUT a reference return matching patterns.
9. API contract: run response has `pattern` and `summary.pattern`, `pattern_overlay` returns 200 PNG,
   and clean wafer returns `INSUFFICIENT_DATA`.
"""

from io import BytesIO
import json
from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pytest

from app.core.config import settings
from app.cv.pattern_analysis import PatternConfig, analyze_spatial_pattern
from app.cv.pipeline import DetectionConfig, PreprocessingConfig, run_full_pipeline
from app.main import app
from app.utils.synthetic import (
    DefectLayout,
    LAYOUT_TO_PATTERN_LABEL,
    generate_layout_centroids,
    generate_layout_wafer,
    generate_synthetic_wafer,
)

client = TestClient(app)


def _centroids_to_records(centroids):
    return [
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


def _to_png_bytes(arr: np.ndarray) -> bytes:
    buf = BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 1. Unit Tests: 8 Points per Pattern -> Correct Classification with p <= 0.01
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "layout",
    ["edge_ring", "center_cluster", "donut", "scratch_line", "local_cluster"],
)
def test_pattern_analysis_8_points_each_pattern_p_le_001(layout: DefectLayout):
    """Verify each of the 5 structured patterns (8 synthetic centroids) is classified with p <= 0.01."""
    center = (256.0, 256.0)
    radius = 225.28
    centroids, expected_label = generate_layout_centroids(
        layout=layout,
        count=8,
        center=center,
        radius=radius,
        seed=7001,
    )
    res = analyze_spatial_pattern(
        measurements=_centroids_to_records(centroids),
        wafer_center=center,
        wafer_radius=radius,
        image_size=(512, 512),
        scale_factor=1.0,
        config=PatternConfig(mc_trials=2000, mc_seed=42),
    )

    assert res.type == expected_label, f"Expected {expected_label}, got {res.type}: {res.explanation}"
    assert res.p_value is not None and res.p_value <= 0.01
    assert abs(res.confidence - round(1.0 - res.p_value, 4)) < 1e-6
    assert len(res.supporting_defect_ids) >= 6


# ---------------------------------------------------------------------------
# 2. Random Layouts Over 100 Seeds: False-Pattern Rate <= 5%
# ---------------------------------------------------------------------------
def test_random_layouts_100_seeds_low_false_pattern_rate():
    """Verify false-pattern rate across 100 seeded uniform random layouts (n=8) is <= 5%."""
    center = (256.0, 256.0)
    radius = 225.28
    cfg = PatternConfig(mc_trials=2000, mc_seed=42)

    false_patterns = 0
    num_seeds = 100
    for seed in range(9000, 9000 + num_seeds):
        centroids, _ = generate_layout_centroids(
            layout="random",
            count=8,
            center=center,
            radius=radius,
            seed=seed,
            min_separation_px=0.0,
        )
        res = analyze_spatial_pattern(
            measurements=_centroids_to_records(centroids),
            wafer_center=center,
            wafer_radius=radius,
            image_size=(512, 512),
            config=cfg,
        )
        if res.type != "RANDOM_SCATTER":
            false_patterns += 1

    # Stated bound: <= 5% (expected ~1%-5% under 5 union tests at alpha=0.01)
    assert false_patterns <= 5, f"False-pattern rate {false_patterns}/{num_seeds} exceeded 5% bound"


# ---------------------------------------------------------------------------
# 3. n = 2 Non-Elongated -> INSUFFICIENT_DATA; Single Elongated Scratch -> SCRATCH_LINE
# ---------------------------------------------------------------------------
def test_insufficient_data_and_single_elongated_scratch_exception():
    """Verify n=2 blobs give INSUFFICIENT_DATA while a single long thin scratch gives SCRATCH_LINE."""
    center = (256.0, 256.0)
    radius = 225.28  # diameter = 450.56 -> 15% = 67.6 px

    two_blobs = [
        {"id": 1, "centroid": (200.0, 200.0), "area_px": 150, "length_px": 14.0, "elongation": 1.1, "severity": "MEDIUM"},
        {"id": 2, "centroid": (300.0, 280.0), "area_px": 160, "length_px": 15.0, "elongation": 1.2, "severity": "MEDIUM"},
    ]
    res_insuff = analyze_spatial_pattern(
        measurements=two_blobs,
        wafer_center=center,
        wafer_radius=radius,
        image_size=(512, 512),
    )
    assert res_insuff.type == "INSUFFICIENT_DATA"
    assert "2" in res_insuff.explanation

    single_scratch = [
        {"id": 1, "centroid": (256.0, 256.0), "area_px": 320, "length_px": 95.0, "elongation": 19.0, "severity": "MEDIUM"},
    ]
    res_scratch = analyze_spatial_pattern(
        measurements=single_scratch,
        wafer_center=center,
        wafer_radius=radius,
        image_size=(512, 512),
    )
    assert res_scratch.type == "SCRATCH_LINE"
    assert "shape" in res_scratch.explanation.lower()
    assert "19.0" in res_scratch.explanation
    assert "95.0" in res_scratch.explanation
    assert res_scratch.supporting_defect_ids == [1]


# ---------------------------------------------------------------------------
# 4. Determinism & Explanation Changes with Input
# ---------------------------------------------------------------------------
def test_pattern_determinism_and_explanation_reflects_input():
    """Verify identical runs produce identical outputs and changing input changes explanation with numbers."""
    center = (256.0, 256.0)
    radius = 225.28
    cfg = PatternConfig(mc_trials=2000, mc_seed=42)

    pts_edge, _ = generate_layout_centroids("edge_ring", count=7, center=center, radius=radius, seed=8101)
    pts_donut, _ = generate_layout_centroids("donut", count=8, center=center, radius=radius, seed=8102)

    r1 = analyze_spatial_pattern(_centroids_to_records(pts_edge), center, radius, (512, 512), config=cfg)
    r2 = analyze_spatial_pattern(_centroids_to_records(pts_edge), center, radius, (512, 512), config=cfg)
    r3 = analyze_spatial_pattern(_centroids_to_records(pts_donut), center, radius, (512, 512), config=cfg)

    assert r1.type == r2.type == "EDGE_RING"
    assert r1.p_value == r2.p_value
    assert r1.confidence == r2.confidence
    assert r1.explanation == r2.explanation
    assert r1.metrics == r2.metrics

    assert r3.type == "DONUT"
    assert r3.explanation != r1.explanation
    assert "7 of 7" in r1.explanation
    assert "8 of 8" in r3.explanation


# ---------------------------------------------------------------------------
# 5. Scale Factor Mapping & Full-Frame Warning
# ---------------------------------------------------------------------------
def test_scale_factor_mapping_and_full_frame_warning():
    """Verify scale=0.5 maps processed wafer geometry to original coords and full-frame adds warning."""
    # Original 512x512 wafer has center (256, 256), radius 225.28.
    # At scale_factor = 0.5, detection reports processed center (128, 128) and radius 112.64,
    # while measurements are already in ORIGINAL 512x512 coordinates.
    centroids, _ = generate_layout_centroids(
        layout="edge_ring",
        count=8,
        center=(256.0, 256.0),
        radius=225.28,
        seed=8201,
    )
    res_scaled = analyze_spatial_pattern(
        measurements=_centroids_to_records(centroids),
        wafer_center=(128.0, 128.0),
        wafer_radius=112.64,
        image_size=(512, 512),
        scale_factor=0.5,
        is_full_frame=False,
    )
    assert res_scaled.type == "EDGE_RING"
    assert res_scaled.metrics["wafer_center_orig"] == (256.0, 256.0)
    assert abs(res_scaled.metrics["wafer_radius_orig"] - 225.28) < 0.05
    assert res_scaled.warnings == []

    # Full-frame fallback uses inscribed circle and adds "assumed wafer geometry" warning
    res_ff = analyze_spatial_pattern(
        measurements=_centroids_to_records(centroids),
        wafer_center=(100.0, 100.0),
        wafer_radius=50.0,
        image_size=(512, 512),
        scale_factor=1.0,
        is_full_frame=True,
    )
    assert any("assumed wafer geometry" in w.lower() for w in res_ff.warnings)
    assert res_ff.metrics["wafer_center_orig"] == (256.0, 256.0)
    assert res_ff.metrics["wafer_radius_orig"] == 256.0


# ---------------------------------------------------------------------------
# 6. End-to-End: All 6 Layouts on Patterned Wafers (NO Reference)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "layout",
    ["random", "edge_ring", "center_cluster", "donut", "scratch_line", "local_cluster"],
)
def test_end_to_end_all_six_layouts_no_reference_patterned_wafer(layout: DefectLayout):
    """Verify full pipeline (no reference, patterned wafer) diagnoses all 6 spatial layouts accurately."""
    img, gt, expected_label = generate_layout_wafer(
        defect_layout=layout,
        count=8,
        size=512,
        seed=8305,
        grid_pitch=32,
        die_pattern="grid",
        noise_std=5.0,
        radius=7.0,
        contrast=65.0,
    )
    res = run_full_pipeline(
        test_image=img,
        reference_image=None,
        det_config=DetectionConfig(no_reference_mode="auto"),
    )

    assert res.detection.path == "self_reference"
    assert len(res.measurements) == len(gt) == 8
    assert res.pattern.type == expected_label, (
        f"Layout '{layout}': expected {expected_label}, got {res.pattern.type} "
        f"(p={res.pattern.p_value}, expl='{res.pattern.explanation}')"
    )
    if expected_label != "RANDOM_SCATTER":
        assert res.pattern.p_value is not None and res.pattern.p_value <= 0.01


# ---------------------------------------------------------------------------
# 7. API Contract: Pattern in Response & Summary, pattern_overlay 200, Clean -> INSUFFICIENT_DATA
# ---------------------------------------------------------------------------
def test_api_pattern_response_overlay_and_clean_insufficient_data():
    """Verify POST /inspection/run includes pattern & timing, serves pattern_overlay 200, and clean gives INSUFFICIENT_DATA."""
    # A. Edge-ring patterned wafer (no reference)
    img, _, expected_label = generate_layout_wafer(
        defect_layout="edge_ring",
        count=8,
        size=512,
        seed=8401,
        grid_pitch=32,
        die_pattern="grid",
    )
    up = client.post(
        "/inspection/upload",
        files={"test_image": ("edge_ring_wafer.png", _to_png_bytes(img), "image/png")},
    )
    assert up.status_code == 200
    upload_id = up.json()["upload_id"]

    run_res = client.post("/inspection/run", json={"upload_id": upload_id})
    assert run_res.status_code == 200
    body = run_res.json()

    assert body["pattern"]["type"] == expected_label
    assert body["summary"]["pattern"]["type"] == expected_label
    assert body["pattern"]["p_value"] <= 0.01
    assert "radial_bins" in body["pattern"]["metrics"]
    assert len(body["pattern"]["metrics"]["radial_bins"]) == 5
    assert "pattern_analysis" in body["stage_timings_ms"]
    assert body["pattern_overlay_image_url"] == f"/inspection/run/{upload_id}/image/pattern_overlay"

    # pattern_overlay endpoint returns 200 PNG
    ov_res = client.get(body["pattern_overlay_image_url"])
    assert ov_res.status_code == 200
    assert ov_res.headers["content-type"] == "image/png"

    # Persisted in run.json
    persisted = json.loads((settings.results_path / upload_id / "run.json").read_text(encoding="utf-8"))
    assert persisted["pattern"]["type"] == expected_label
    assert persisted["summary"]["pattern"]["type"] == expected_label

    # B. Clean wafer -> INSUFFICIENT_DATA and severity NONE
    clean_img, _ = generate_synthetic_wafer(size=512, seed=8402, grid_pitch=32, die_pattern="circuit", defects=None)
    up_clean = client.post(
        "/inspection/upload",
        files={"test_image": ("clean_wafer.png", _to_png_bytes(clean_img), "image/png")},
    )
    assert up_clean.status_code == 200
    clean_id = up_clean.json()["upload_id"]

    clean_run = client.post("/inspection/run", json={"upload_id": clean_id})
    assert clean_run.status_code == 200
    clean_body = clean_run.json()
    assert clean_body["summary"]["severity"] == "NONE"
    assert clean_body["pattern"]["type"] == "INSUFFICIENT_DATA"
    assert client.get(f"/inspection/run/{clean_id}/image/pattern_overlay").status_code == 200
