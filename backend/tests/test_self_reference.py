"""Deterministic synthetic tests for Phase 5B die-to-die self-reference detection.

Validates:
1. Exact integer pitch recovery for pitches 32, 40, and 48 in both 'grid' and 'circuit' patterns across seeds.
2. Non-periodic plain noisy wafer triggers 'not periodic' and auto-mode falls back to background with warning numbers.
3. 4 injected defects (bright, dark, scratch, near-rim) all detected within centroid tolerance in both die patterns.
4. Clean patterned wafers produce 0 final defects across 20 seeds in both 'grid' and 'circuit' patterns.
5. Illumination gradient alone on a patterned wafer produces 0 final defects.
6. Verifiable coverage & unverifiable border exclusion: covered defect found, unverifiable border excluded & reported.
7. Funnel honesty notes: non-empty note when a stage increases candidate count; non-increasing on self-reference path.
8. Determinism across identical runs.
9. API response includes path, label, die_pitch, peak_strength, coverage, and unverifiable_fraction.
"""

from io import BytesIO
from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pytest

from app.cv.detection import estimate_wafer_mask
from app.cv.noise_filter import NoiseFilterConfig
from app.cv.pipeline import DetectionConfig, run_detection, run_full_pipeline
from app.cv.self_reference import build_self_reference, estimate_die_pitch
from app.main import app
from app.utils.synthetic import generate_synthetic_wafer

client = TestClient(app)


# ---------------------------------------------------------------------------
# 1. Exact Pitch Estimation (32, 40, 48 in grid and circuit across seeds)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("die_pattern", ["grid", "circuit"])
@pytest.mark.parametrize("pitch", [32, 40, 48])
def test_pitch_estimation_exact_across_patterns_and_seeds(die_pattern: str, pitch: int):
    """Verify FFT autocorrelation recovers true integer pitch exactly for 32, 40, 48 across seeds."""
    for seed in (10, 42, 105):
        img, _ = generate_synthetic_wafer(
            size=512,
            seed=seed,
            grid_pitch=pitch,
            die_pattern=die_pattern,
            noise_std=5.0,
        )
        wafer_mask, _, _, _ = estimate_wafer_mask(img)
        est = estimate_die_pitch(img, wafer_mask)

        assert est.periodic is True, f"Failed periodicity for {die_pattern} pitch={pitch} seed={seed}: {est.reason}"
        assert est.pitch_x == pitch, f"Expected pitch_x={pitch}, got {est.pitch_x} ({die_pattern}, seed={seed})"
        assert est.pitch_y == pitch, f"Expected pitch_y={pitch}, got {est.pitch_y} ({die_pattern}, seed={seed})"
        assert est.peak_strength >= 0.10


# ---------------------------------------------------------------------------
# 2. Non-Periodic Image Fallback & Warning Numbers
# ---------------------------------------------------------------------------
def test_non_periodic_wafer_falls_back_to_background_with_numbers():
    """Verify non-periodic plain noisy wafer reports 'not periodic' and auto mode falls back with metrics."""
    img, _ = generate_synthetic_wafer(
        size=512,
        seed=77,
        grid_pitch=0,
        noise_std=6.0,
        defects=None,
    )
    wafer_mask, _, _, _ = estimate_wafer_mask(img)
    est = estimate_die_pitch(img, wafer_mask)

    assert est.periodic is False
    assert "not periodic" in est.reason
    assert est.pitch_x is None
    assert est.pitch_y is None

    res = run_detection(
        test_image=img,
        reference_image=None,
        config=DetectionConfig(no_reference_mode="auto"),
    )
    assert res.path == "no_reference"
    assert res.label == "Experimental no-reference detection"
    assert len(res.warnings) >= 1
    assert "not periodic" in res.warnings[0]
    assert f"{est.peak_strength_x:.4f}" in res.warnings[0]


# ---------------------------------------------------------------------------
# 3. 4 Injected Defects Found in Both Die Patterns (No Reference)
# ---------------------------------------------------------------------------
def _point_to_segment_distance(
    px: float,
    py: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:
    """Compute Euclidean distance from point (px, py) to line segment (x1, y1)-(x2, y2)."""
    dx, dy = x2 - x1, y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq <= 1e-9:
        return float(np.hypot(px - x1, py - y1))
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / seg_len_sq))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return float(np.hypot(px - proj_x, py - proj_y))


@pytest.mark.parametrize("die_pattern", ["grid", "circuit"])
def test_self_reference_finds_four_defects_in_both_patterns(die_pattern: str):
    """Verify bright, dark, scratch, and near-rim defects are all found with zero false positives.

    Tests both exact integer pitch (shift_tolerance=False -> 4 intact regions) and default
    3x3 shift tolerance (all 4 defects detected and zero spurious detections outside the defects).
    """
    defects = [
        {"type": "bright_blob", "center": (165.0, 175.0), "radius": 8.0, "contrast": 65.0},
        {"type": "dark_blob", "center": (345.0, 185.0), "radius": 9.0, "contrast": -65.0},
        {"type": "scratch", "start": (190.0, 315.0), "end": (260.0, 345.0), "width": 3, "contrast": 70.0},
        {"type": "bright_blob", "center": (445.0, 256.0), "radius": 9.0, "contrast": 65.0},
    ]
    img, gt = generate_synthetic_wafer(
        size=512,
        seed=505,
        grid_pitch=32,
        die_pattern=die_pattern,
        noise_std=5.0,
        defects=defects,
    )

    # A. Without 3x3 shift window: all 4 defects are intact (exact count == 4, centroids within 12 px)
    res_exact = run_full_pipeline(
        test_image=img,
        reference_image=None,
        det_config=DetectionConfig(no_reference_mode="auto", self_ref_shift_tolerance=False),
        filter_config=NoiseFilterConfig(),
    )
    assert res_exact.detection.path == "self_reference"
    assert res_exact.detection.label == "Die-to-die self-reference detection (experimental)"
    assert res_exact.detection.die_pitch == (32, 32)
    assert len(res_exact.accepted_regions) == 4
    exact_centroids = [r.centroid for r in res_exact.accepted_regions]
    for d in gt:
        gx, gy = d["center"]
        min_dist = min(np.hypot(cx - gx, cy - gy) for cx, cy in exact_centroids)
        assert min_dist <= 12.0

    # B. Default config (shift_tolerance=True, merge_gap_px=3): fragment merging reunites the
    #    scratch crossing die streets into ONE single region -> strictly 4 regions matching all 4 GT defects
    res_tol = run_full_pipeline(
        test_image=img,
        reference_image=None,
        det_config=DetectionConfig(),
        filter_config=NoiseFilterConfig(),
    )
    assert res_tol.detection.path == "self_reference"
    assert res_tol.detection.die_pitch == (32, 32)
    assert len(res_tol.accepted_regions) == 4, (
        f"{die_pattern}: expected strictly 4 merged regions under default config, "
        f"got {len(res_tol.accepted_regions)}: {[(r.id, r.area_px, r.centroid) for r in res_tol.accepted_regions]}"
    )

    # Verify the after_fragment_merge stage is present in the funnel
    funnel_stages = [sc["stage"] for sc in res_tol.stage_counts]
    assert "after_fragment_merge" in funnel_stages

    # Every ground-truth defect must match a unique accepted region (1-to-1)
    matched_region_ids = set()
    for d in gt:
        gx, gy = d["center"]
        best_region = min(
            res_tol.accepted_regions,
            key=lambda r: float(np.hypot(r.centroid[0] - gx, r.centroid[1] - gy)),
        )
        dist = float(np.hypot(best_region.centroid[0] - gx, best_region.centroid[1] - gy))
        assert dist <= 12.0, (
            f"{die_pattern}: GT defect {d['type']} at ({gx}, {gy}) not matched within 12 px "
            f"(nearest region #{best_region.id} at {best_region.centroid}, dist={dist:.1f})"
        )
        matched_region_ids.add(best_region.id)

    assert len(matched_region_ids) == 4


def test_fragment_merge_separates_20px_merges_2px_and_preserves_true_area():
    """Verify fragments 2 px apart merge with exact pixel sum, while defects 20 px apart stay separate."""
    from app.cv.segmentation import segment_regions

    mask = np.zeros((160, 220), dtype=np.uint8)
    signed = np.zeros((160, 220), dtype=np.int16)

    # 1. Scratch split into three segments separated by 2-px gaps (simulate die-street crossings)
    # Segment A: y=40..42, x=20..49 -> 3 * 30 = 90 px
    # Gap:       x=50..51 (2 px gap)
    # Segment B: y=40..42, x=52..81 -> 3 * 30 = 90 px
    # Gap:       x=82..83 (2 px gap)
    # Segment C: y=40..42, x=84..113 -> 3 * 30 = 90 px
    mask[40:43, 20:50] = 255
    mask[40:43, 52:82] = 255
    mask[40:43, 84:114] = 255
    signed[40:43, 20:50] = 65
    signed[40:43, 52:82] = 65
    signed[40:43, 84:114] = 65

    # 2. Two distinct defects 20 px apart (must NOT merge under merge_gap_px=3)
    # Defect 1: y=100..111, x=40..51 -> 12 * 12 = 144 px
    # Gap:      x=52..71 (20 px gap)
    # Defect 2: y=100..111, x=72..83 -> 12 * 12 = 144 px
    mask[100:112, 40:52] = 255
    mask[100:112, 72:84] = 255
    signed[100:112, 40:52] = 60
    signed[100:112, 72:84] = -60

    full_wafer = np.full(mask.shape, 255, dtype=np.uint8)

    # Without merging: 3 scratch fragments + 2 separate blobs = 5 components
    unmerged = segment_regions(mask, signed, wafer_mask=full_wafer, merge_gap_px=0)
    assert len(unmerged) == 5
    fragment_area_sum = sum(r.area_px for r in unmerged if r.centroid[1] < 60.0)
    assert fragment_area_sum == 270

    # With default merge_gap_px=3: the 3 scratch fragments (2 px apart) merge into 1,
    # while the 2 defects 20 px apart remain 2 distinct regions -> total 3 regions
    merged = segment_regions(mask, signed, wafer_mask=full_wafer, merge_gap_px=3)
    assert len(merged) == 3

    scratch_regions = [r for r in merged if r.centroid[1] < 60.0]
    separate_regions = [r for r in merged if r.centroid[1] >= 60.0]

    assert len(scratch_regions) == 1
    assert len(separate_regions) == 2

    # Merged scratch area_px MUST equal the true sum of its original fragment pixels (within 1 px; exact here)
    assert abs(scratch_regions[0].area_px - fragment_area_sum) <= 1
    assert scratch_regions[0].area_px == 270
    # Bounding box spans the full merged scratch (x=20..113 -> width=94)
    assert scratch_regions[0].bbox == (20, 40, 94, 3)


# ---------------------------------------------------------------------------
# 4. Clean Patterned Wafer: Zero Final Defects Over 20 Seeds (Both Patterns)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("die_pattern", ["grid", "circuit"])
def test_clean_patterned_wafer_zero_defects_over_20_seeds(die_pattern: str):
    """Verify clean patterned wafers produce strictly 0 final defects over 20 seeds in both patterns."""
    for seed in range(1000, 1020):
        img, _ = generate_synthetic_wafer(
            size=512,
            seed=seed,
            grid_pitch=32,
            die_pattern=die_pattern,
            noise_std=5.0,
            defects=None,
        )
        result = run_full_pipeline(
            test_image=img,
            reference_image=None,
            det_config=DetectionConfig(no_reference_mode="auto"),
        )
        assert result.detection.path == "self_reference"
        assert len(result.accepted_regions) == 0, (
            f"False positive on clean '{die_pattern}' wafer seed={seed}: "
            f"{[(r.area_px, r.mean_contrast, r.centroid) for r in result.accepted_regions]}"
        )


# ---------------------------------------------------------------------------
# 5. Illumination Gradient Alone on Patterned Wafer: Zero Final Defects
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("die_pattern", ["grid", "circuit"])
def test_illumination_gradient_alone_on_patterned_wafer_zero_defects(die_pattern: str):
    """Verify strong illumination gradient alone on a patterned wafer yields 0 final defects."""
    img, _ = generate_synthetic_wafer(
        size=512,
        seed=88,
        grid_pitch=32,
        die_pattern=die_pattern,
        noise_std=5.0,
        has_gradient=True,
        gradient_strength=40.0,
        defects=None,
    )
    result = run_full_pipeline(
        test_image=img,
        reference_image=None,
        det_config=DetectionConfig(no_reference_mode="auto"),
    )
    assert result.detection.path == "self_reference"
    assert len(result.accepted_regions) == 0


# ---------------------------------------------------------------------------
# 6. Coverage & Unverifiable Border Exclusion
# ---------------------------------------------------------------------------
def test_coverage_and_unverifiable_border_exclusion():
    """Verify coverage is reported, covered defect is found, and unverifiable rim band is excluded."""
    # On a 256x256 wafer (effective radius ~110) with pitch=84 (<= 256//3=85, >=3 repeats),
    # pixels near the outer rim have fewer than 3 valid in-wafer neighbours and form an
    # unverifiable border band that must be explicitly excluded and reported via coverage.
    wafer_clean, _ = generate_synthetic_wafer(size=256, seed=321, grid_pitch=84, die_pattern="circuit", noise_std=5.0)
    wafer_mask, _, _, _ = estimate_wafer_mask(wafer_clean)
    self_ref_probe = build_self_reference(wafer_clean, 84, 84, wafer_mask, min_valid_neighbors=3)

    assert 0.5 < self_ref_probe.coverage < 1.0
    assert self_ref_probe.unverifiable_fraction > 0.0
    assert abs((self_ref_probe.coverage + self_ref_probe.unverifiable_fraction) - 1.0) <= 1e-3

    # Locate an actual unverifiable pixel inside wafer_mask (fewer than 3 valid neighbours)
    unverifiable_band = (wafer_mask > 0) & (self_ref_probe.verifiable_mask == 0)
    ys, xs = np.nonzero(unverifiable_band)
    assert len(xs) > 0, "Expected non-empty unverifiable border band inside wafer"

    # Pick the unverifiable pixel furthest from the verifiable interior so a small defect stays inside it
    rim_idx = len(xs) // 2
    rim_x, rim_y = float(xs[rim_idx]), float(ys[rim_idx])

    defects = [
        {"type": "bright_blob", "center": (128.0, 128.0), "radius": 7.0, "contrast": 70.0},  # covered interior
        {"type": "bright_blob", "center": (rim_x, rim_y), "radius": 3.0, "contrast": 80.0},  # unverifiable rim
    ]
    img, _ = generate_synthetic_wafer(
        size=256,
        seed=321,
        grid_pitch=84,
        die_pattern="circuit",
        noise_std=5.0,
        defects=defects,
    )

    result = run_full_pipeline(
        test_image=img,
        reference_image=None,
        det_config=DetectionConfig(no_reference_mode="auto", self_ref_min_neighbors=3),
    )

    assert result.detection.path == "self_reference"
    assert result.detection.coverage is not None and result.detection.coverage < 1.0
    assert result.detection.unverifiable_fraction is not None and result.detection.unverifiable_fraction > 0.0

    # Covered interior defect MUST be found; unverifiable rim defect is excluded and not falsely reported
    assert len(result.accepted_regions) == 1
    cx, cy = result.accepted_regions[0].centroid
    assert np.hypot(cx - 128.0, cy - 128.0) <= 10.0


# ---------------------------------------------------------------------------
# 7. Funnel Honesty Notes & Non-Increasing Self-Reference Funnel
# ---------------------------------------------------------------------------
def test_funnel_notes_on_increase_and_non_increasing_self_reference():
    """Verify stage notes appear when morphology splits regions, and self-reference funnel is non-increasing."""
    defects = [
        {"type": "bright_blob", "center": (170.0, 180.0), "radius": 8.0, "contrast": 65.0},
        {"type": "dark_blob", "center": (340.0, 300.0), "radius": 9.0, "contrast": -65.0},
    ]
    img, _ = generate_synthetic_wafer(size=512, seed=404, grid_pitch=32, noise_std=5.0, defects=defects)

    # A. Self-reference path: funnel stages are non-increasing and have no increase notes
    res_self = run_full_pipeline(
        test_image=img,
        reference_image=None,
        det_config=DetectionConfig(no_reference_mode="auto"),
    )
    assert res_self.detection.path == "self_reference"
    counts_self = [sc["candidate_count"] for sc in res_self.stage_counts]
    for prev_c, curr_c in zip(counts_self, counts_self[1:]):
        assert curr_c <= prev_c, f"Expected non-increasing self-reference funnel, got {res_self.stage_counts}"

    # B. Background path on patterned wafer: morphology breaks connected grid lines into many pieces -> note added
    res_bg = run_full_pipeline(
        test_image=img,
        reference_image=None,
        det_config=DetectionConfig(no_reference_mode="background"),
    )
    morph_entry = next(sc for sc in res_bg.stage_counts if sc["stage"] == "after_morphology")
    raw_entry = next(sc for sc in res_bg.stage_counts if sc["stage"] == "raw_threshold")
    assert morph_entry["candidate_count"] > raw_entry["candidate_count"]
    assert morph_entry.get("note") is not None
    assert "split connected regions" in morph_entry["note"]


# ---------------------------------------------------------------------------
# 8. Determinism
# ---------------------------------------------------------------------------
def test_self_reference_determinism():
    """Verify identical seed and config yield bitwise identical masks and regions."""
    defects = [
        {"type": "bright_blob", "center": (200.0, 220.0), "radius": 8.0, "contrast": 60.0},
    ]
    img, _ = generate_synthetic_wafer(size=512, seed=909, grid_pitch=32, die_pattern="circuit", defects=defects)

    res1 = run_full_pipeline(img, reference_image=None, det_config=DetectionConfig(no_reference_mode="auto"))
    res2 = run_full_pipeline(img, reference_image=None, det_config=DetectionConfig(no_reference_mode="auto"))

    assert res1.detection.die_pitch == res2.detection.die_pitch
    assert res1.detection.coverage == res2.detection.coverage
    assert np.array_equal(res1.detection.difference_image, res2.detection.difference_image)
    assert np.array_equal(res1.filtered_mask, res2.filtered_mask)
    assert [(r.id, r.area_px, r.centroid) for r in res1.accepted_regions] == [
        (r.id, r.area_px, r.centroid) for r in res2.accepted_regions
    ]


# ---------------------------------------------------------------------------
# 9. API Contract Test
# ---------------------------------------------------------------------------
def test_api_no_reference_returns_self_reference_fields():
    """Verify POST /inspection/run without reference returns path, label, pitch, strength, and coverage."""
    defects = [
        {"type": "bright_blob", "center": (220.0, 230.0), "radius": 8.0, "contrast": 65.0},
    ]
    img, _ = generate_synthetic_wafer(size=512, seed=711, grid_pitch=32, die_pattern="circuit", defects=defects)
    buf = BytesIO()
    Image.fromarray(img).save(buf, format="PNG")

    up = client.post(
        "/inspection/upload",
        files={"test_image": ("patterned_wafer.png", buf.getvalue(), "image/png")},
    )
    assert up.status_code == 200
    upload_id = up.json()["upload_id"]

    run_res = client.post("/inspection/run", json={"upload_id": upload_id})
    assert run_res.status_code == 200
    det = run_res.json()["detection"]

    assert det["path"] == "self_reference"
    assert det["label"] == "Die-to-die self-reference detection (experimental)"
    assert det["die_pitch"] == [32, 32]
    assert det["peak_strength"] is not None and det["peak_strength"] >= 0.10
    assert det["coverage"] is not None and 0.5 < det["coverage"] <= 1.0
    assert det["unverifiable_fraction"] is not None and det["unverifiable_fraction"] >= 0.0
    assert len(run_res.json()["regions"]) == 1
