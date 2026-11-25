"""Tests for Phase 9 noise filtering evaluation, matching logic, and sweep endpoints."""

from io import BytesIO
from pathlib import Path
from typing import Tuple
from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pytest
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from app.cv.noise_experiment import (
    compute_four_cases_from_components,
    match_candidates_to_ground_truth,
    run_noise_sweep,
    run_single_noise_evaluation,
)
from app.cv.noise_filter import NoiseFilterConfig
from app.utils.synthetic import generate_layout_wafer


# ---------------------------------------------------------------------------
# 1. Pure Matching Logic Tests with Hand-Built Ground Truth
# ---------------------------------------------------------------------------


def test_match_candidates_to_ground_truth_hand_built() -> None:
    """Verify greedy 1-to-1 nearest neighbor centroid matching with known TP, FP, FN."""
    # 3 Ground-truth defects
    gt = [
        {"center": [100.0, 100.0]},
        {"center": [200.0, 200.0]},
        {"center": [300.0, 300.0]},
    ]

    # 4 Candidates: 2 close to GT within 15px, 2 far away false alarms
    candidates = [
        (103.0, 102.0),  # Matches GT 0 (dist ~3.6 px)
        (205.0, 196.0),  # Matches GT 1 (dist ~6.4 px)
        (400.0, 400.0),  # FP false alarm
        (500.0, 500.0),  # FP false alarm
    ]

    metrics = match_candidates_to_ground_truth(candidates, gt, tolerance_px=15.0)
    assert metrics.tp == 2, f"Expected 2 true positives, got {metrics.tp}"
    assert metrics.fp == 2, f"Expected 2 false positives, got {metrics.fp}"
    assert metrics.fn == 1, f"Expected 1 false negative, got {metrics.fn}"
    assert metrics.precision == pytest.approx(0.5, abs=1e-3)
    assert metrics.recall == pytest.approx(2.0 / 3.0, abs=1e-3)
    assert metrics.ground_truth_count == 3
    assert metrics.candidate_count == 4


def test_match_candidates_empty_cases() -> None:
    """Verify edge cases: zero candidates and zero ground-truth defects."""
    gt = [{"center": [100.0, 100.0]}]

    # Empty candidates
    m1 = match_candidates_to_ground_truth([], gt)
    assert m1.tp == 0
    assert m1.fp == 0
    assert m1.fn == 1
    assert m1.precision == 0.0
    assert m1.recall == 0.0

    # Empty ground truth (clean wafer with candidates)
    m2 = match_candidates_to_ground_truth([(50.0, 50.0)], [])
    assert m2.tp == 0
    assert m2.fp == 1
    assert m2.fn == 0
    assert m2.precision == 0.0
    assert m2.recall == 1.0


# ---------------------------------------------------------------------------
# 2. Pipeline Stage Monotonicity & Sensitivity Tests
# ---------------------------------------------------------------------------


def test_stage_filtering_candidate_monotonicity() -> None:
    """Verify Stage 2 <= Stage 1 and Stage 4 <= Stage 1 candidate counts on real CV output."""
    test_img, gt, _ = generate_layout_wafer(
        defect_layout="edge_ring",
        count=8,
        size=512,
        seed=123,
        grid_pitch=32,
        die_pattern="grid",
        noise_std=8.0,
    )

    eval_res = run_single_noise_evaluation(
        test_image=test_img,
        reference_image=None,
        ground_truth=gt,
    )

    cases = eval_res["cases"]
    c1 = cases[0]["candidate_count"]
    c2 = cases[1]["candidate_count"]
    c3 = cases[2]["candidate_count"]
    c4 = cases[3]["candidate_count"]

    assert c2 <= c1, f"Min-area candidate count ({c2}) cannot exceed raw count ({c1})"
    assert c4 <= c1, f"Final filtered count ({c4}) cannot exceed raw count ({c1})"
    assert eval_res["has_ground_truth"] is True
    assert eval_res["ground_truth_count"] == len(gt)


def test_raw_candidates_grow_with_noise() -> None:
    """Verify raw threshold candidates grow significantly as Gaussian noise sigma increases."""
    img_low, _, _ = generate_layout_wafer(
        defect_layout="center_cluster", count=8, size=512, seed=77, noise_std=2.0
    )
    img_high, _, _ = generate_layout_wafer(
        defect_layout="center_cluster", count=8, size=512, seed=77, noise_std=15.0
    )

    res_low = run_single_noise_evaluation(test_image=img_low)
    res_high = run_single_noise_evaluation(test_image=img_high)

    raw_low = res_low["cases"][0]["candidate_count"]
    raw_high = res_high["cases"][0]["candidate_count"]

    assert raw_high > raw_low, f"Raw candidates at sigma=15 ({raw_high}) must exceed sigma=2 ({raw_low})"


def test_min_area_filter_sensitivity() -> None:
    """Verify increasing min_area_px monotonically decreases or keeps equal Stage 2 count."""
    raw_mask = np.zeros((100, 100), dtype=np.uint8)
    # 1 component of size 4 px
    raw_mask[10:12, 10:12] = 255
    # 1 component of size 16 px
    raw_mask[30:34, 30:34] = 255
    # 1 component of size 64 px
    raw_mask[60:68, 60:68] = 255

    morph_mask = raw_mask.copy()

    cases_small = compute_four_cases_from_components(
        raw_mask=raw_mask,
        morph_mask=morph_mask,
        accepted_regions=[],
        min_area_px=5,
    )
    cases_large = compute_four_cases_from_components(
        raw_mask=raw_mask,
        morph_mask=morph_mask,
        accepted_regions=[],
        min_area_px=20,
    )

    assert cases_small[1]["candidate_count"] == 2  # 16px and 64px pass
    assert cases_large[1]["candidate_count"] == 1  # only 64px passes
    assert cases_large[1]["candidate_count"] < cases_small[1]["candidate_count"]


def test_sweep_determinism_for_fixed_seed() -> None:
    """Verify run_noise_sweep produces bitwise identical numeric metrics for identical seeds."""
    sweep1 = run_noise_sweep(
        layout="edge_ring",
        path="self_reference",
        noise_levels=[5.0, 10.0],
        seeds=[101, 102],
        size=256,
        defect_count=5,
    )
    sweep2 = run_noise_sweep(
        layout="edge_ring",
        path="self_reference",
        noise_levels=[5.0, 10.0],
        seeds=[101, 102],
        size=256,
        defect_count=5,
    )

    for pt1, pt2 in zip(sweep1["series"], sweep2["series"]):
        assert pt1["noise_level"] == pt2["noise_level"]
        for c1, c2 in zip(pt1["cases"], pt2["cases"]):
            assert c1["mean_count"] == c2["mean_count"]
            assert c1["mean_precision"] == c2["mean_precision"]
            assert c1["mean_recall"] == c2["mean_recall"]


def test_sweep_precision_improvement_over_raw() -> None:
    """Verify Stage 4 final precision is >= Stage 1 raw precision on sweep averages."""
    sweep = run_noise_sweep(
        layout="edge_ring",
        path="self_reference",
        noise_levels=[10.0],
        seeds=[42, 43, 44],
        size=512,
        defect_count=8,
    )
    pt = sweep["series"][0]
    raw_prec = pt["cases"][0]["mean_precision"]
    final_prec = pt["cases"][3]["mean_precision"]

    assert final_prec >= raw_prec, f"Final precision ({final_prec}) must be >= raw ({raw_prec})"


# ---------------------------------------------------------------------------
# 3. HTTP API Endpoint Tests
# ---------------------------------------------------------------------------


def test_get_inspection_noise_experiment_404(
    isolated_api_client: Tuple[TestClient, Engine, sessionmaker, Path, str],
) -> None:
    """GET /inspection/{id}/noise-experiment returns 404 for unknown ID."""
    client, _, _, _, _ = isolated_api_client
    resp = client.get("/inspection/999999/noise-experiment")
    assert resp.status_code == 404
    data = resp.json()
    assert data["detail"]["code"] == "INSPECTION_NOT_FOUND"


def test_get_inspection_noise_experiment_sample_vs_non_sample(
    isolated_api_client: Tuple[TestClient, Engine, sessionmaker, Path, str],
) -> None:
    """Verify synthetic sample yields ground truth with metrics while manual upload yields null GT."""
    client, _, _, _, _ = isolated_api_client

    # 1. Synthetic sample upload
    sample_upload = client.post("/samples/edge_ring")
    assert sample_upload.status_code == 200
    sample_insp_id = sample_upload.json()["inspection_id"]

    # Run inspection
    run_resp = client.post("/inspection/run", json={"inspection_id": sample_insp_id})
    assert run_resp.status_code == 200

    # Fetch noise experiment for synthetic sample
    exp_sample = client.get(f"/inspection/{sample_insp_id}/noise-experiment")
    assert exp_sample.status_code == 200
    sample_data = exp_sample.json()

    assert sample_data["has_ground_truth"] is True
    assert sample_data["ground_truth_count"] is not None
    assert sample_data["ground_truth_count"] > 0
    assert sample_data["disclaimer"] == "Synthetic demonstration data"
    # Stage 4 precision & recall must be numeric
    assert sample_data["cases"][3]["precision"] is not None
    assert sample_data["cases"][3]["recall"] is not None

    # 2. Manual non-sample upload (plain image, zero synthetic metadata)
    raw_img = Image.new("L", (256, 256), color=128)
    buf = BytesIO()
    raw_img.save(buf, format="PNG")
    buf.seek(0)

    manual_upload = client.post(
        "/inspection/upload",
        files={"test_image": ("manual_wafer.png", buf.getvalue(), "image/png")},
    )
    assert manual_upload.status_code == 200
    manual_insp_id = manual_upload.json()["inspection_id"]

    run_manual = client.post("/inspection/run", json={"inspection_id": manual_insp_id})
    assert run_manual.status_code == 200

    # Fetch noise experiment for manual non-sample
    exp_manual = client.get(f"/inspection/{manual_insp_id}/noise-experiment")
    assert exp_manual.status_code == 200
    manual_data = exp_manual.json()

    assert manual_data["has_ground_truth"] is False
    assert manual_data["ground_truth_count"] is None
    assert manual_data["disclaimer"] is None
    # Metrics must be null
    assert manual_data["cases"][3]["precision"] is None
    assert manual_data["cases"][3]["recall"] is None
    assert manual_data["cases"][3]["tp"] is None


def test_post_noise_sweep_endpoint_and_caps(
    isolated_api_client: Tuple[TestClient, Engine, sessionmaker, Path, str],
) -> None:
    """Verify POST /experiments/noise-sweep success and 422 rejections on exceeding caps."""
    client, _, _, _, _ = isolated_api_client

    # Successful call with small fast sweep
    valid_payload = {
        "layout": "center_cluster",
        "path": "self_reference",
        "noise_levels": [5.0, 10.0],
        "seeds": [42, 43],
        "size": 256,
        "defect_count": 6,
    }
    resp = client.post("/experiments/noise-sweep", json=valid_payload)
    assert resp.status_code == 200
    data = resp.json()

    assert len(data["series"]) == 2
    assert data["true_defect_count"] == 6
    assert "Noise filtering removed an average" in data["summary"]
    assert data["disclaimer"] == "Synthetic demonstration data"

    # Exceeding noise levels cap (> 8)
    resp_levels = client.post(
        "/experiments/noise-sweep",
        json={"noise_levels": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]},
    )
    assert resp_levels.status_code == 422

    # Exceeding seeds cap (> 10)
    resp_seeds = client.post(
        "/experiments/noise-sweep",
        json={"seeds": list(range(11))},
    )
    assert resp_seeds.status_code == 422

    # Exceeding image size cap (> 1024)
    resp_size = client.post(
        "/experiments/noise-sweep",
        json={"size": 2048},
    )
    assert resp_size.status_code == 422
