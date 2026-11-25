"""Targeted tests to strengthen coverage across thin areas in cv, services, api, and utils."""

import uuid
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
from app.cv.detection import apply_threshold
from app.cv.noise_filter import (
    NoiseFilterConfig,
    compute_region_aspect_ratio,
    compute_region_solidity,
    filter_regions,
)
from app.cv.pipeline import DetectionConfig, run_detection
from app.cv.preprocessing import (
    denoise,
    enhance_contrast,
    tile_image,
)
from app.cv.segmentation import Region
from app.db import crud
from fastapi.testclient import TestClient
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker


def _make_region(
    region_id: int,
    area_px: int,
    bbox: Tuple[int, int, int, int],
    mask: np.ndarray,
    mean_contrast: float = 50.0,
    touches_edge: bool = False,
) -> Region:
    return Region(
        id=region_id,
        bbox=bbox,
        area_px=area_px,
        centroid=(float(bbox[0] + bbox[2] / 2), float(bbox[1] + bbox[3] / 2)),
        mask=mask,
        contour=[(bbox[0], bbox[1])],
        polarity="bright",
        mean_contrast=mean_contrast,
        peak_contrast=mean_contrast + 10.0,
        touches_edge=touches_edge,
    )


# ---------------------------------------------------------------------------
# 1. cv/noise_filter.py Coverage: Solidity & Aspect Ratio Filters
# ---------------------------------------------------------------------------


def test_compute_region_solidity_edge_cases_and_filter() -> None:
    """Verify compute_region_solidity on empty, minimal, and non-convex shapes, and solidity filtering."""
    # Empty mask
    empty_mask = np.zeros((20, 20), dtype=np.uint8)
    r_empty = _make_region(1, 0, (0, 0, 1, 1), empty_mask)
    assert compute_region_solidity(r_empty) == 0.0

    # 2-pixel mask (less than 3 points for convex hull)
    min_mask = np.zeros((20, 20), dtype=np.uint8)
    min_mask[5, 5] = 1
    min_mask[5, 6] = 1
    r_min = _make_region(2, 2, (5, 5, 2, 1), min_mask)
    assert compute_region_solidity(r_min) == 1.0

    # Solid 10x10 square (solidity = 1.0)
    square_mask = np.zeros((30, 30), dtype=np.uint8)
    square_mask[10:20, 10:20] = 1
    r_square = _make_region(3, 100, (10, 10, 10, 10), square_mask, mean_contrast=60.0)
    assert compute_region_solidity(r_square) == pytest.approx(1.0, abs=1e-2)

    # L-shaped non-convex region (area=51, convex hull ~ 100, solidity ~ 0.5)
    l_mask = np.zeros((30, 30), dtype=np.uint8)
    l_mask[5:15, 5:8] = 1  # vertical stem 10x3 = 30
    l_mask[12:15, 8:15] = 1  # horizontal base 3x7 = 21
    l_area = int(np.count_nonzero(l_mask))
    r_l = _make_region(4, l_area, (5, 5, 10, 10), l_mask, mean_contrast=60.0)
    solidity_l = compute_region_solidity(r_l)
    assert solidity_l < 0.75, f"Expected non-convex solidity < 0.75, got {solidity_l}"

    # Filter with min_solidity=0.85: square must survive, L-shape must be rejected
    cfg = NoiseFilterConfig(min_area_px=10, min_solidity=0.85)
    res = filter_regions([r_square, r_l], noise_sigma=5.0, config=cfg)

    accepted_ids = [a.id for a in res.accepted_regions]
    rejected_ids = [r.id for r in res.rejected_regions]
    assert 3 in accepted_ids, "Solid square must be accepted"
    assert 4 in rejected_ids, "L-shape must be rejected by solidity filter"

    rej_l = next(r for r in res.rejected_regions if r.id == 4)
    assert rej_l.rule == "solidity_filter"
    assert "solidity=" in rej_l.reason


def test_aspect_ratio_filter_rejection() -> None:
    """Verify max_aspect_ratio filter rejects regions exceeding the threshold."""
    # 2x30 scratch-like region (aspect ratio > 10)
    long_mask = np.zeros((40, 40), dtype=np.uint8)
    long_mask[10:12, 5:35] = 1
    r_long = _make_region(1, 60, (5, 10, 30, 2), long_mask)
    aspect = compute_region_aspect_ratio(r_long)
    assert aspect > 5.0

    cfg = NoiseFilterConfig(min_area_px=10, max_aspect_ratio=3.0)
    res = filter_regions([r_long], noise_sigma=5.0, config=cfg)

    assert len(res.accepted_regions) == 0
    assert len(res.rejected_regions) == 1
    assert res.rejected_regions[0].rule == "aspect_ratio_filter"


# ---------------------------------------------------------------------------
# 2. cv/preprocessing.py Coverage: Tiling & Contrast Enhancement
# ---------------------------------------------------------------------------


def test_tile_image_validation_and_small_images() -> None:
    """Verify tile_image validation rules and single-tile return for small images."""
    arr = np.zeros((100, 100), dtype=np.uint8)

    with pytest.raises(TypeError, match="Expected numpy.ndarray"):
        tile_image("not_an_array", 64, 16)  # type: ignore

    with pytest.raises(ValueError, match="tile_size must be positive"):
        tile_image(arr, 0, 0)

    with pytest.raises(ValueError, match="overlap must be non-negative"):
        tile_image(arr, 64, 64)

    # Image smaller than tile size: single tile
    tiles = tile_image(arr, 128, 16)
    assert len(tiles) == 1
    assert tiles[0][1] == 0 and tiles[0][2] == 0
    assert tiles[0][0].shape == (100, 100)


def test_enhance_contrast_clahe_modes() -> None:
    """Verify enhance_contrast forced CLAHE and clip limit parameterization."""
    low_contrast = np.full((64, 64), 120, dtype=np.uint8)
    low_contrast[20:40, 20:40] = 125

    # Forced CLAHE with custom clip limit
    enhanced, info = enhance_contrast(low_contrast, method="clahe", clip_limit=4.0)
    assert info["applied"] is True
    assert "Applied CLAHE" in info["reason"]
    assert enhanced.shape == (64, 64)

    # Method "none"
    unchanged, info_none = enhance_contrast(low_contrast, method="none")
    assert info_none["applied"] is False
    assert "disabled" in info_none["reason"].lower()
    assert np.array_equal(unchanged, low_contrast)


def test_denoise_median_and_invalid_kernel() -> None:
    """Verify median filter denoising and rejection of even/non-positive kernels."""
    img = np.zeros((32, 32), dtype=np.uint8)
    img[10, 10] = 255  # salt noise

    denoised = denoise(img, method="median", kernel_size=3)
    assert denoised[10, 10] == 0  # isolated single pixel removed

    with pytest.raises(ValueError, match="kernel_size must be a positive odd integer"):
        denoise(img, method="median", kernel_size=4)

    with pytest.raises(ValueError, match="kernel_size must be a positive odd integer"):
        denoise(img, method="median", kernel_size=0)


# ---------------------------------------------------------------------------
# 3. cv/detection.py Coverage: Custom Thresholding & Edge Guards
# ---------------------------------------------------------------------------


def test_apply_threshold_invalid_method() -> None:
    """Verify apply_threshold raises ValueError on invalid method."""
    diff = np.zeros((32, 32), dtype=np.uint8)
    mask = np.ones((32, 32), dtype=np.uint8) * 255
    with pytest.raises(ValueError, match="Unknown threshold method"):
        apply_threshold(diff, wafer_mask=mask, method="unsupported_method")


def test_run_detection_fixed_threshold_mode() -> None:
    """Verify run_detection supports fixed thresholding with explicit cutoff."""
    test_img = np.full((128, 128), 100, dtype=np.uint8)
    ref_img = np.full((128, 128), 100, dtype=np.uint8)
    # Inject 50px bright square of contrast 60
    test_img[40:60, 40:60] = 160

    cfg = DetectionConfig(
        path="golden",
        threshold_method="fixed",
        fixed_threshold=40.0,
    )
    res = run_detection(test_img, reference_image=ref_img, config=cfg)
    assert res.threshold_method == "fixed"
    assert res.threshold_value == 40.0
    assert np.count_nonzero(res.morphology_mask[40:60, 40:60]) > 0


# ---------------------------------------------------------------------------
# 4. services/inspection_service.py Coverage: Statuses & Error Handling
# ---------------------------------------------------------------------------


def test_inspection_report_by_id_pending_and_failed_states(
    isolated_api_client: Tuple[TestClient, Engine, sessionmaker, Path, str],
) -> None:
    """Verify get_inspection_by_id returns 409 for pending or failed database records."""
    client, eng, SessionFactory, _, _ = isolated_api_client
    db = SessionFactory()
    try:
        # Create an inspection row with status 'pending'
        pending_insp = crud.create_inspection(
            db,
            upload_id=str(uuid.uuid4()),
            original_filename="pending_wafer.png",
            image_width=512,
            image_height=512,
            image_mode="L",
            image_format="PNG",
            file_size=1024,
            has_reference=False,
            status="pending",
        )
        pid = pending_insp.id

        resp_pending = client.get(f"/inspection/{pid}")
        assert resp_pending.status_code == 409
        assert resp_pending.json()["detail"]["code"] == "RUN_NOT_FINISHED"

        # Update status to 'failed'
        crud.set_status(db, pid, "failed", error_message="Simulated CV crash")
        resp_failed = client.get(f"/inspection/{pid}")
        assert resp_failed.status_code == 409
        assert resp_failed.json()["detail"]["code"] == "RUN_FAILED"
        assert "Simulated CV crash" in resp_failed.json()["detail"]["message"]

        # Call defects on unknown id
        resp_def_404 = client.get("/inspection/999999/defects")
        assert resp_def_404.status_code == 404
        assert resp_def_404.json()["detail"]["code"] == "INSPECTION_NOT_FOUND"

        # Call images on unknown id
        resp_img_404 = client.get("/inspection/999999/images")
        assert resp_img_404.status_code == 404
        assert resp_img_404.json()["detail"]["code"] == "INSPECTION_NOT_FOUND"

    finally:
        db.close()
