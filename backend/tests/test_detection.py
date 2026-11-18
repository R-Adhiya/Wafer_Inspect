"""Deterministic synthetic tests for Phase 4 CV defect detection algorithms and API.

Validates:
1. Wafer mask segmentation and full-frame boundary detection.
2. Golden-image differential detection and phase-correlation subpixel alignment.
3. Experimental no-reference background subtraction and residual detection.
4. Robust contrast guard and threshold algorithms (fixed, otsu, adaptive).
5. Mathematical morphology filtering (opening, closing, erosion, dilation) and validation.
6. Pipeline determinism and immutability.
7. API endpoints, artifact serving (difference, mask), and REFERENCE_INCOMPATIBLE 422 validation.
"""

from io import BytesIO
from typing import Dict, List
import cv2
from fastapi.testclient import TestClient
import numpy as np
import pytest
from PIL import Image

from app.core.config import settings
from app.cv.detection import (
    align_reference,
    apply_morphology,
    apply_threshold,
    count_connected_candidates,
    estimate_wafer_mask,
    morph_close,
    morph_dilate,
    morph_erode,
    morph_open,
)
from app.cv.pipeline import DetectionConfig, PreprocessingConfig, run_detection
from app.main import app
from app.utils.synthetic import create_standard_synthetic_pair, generate_synthetic_wafer

client = TestClient(app)


# ---------------------------------------------------------------------------
# 1. Wafer Mask & Geometry Extraction Tests
# ---------------------------------------------------------------------------
def test_wafer_mask_radius_estimation_within_2_percent():
    """Verify wafer disk radius estimation is within 2% of ground truth."""
    size = 512
    wafer_ratio = 0.44
    true_radius = size * wafer_ratio  # 225.28

    image, _ = generate_synthetic_wafer(
        size=size,
        seed=100,
        wafer_radius_ratio=wafer_ratio,
        noise_std=8.0,
    )

    mask, center, radius, is_full_frame = estimate_wafer_mask(image, edge_margin_ratio=0.02)

    assert not is_full_frame
    radius_error = abs(radius - true_radius) / true_radius
    assert radius_error <= 0.02, f"Radius error {radius_error:.3f} exceeded 2% limit"
    assert abs(center[0] - size / 2.0) <= 4.0
    assert abs(center[1] - size / 2.0) <= 4.0
    assert mask.shape == (size, size)
    assert mask.dtype == np.uint8


def test_wafer_mask_full_frame_detection():
    """Verify images with > 95% foreground are reported as full-frame."""
    size = 256
    full_frame_img = np.full((size, size), 160, dtype=np.uint8)

    mask, center, radius, is_full_frame = estimate_wafer_mask(full_frame_img)

    assert is_full_frame is True
    assert np.all(mask == 255)
    assert center == (size / 2.0, size / 2.0)
    assert radius == size / 2.0


# ---------------------------------------------------------------------------
# 2. Golden-Image Detection Tests
# ---------------------------------------------------------------------------
def test_golden_detection_finds_injected_defects():
    """Verify golden reference inspection reliably identifies injected defect centroids."""
    size = 512
    injected_defects = [
        {"type": "bright_blob", "center": (160.0, 180.0), "radius": 8.0, "contrast": 55.0},
        {"type": "dark_blob", "center": (340.0, 200.0), "radius": 9.0, "contrast": -55.0},
        {"type": "scratch", "start": (200.0, 300.0), "end": (260.0, 360.0), "width": 3, "contrast": 60.0},
    ]

    ref_img, test_img, ground_truth = create_standard_synthetic_pair(
        size=size,
        seed=42,
        shift=(0.0, 0.0),
        defects=injected_defects,
    )

    result = run_detection(
        test_image=test_img,
        reference_image=ref_img,
        config=DetectionConfig(threshold_method="otsu", min_robust_contrast=8.0),
    )

    assert result.path == "golden"
    assert result.label == "Golden Reference Inspection"
    assert result.morphology_mask.dtype == np.uint8

    # Extract connected components from morphology mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(result.morphology_mask)
    detected_centroids = centroids[1:]  # skip background

    assert len(detected_centroids) >= len(injected_defects)

    # Verify each injected ground truth defect is matched to a candidate centroid within tolerance
    for gt in ground_truth:
        gt_cx, gt_cy = gt["center"]
        dists = [np.hypot(c[0] - gt_cx, c[1] - gt_cy) for c in detected_centroids]
        min_dist = min(dists)
        assert min_dist <= 12.0, f"Defect {gt['type']} at ({gt_cx}, {gt_cy}) had nearest match at distance {min_dist:.1f}"


def test_golden_clean_wafer_produces_zero_candidates():
    """Verify comparison of two clean noisy wafers produces zero candidate defect regions."""
    clean_ref, _ = generate_synthetic_wafer(size=512, seed=123, noise_std=10.0, defects=None)
    clean_test, _ = generate_synthetic_wafer(size=512, seed=124, noise_std=10.0, defects=None)

    result = run_detection(
        test_image=clean_test,
        reference_image=clean_ref,
        config=DetectionConfig(threshold_method="otsu", min_robust_contrast=8.0),
    )

    candidates = count_connected_candidates(result.morphology_mask)
    assert candidates == 0
    assert result.stage_counts[-1]["candidate_count"] == 0


# ---------------------------------------------------------------------------
# 3. Alignment Tests
# ---------------------------------------------------------------------------
def test_alignment_compensates_subpixel_shift():
    """Verify phase correlation accurately estimates (3, 2) pixel translation and reduces false alarms."""
    size = 512
    true_shift = (3.0, 2.0)
    injected_defects = [
        {"type": "bright_blob", "center": (256.0, 256.0), "radius": 8.0, "contrast": 60.0},
    ]

    ref_img, test_img, _ = create_standard_synthetic_pair(
        size=size,
        seed=888,
        shift=true_shift,
        defects=injected_defects,
    )

    # 1. Detection WITH alignment
    result_with_align = run_detection(
        test_image=test_img,
        reference_image=ref_img,
        config=DetectionConfig(
            max_alignment_shift=10.0,
            min_alignment_response=0.02,
            threshold_method="otsu",
        ),
    )

    # 2. Detection WITHOUT alignment (max_alignment_shift = 0.0 disables alignment)
    result_without_align = run_detection(
        test_image=test_img,
        reference_image=ref_img,
        config=DetectionConfig(
            max_alignment_shift=0.0,
            threshold_method="otsu",
        ),
    )

    align_info = result_with_align.alignment_info
    assert align_info["applied"] is True

    # Note: cv2.phaseCorrelate measures shift from test to ref or ref to test; magnitude matches true translation
    assert abs(align_info["shift_magnitude"] - np.hypot(3.0, 2.0)) <= 0.8

    candidates_with = result_with_align.stage_counts[-1]["candidate_count"]
    candidates_without = result_without_align.stage_counts[-1]["candidate_count"]

    # Without alignment, the unshifted die grid triggers dozens of misregistration false alarms
    assert candidates_with < candidates_without
    assert candidates_with == 1  # Only the real injected blob survives


# ---------------------------------------------------------------------------
# 4. No-Reference Detection Tests
# ---------------------------------------------------------------------------
def test_no_reference_finds_bright_and_dark_defects():
    """Verify experimental no-reference residual inspection detects high-contrast local anomalies."""
    size = 512
    injected_defects = [
        {"type": "bright_blob", "center": (180.0, 220.0), "radius": 9.0, "contrast": 65.0},
        {"type": "dark_blob", "center": (330.0, 260.0), "radius": 9.0, "contrast": -65.0},
    ]

    test_img, ground_truth = generate_synthetic_wafer(
        size=size,
        seed=55,
        noise_std=8.0,
        has_gradient=False,
        defects=injected_defects,
    )

    result = run_detection(
        test_image=test_img,
        reference_image=None,
        config=DetectionConfig(threshold_method="fixed", fixed_threshold=28.0),
    )

    assert result.path == "no_reference"
    assert result.label == "Experimental no-reference detection"

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(result.morphology_mask)
    detected_centroids = centroids[1:]

    assert len(detected_centroids) >= 2
    for gt in ground_truth:
        gt_x, gt_y = gt["center"]
        min_d = min(np.hypot(c[0] - gt_x, c[1] - gt_y) for c in detected_centroids)
        assert min_d <= 12.0


def test_no_reference_illumination_gradient_produces_zero_candidates():
    """Verify continuous low-frequency illumination gradient alone (unpatterned wafer) produces zero false candidates."""
    size = 512
    gradient_img, _ = generate_synthetic_wafer(
        size=size,
        seed=77,
        grid_pitch=0,
        noise_std=8.0,
        has_gradient=True,
        gradient_strength=45.0,
        defects=None,
    )

    result = run_detection(
        test_image=gradient_img,
        reference_image=None,
        config=DetectionConfig(threshold_method="otsu", min_robust_contrast=15.0),
    )

    candidates = count_connected_candidates(result.morphology_mask)
    assert candidates == 0


def test_no_reference_wafer_edge_not_flagged():
    """Verify wafer boundary margin prevents bevel boundary from triggering false edge defects on a clean noisy wafer."""
    size = 512
    clean_img, _ = generate_synthetic_wafer(
        size=size,
        seed=99,
        grid_pitch=0,
        noise_std=8.0,
        defects=None,
    )

    result = run_detection(
        test_image=clean_img,
        reference_image=None,
        config=DetectionConfig(
            edge_margin_ratio=0.03,
            threshold_method="otsu",
            min_robust_contrast=8.0,
        ),
    )

    # Check edge touching and clean candidate count
    morph_mask = result.morphology_mask
    h, w = morph_mask.shape
    border_pixels = np.sum(morph_mask[0, :]) + np.sum(morph_mask[h - 1, :]) + np.sum(morph_mask[:, 0]) + np.sum(morph_mask[:, w - 1])
    assert border_pixels == 0
    assert count_connected_candidates(morph_mask) == 0


# ---------------------------------------------------------------------------
# 5. Threshold Method & Robust Contrast Guard Tests
# ---------------------------------------------------------------------------
def test_threshold_methods_return_uint8_binary_masks():
    """Verify fixed, otsu, and adaptive thresholding return uint8 0/255 binary masks."""
    diff = np.zeros((100, 100), dtype=np.uint8)
    diff[30:70, 30:70] = 50  # defect
    diff += np.random.randint(0, 5, (100, 100), dtype=np.uint8)
    wafer_mask = np.full((100, 100), 255, dtype=np.uint8)

    for method in ("fixed", "otsu", "adaptive"):
        mask, val, reason = apply_threshold(
            diff,
            wafer_mask,
            method=method,
            fixed_threshold=20.0,
            adaptive_block_size=15,
            adaptive_c=3.0,
        )
        assert mask.shape == (100, 100)
        assert mask.dtype == np.uint8
        assert set(np.unique(mask)).issubset({0, 255})
        assert val > 0.0
        assert len(reason) > 0


def test_otsu_low_contrast_guard_returns_empty_mask():
    """Verify Otsu guard suppresses segmentation on difference maps with noise-only variance."""
    rng = np.random.default_rng(42)
    noise_diff = rng.integers(0, 4, size=(128, 128), dtype=np.uint8)
    wafer_mask = np.full((128, 128), 255, dtype=np.uint8)

    mask, thresh_val, reason = apply_threshold(
        noise_diff,
        wafer_mask,
        method="otsu",
        min_robust_contrast=8.0,
    )

    assert np.all(mask == 0)
    assert thresh_val == 0.0
    assert "no significant difference found" in reason


# ---------------------------------------------------------------------------
# 6. Mathematical Morphology Tests
# ---------------------------------------------------------------------------
def test_morphology_open_removes_single_pixel_speck_and_preserves_blob():
    """Verify opening eliminates 1-pixel noise spikes while preserving solid 15x15 defect cores."""
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10, 10] = 255  # 1px speck
    mask[25:40, 25:40] = 255  # 15x15 blob

    cleaned = morph_open(mask, kernel_size=3)

    assert cleaned[10, 10] == 0, "1px speck was not eliminated by opening"
    assert np.sum(cleaned[28:37, 28:37] == 255) > 50, "15x15 blob core was not preserved"


def test_morphology_close_fills_pinhole():
    """Verify closing seals interior 3-pixel void inside solid defect component."""
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[20:45, 20:45] = 255
    mask[30:33, 30:33] = 0  # 3x3 pinhole inside component

    assert mask[31, 31] == 0
    closed = morph_close(mask, kernel_size=5)
    assert closed[31, 31] == 255, "3x3 hole was not closed"


def test_morphology_erosion_and_dilation_area_deltas():
    """Verify erosion decreases area and dilation expands area monotonically."""
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[20:40, 20:40] = 255
    orig_area = np.count_nonzero(mask)

    eroded = morph_erode(mask, kernel_size=3)
    dilated = morph_dilate(mask, kernel_size=3)

    eroded_area = np.count_nonzero(eroded)
    dilated_area = np.count_nonzero(dilated)

    assert eroded_area < orig_area
    assert dilated_area > orig_area


def test_morphology_even_kernel_raises_value_error():
    """Verify even or non-positive kernel dimensions raise ValueError."""
    mask = np.zeros((32, 32), dtype=np.uint8)
    with pytest.raises(ValueError, match="odd positive"):
        morph_open(mask, kernel_size=4)

    with pytest.raises(ValueError, match="odd positive"):
        morph_close(mask, kernel_size=0)

    with pytest.raises(ValueError, match="odd positive"):
        apply_morphology(mask, ops=[("erode", -2)])


# ---------------------------------------------------------------------------
# 7. Pipeline Determinism Tests
# ---------------------------------------------------------------------------
def test_detection_determinism():
    """Verify executing detection multiple times on identical inputs yields bitwise identical masks."""
    ref_img, test_img, _ = create_standard_synthetic_pair(
        size=256,
        seed=42,
        defects=[{"type": "bright_blob", "center": (128.0, 128.0), "radius": 6.0, "contrast": 50.0}],
    )

    config = DetectionConfig(threshold_method="otsu")
    res1 = run_detection(test_img, ref_img, config)
    res2 = run_detection(test_img, ref_img, config)

    assert np.array_equal(res1.difference_image, res2.difference_image)
    assert np.array_equal(res1.morphology_mask, res2.morphology_mask)
    assert res1.threshold_value == res2.threshold_value
    assert res1.stage_counts == res2.stage_counts


# ---------------------------------------------------------------------------
# 8. API Integration Tests
# ---------------------------------------------------------------------------
def _create_synthetic_png_bytes(width: int = 128, height: int = 128) -> bytes:
    """Helper to generate in-memory synthetic PNG bytes."""
    img, _ = generate_synthetic_wafer(size=width, seed=42)
    buf = BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()


def test_api_run_with_and_without_reference():
    """Verify POST /inspection/run executes golden path when reference is present and no-ref otherwise."""
    test_bytes = _create_synthetic_png_bytes(128, 128)
    ref_bytes = _create_synthetic_png_bytes(128, 128)

    # 1. Upload with reference
    upload_res = client.post(
        "/inspection/upload",
        files={
            "test_image": ("test.png", test_bytes, "image/png"),
            "reference_image": ("ref.png", ref_bytes, "image/png"),
        },
    )
    assert upload_res.status_code == 200
    upload_id = upload_res.json()["upload_id"]

    # Run inspection
    run_res = client.post("/inspection/run", json={"upload_id": upload_id})
    assert run_res.status_code == 200
    data = run_res.json()

    assert data["detection"]["path"] == "golden"
    assert data["detection"]["label"] == "Golden Reference Inspection"
    assert len(data["detection"]["stage_counts"]) >= 2
    assert "difference_image_url" in data
    assert "mask_image_url" in data

    # Verify difference and mask endpoints
    diff_res = client.get(data["difference_image_url"])
    assert diff_res.status_code == 200
    assert diff_res.headers["content-type"] == "image/png"

    mask_res = client.get(data["mask_image_url"])
    assert mask_res.status_code == 200
    assert mask_res.headers["content-type"] == "image/png"

    # 2. Upload without reference
    upload_no_ref = client.post(
        "/inspection/upload",
        files={"test_image": ("test_single.png", test_bytes, "image/png")},
    )
    assert upload_no_ref.status_code == 200
    upload_id_no_ref = upload_no_ref.json()["upload_id"]

    run_no_ref = client.post("/inspection/run", json={"upload_id": upload_id_no_ref})
    assert run_no_ref.status_code == 200
    data_no_ref = run_no_ref.json()
    assert data_no_ref["detection"]["path"] == "no_reference"
    assert data_no_ref["detection"]["label"] == "Experimental no-reference detection"


def test_api_reference_incompatible_aspect_ratio_returns_422():
    """Verify reference image with aspect ratio delta > 5% returns HTTP 422 REFERENCE_INCOMPATIBLE."""
    test_bytes = _create_synthetic_png_bytes(128, 128)  # aspect ratio 1.0

    # Incompatible reference (128x200, aspect ratio 0.64, delta 36% > 5%)
    buf = BytesIO()
    Image.new("L", (128, 200), color=128).save(buf, format="PNG")
    bad_ref_bytes = buf.getvalue()

    upload_res = client.post(
        "/inspection/upload",
        files={
            "test_image": ("test.png", test_bytes, "image/png"),
            "reference_image": ("incompatible_ref.png", bad_ref_bytes, "image/png"),
        },
    )
    assert upload_res.status_code == 200
    upload_id = upload_res.json()["upload_id"]

    run_res = client.post("/inspection/run", json={"upload_id": upload_id})
    assert run_res.status_code == 422
    detail = run_res.json()["detail"]
    assert detail["code"] == "REFERENCE_INCOMPATIBLE"
    assert "5% tolerance" in detail["message"]
