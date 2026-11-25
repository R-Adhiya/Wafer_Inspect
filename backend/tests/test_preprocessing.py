"""Tests for Phase 3: CV Preprocessing algorithms, pipeline execution, and API endpoints."""

import io

import numpy as np
import pytest
from app.core.config import settings
from app.cv.pipeline import run_preprocessing
from app.cv.preprocessing import (
    PreprocessingConfig,
    denoise,
    enhance_contrast,
    normalize,
    resize_preserving_aspect,
    tile_image,
    to_grayscale,
    verify_tiling_coverage,
)
from app.main import app
from fastapi.testclient import TestClient
from PIL import Image

client = TestClient(app)


# ---------------------------------------------------------------------------
# 1. Grayscale Conversion Tests
# ---------------------------------------------------------------------------
def test_grayscale_pure_red():
    """Verify pure red RGB (255, 0, 0) converts to 76 using ITU-R 601 weights."""
    rgb = np.zeros((1, 1, 3), dtype=np.uint8)
    rgb[0, 0] = [255, 0, 0]
    gray = to_grayscale(rgb)
    assert gray.shape == (1, 1)
    assert gray.dtype == np.uint8
    assert gray[0, 0] == 76


def test_grayscale_rgba_ignores_alpha():
    """Verify RGBA ignores alpha channel and yields identical luma to RGB."""
    rgba = np.zeros((1, 1, 4), dtype=np.uint8)
    rgba[0, 0] = [255, 0, 0, 50]  # Alpha 50 must be ignored
    gray = to_grayscale(rgba)
    assert gray.shape == (1, 1)
    assert gray[0, 0] == 76


def test_grayscale_16bit_scaling():
    """Verify 16-bit grayscale values are scaled by dividing by 257 (65535 -> 255, 257 -> 1)."""
    i16 = np.array([[65535, 257, 0]], dtype=np.uint16)
    gray = to_grayscale(i16)
    assert gray.dtype == np.uint8
    assert gray[0, 0] == 255
    assert gray[0, 1] == 1
    assert gray[0, 2] == 0


def test_grayscale_8bit_unchanged():
    """Verify 8-bit grayscale image is returned with identical values without mutating input."""
    g8 = np.array([[42, 180], [90, 220]], dtype=np.uint8)
    res = to_grayscale(g8)
    assert np.array_equal(res, g8)
    assert res is not g8  # Must be a copy, not the same reference


# ---------------------------------------------------------------------------
# 2. Aspect-Preserving Resize Tests
# ---------------------------------------------------------------------------
def test_resize_preserves_aspect_ratio_and_returns_scale():
    """Verify downscaling preserves aspect ratio and returns exact scale factor."""
    img = np.zeros((200, 100), dtype=np.uint8)
    resized, scale = resize_preserving_aspect(img, max_side=100)
    assert scale == 0.5
    assert resized.shape == (100, 50)


def test_resize_never_upscales():
    """Verify an image smaller than max_side is not enlarged."""
    img = np.zeros((50, 40), dtype=np.uint8)
    resized, scale = resize_preserving_aspect(img, max_side=100)
    assert scale == 1.0
    assert resized.shape == (50, 40)


# ---------------------------------------------------------------------------
# 3. Image Tiling and Coverage Tests
# ---------------------------------------------------------------------------
def test_tiling_full_coverage_and_edge_remainders():
    """Verify tiles cover 100% of the image including edge remainders."""
    img = np.arange(100 * 100, dtype=np.uint8).reshape((100, 100))
    tile_size = 64
    overlap = 16
    tiles = tile_image(img, tile_size=tile_size, overlap=overlap)

    assert len(tiles) > 0
    # Every tile should be tile_size x tile_size
    for tile_data, x, y in tiles:
        assert tile_data.shape == (tile_size, tile_size)
        assert x >= 0 and y >= 0
        assert x + tile_size <= 100
        assert y + tile_size <= 100

    # Prove 100% pixel coverage
    assert verify_tiling_coverage(img.shape, tiles) is True


# ---------------------------------------------------------------------------
# 4. Denoising Tests
# ---------------------------------------------------------------------------
def test_denoise_gaussian_lowers_std():
    """Verify Gaussian filter reduces standard deviation of a noisy image."""
    rng = np.random.RandomState(42)
    flat_noisy = np.clip(128 + rng.normal(0, 15, (100, 100)), 0, 255).astype(np.uint8)
    std_before = float(np.std(flat_noisy))

    denoised = denoise(flat_noisy, method="gaussian", kernel_size=5)
    std_after = float(np.std(denoised))

    assert std_after < std_before


def test_denoise_median_removes_salt_and_pepper():
    """Verify median filter suppresses extreme impulse noise pixels."""
    img = np.full((100, 100), 128, dtype=np.uint8)
    # Inject salt and pepper impulses
    img[10:15, 10:15] = 255
    img[50:55, 50:55] = 0
    extremes_before = int(np.sum((img == 255) | (img == 0)))

    denoised = denoise(img, method="median", kernel_size=5)
    extremes_after = int(np.sum((denoised == 255) | (denoised == 0)))

    assert extremes_after < extremes_before


def test_denoise_even_kernel_raises_error():
    """Verify even or non-positive kernel sizes raise ValueError."""
    img = np.zeros((64, 64), dtype=np.uint8)
    with pytest.raises(ValueError, match="positive odd integer"):
        denoise(img, kernel_size=4)
    with pytest.raises(ValueError, match="positive odd integer"):
        denoise(img, kernel_size=0)


# ---------------------------------------------------------------------------
# 5. Normalization Tests
# ---------------------------------------------------------------------------
def test_normalize_ranges():
    """Verify min-max normalization produces correct bounds and dtypes."""
    img = np.array([[50, 100], [150, 200]], dtype=np.uint8)

    norm_255 = normalize(img, mode="0-255")
    assert norm_255.dtype == np.uint8
    assert np.min(norm_255) == 0
    assert np.max(norm_255) == 255

    norm_1 = normalize(img, mode="0-1")
    assert norm_1.dtype == np.float32
    assert np.isclose(float(np.min(norm_1)), 0.0)
    assert np.isclose(float(np.max(norm_1)), 1.0)


def test_normalize_flat_image_returns_zeros():
    """Verify flat image (max == min) safely returns zeros without ZeroDivisionError."""
    flat = np.full((64, 64), 128, dtype=np.uint8)

    norm_255 = normalize(flat, mode="0-255")
    assert norm_255.dtype == np.uint8
    assert np.all(norm_255 == 0)

    norm_1 = normalize(flat, mode="0-1")
    assert norm_1.dtype == np.float32
    assert np.all(norm_1 == 0.0)


# ---------------------------------------------------------------------------
# 6. Contrast Enhancement Tests
# ---------------------------------------------------------------------------
def test_clahe_raises_std_on_low_contrast():
    """Verify CLAHE increases contrast on a low-contrast image."""
    rng = np.random.RandomState(42)
    low_contrast = rng.randint(120, 130, size=(100, 100), dtype=np.uint8)
    std_before = float(np.std(low_contrast))

    enhanced, decision = enhance_contrast(low_contrast, method="clahe")
    std_after = float(np.std(enhanced))

    assert decision["applied"] is True
    assert std_after > std_before


def test_auto_contrast_skips_high_contrast_and_applies_to_low():
    """Verify auto mode only applies CLAHE when criteria are met using default parameters."""
    # 1. High contrast, uniform quadrants -> skip using defaults
    quad = np.tile(np.linspace(10, 240, 50, dtype=np.uint8), (50, 1))
    high_contrast = np.block([[quad, quad], [quad, quad]])
    _, decision_skip = enhance_contrast(high_contrast)
    assert decision_skip["applied"] is False
    assert "Skipped CLAHE" in decision_skip["reason"]
    assert str(decision_skip["std"]) in decision_skip["reason"]

    # 2. Low contrast -> apply using defaults
    rng = np.random.RandomState(42)
    low_contrast = rng.randint(120, 130, size=(100, 100), dtype=np.uint8)
    _, decision_apply = enhance_contrast(low_contrast)
    assert decision_apply["applied"] is True
    assert "Applied CLAHE" in decision_apply["reason"]
    assert str(decision_apply["std"]) in decision_apply["reason"]


# ---------------------------------------------------------------------------
# 7. Determinism and Immutability Tests
# ---------------------------------------------------------------------------
def test_pipeline_determinism_and_immutability():
    """Verify executing preprocessing twice with same config yields identical arrays without mutating input."""
    rng = np.random.RandomState(42)
    original = rng.randint(50, 200, size=(128, 128), dtype=np.uint8)
    original_copy = original.copy()

    config = PreprocessingConfig(
        to_grayscale_enabled=True,
        denoise_enabled=True,
        denoise_method="gaussian",
        denoise_kernel_size=5,
        contrast_method="auto",
        normalize_enabled=True,
        normalize_mode="0-255",
    )

    out1, log1 = run_preprocessing(original, config)
    out2, log2 = run_preprocessing(original, config)

    # Determinism: exact same pixel values
    assert np.array_equal(out1, out2)

    # Immutability: original array is pristine
    assert np.array_equal(original, original_copy)


# ---------------------------------------------------------------------------
# 8. Inspection Run API Tests
# ---------------------------------------------------------------------------
def test_api_run_inspection_success_and_retrieval():
    """Verify POST /inspection/run executes preprocessing, saves run.json, and serves images."""
    # 1. Upload a synthetic test image first
    img = Image.new("L", (128, 128), color=120)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    upload_res = client.post(
        "/inspection/upload",
        files={"test_image": ("wafer_for_run.png", buf.getvalue(), "image/png")},
    )
    assert upload_res.status_code == 200
    upload_id = upload_res.json()["upload_id"]

    # 2. Trigger preprocessing run with denoise override
    run_res = client.post(
        "/inspection/run",
        json={
            "upload_id": upload_id,
            "config": {
                "denoise_enabled": True,
                "denoise_method": "gaussian",
                "denoise_kernel_size": 3,
            },
        },
    )
    assert run_res.status_code == 200
    data = run_res.json()
    assert data["upload_id"] == upload_id
    assert len(data["step_log"]) > 0

    # Verify denoise step was applied as configured
    denoise_entry = next((s for s in data["step_log"] if s["step"] == "denoise"), None)
    assert denoise_entry is not None
    assert denoise_entry["applied"] is True

    # Confirm stats are present
    assert data["original_stats"]["width"] == 128
    assert data["original_stats"]["height"] == 128
    assert data["preprocessed_stats"]["width"] == 128
    assert data["preprocessed_stats"]["height"] == 128

    # Confirm saved run.json on disk
    run_json_path = settings.results_path / upload_id / "run.json"
    assert run_json_path.is_file()

    # 3. Fetch original image via run route
    orig_img_res = client.get(f"/inspection/run/{upload_id}/image/original")
    assert orig_img_res.status_code == 200
    assert orig_img_res.headers["content-type"] in ("image/png", "image/jpeg")

    # 4. Fetch preprocessed image via run route
    prep_img_res = client.get(f"/inspection/run/{upload_id}/image/preprocessed")
    assert prep_img_res.status_code == 200
    assert prep_img_res.headers["content-type"] == "image/png"


def test_api_run_unknown_upload_id():
    """Verify POST /inspection/run returns 404 for non-existent upload_id."""
    res = client.post(
        "/inspection/run",
        json={"upload_id": "00000000-0000-0000-0000-000000000000"},
    )
    assert res.status_code == 404
    assert res.json()["detail"]["code"] == "UPLOAD_NOT_FOUND"
