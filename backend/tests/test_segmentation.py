"""Deterministic synthetic tests for Phase 5 segmentation, noise filtering, and API endpoints.

Validates:
1. Connected-component segmentation of N well-separated defects and documented merging of touching defects.
2. Exact area_px == sum(cropped_mask), bbox containment of external contour, and Pick's-theorem contour area tolerance.
3. Noise filtering rules: size filter (1-3 px specks removed, 15x15 kept), noise-derived contrast filter,
   and diagnostic reason strings containing measured values.
4. Thin scratch survival (aspect ratio > 20) under default config and removal when max_aspect_ratio is enabled.
5. Polarity determination ("bright" vs "dark") from signed difference.
6. Deterministic region IDs (sorted by centroid y, x) and input immutability.
7. Wafer rim edge detection flag (touches_edge=True on rim, False in interior).
8. Clean zero-defect image produces empty accepted list without error.
9. Coordinate mapping back to original pixels via Phase 3 scale factor.
10. API response contract (regions, rejected_regions, full stage_counts) and image serving (200 / 400).
"""

from io import BytesIO

import cv2
import numpy as np
from app.cv.noise_filter import NoiseFilterConfig, compute_region_aspect_ratio, filter_regions
from app.cv.pipeline import DetectionConfig, run_full_pipeline
from app.cv.segmentation import (
    map_region_to_original_coords,
    segment_regions,
)
from app.main import app
from app.utils.synthetic import create_standard_synthetic_pair, generate_synthetic_wafer
from fastapi.testclient import TestClient
from PIL import Image

client = TestClient(app)


# ---------------------------------------------------------------------------
# 1. Well-Separated Defects & Touching Component Merging
# ---------------------------------------------------------------------------
def test_segmentation_n_well_separated_defects_and_touching_merge():
    """Verify N separated injected defects yield exactly N regions, while two touching defects merge into 1.

    Documented behavior: 8-connected component labeling merges spatially adjacent or overlapping
    foreground pixels into a single connected component. Two defects whose masks touch or overlap
    therefore produce 1 merged component rather than 2 separate regions.
    """
    # Part A: 4 well-separated defects on golden pair
    defects_4 = [
        {"type": "bright_blob", "center": (160.0, 170.0), "radius": 8.0, "contrast": 60.0},
        {"type": "dark_blob", "center": (340.0, 175.0), "radius": 8.0, "contrast": -60.0},
        {"type": "bright_blob", "center": (180.0, 330.0), "radius": 9.0, "contrast": 55.0},
        {"type": "dark_blob", "center": (330.0, 335.0), "radius": 7.0, "contrast": -55.0},
    ]
    ref_img, test_img, _ = create_standard_synthetic_pair(size=512, seed=101, defects=defects_4)

    result = run_full_pipeline(
        test_image=test_img,
        reference_image=ref_img,
        det_config=DetectionConfig(threshold_method="otsu"),
        filter_config=NoiseFilterConfig(min_area_px=12, contrast_sigma_k=3.0),
    )

    assert len(result.accepted_regions) == 4
    assert result.stage_counts[-1] == {"stage": "final", "candidate_count": 4}

    # Part B: Two touching blobs (centers 12 px apart, each radius 8 px -> overlapping by 4 px)
    touching_defects = [
        {"type": "bright_blob", "center": (250.0, 256.0), "radius": 8.0, "contrast": 65.0},
        {"type": "bright_blob", "center": (262.0, 256.0), "radius": 8.0, "contrast": 65.0},
    ]
    ref_touch, test_touch, _ = create_standard_synthetic_pair(size=512, seed=202, defects=touching_defects)

    res_touch = run_full_pipeline(
        test_image=test_touch,
        reference_image=ref_touch,
        det_config=DetectionConfig(threshold_method="otsu"),
        filter_config=NoiseFilterConfig(min_area_px=12, contrast_sigma_k=3.0),
    )

    # Touching defects merge under 8-connectivity into 1 single region whose area exceeds a single blob
    assert len(res_touch.accepted_regions) == 1
    single_blob_area = int(np.pi * 8.0 * 8.0)
    assert res_touch.accepted_regions[0].area_px > single_blob_area


# ---------------------------------------------------------------------------
# 2. Geometric Consistency: Area, Cropped Mask, BBox Containment, Contour Area
# ---------------------------------------------------------------------------
def test_region_area_bbox_contour_consistency_and_hole_handling():
    """Verify area_px == sum(mask), bbox contains whole contour, and contour area is within tolerance.

    Also verifies that interior holes are ignored by RETR_EXTERNAL contours while area_px retains
    the exact foreground pixel count.
    """
    mask = np.zeros((128, 128), dtype=np.uint8)
    signed = np.zeros((128, 128), dtype=np.int16)

    # Solid 16x16 square (area = 256 px)
    mask[20:36, 25:41] = 255
    signed[20:36, 25:41] = 50

    # Hollow 20x20 ring with 6x6 interior hole (true foreground area = 400 - 36 = 364 px)
    mask[70:90, 70:90] = 255
    mask[77:83, 77:83] = 0
    signed[70:90, 70:90] = -45

    regions = segment_regions(mask, signed, wafer_mask=np.full((128, 128), 255, dtype=np.uint8))
    assert len(regions) == 2

    solid_region, hollow_region = regions[0], regions[1]

    # Verify solid region exact metrics
    assert solid_region.area_px == int(np.sum(solid_region.mask)) == 256
    assert solid_region.mask.shape == (solid_region.bbox[3], solid_region.bbox[2])

    x, y, w, h = solid_region.bbox
    for px, py in solid_region.contour:
        assert x <= px < x + w
        assert y <= py < y + h

    cnt_np = np.array(solid_region.contour, dtype=np.int32).reshape(-1, 1, 2)
    contour_area = float(cv2.contourArea(cnt_np))
    perimeter = float(cv2.arcLength(cnt_np, closed=True))
    # By Pick's theorem, discrete pixel area vs pixel-center polygon area differs by ~perimeter / 2 + 1
    tolerance = perimeter / 2.0 + 2.0
    assert abs(solid_region.area_px - contour_area) <= tolerance

    # Verify hollow region keeps exact pixel count (364) despite RETR_EXTERNAL ignoring interior hole
    assert hollow_region.area_px == 364
    assert hollow_region.area_px == int(np.sum(hollow_region.mask))
    hx, hy, hw, hh = hollow_region.bbox
    for px, py in hollow_region.contour:
        assert hx <= px < hx + hw
        assert hy <= py < hy + hh


# ---------------------------------------------------------------------------
# 3. Noise Filter Rules: Size, Noise-Derived Contrast, and Reason Strings
# ---------------------------------------------------------------------------
def test_noise_filter_size_and_contrast_rules_with_measured_reasons():
    """Verify size filter removes 1-3 px specks & keeps 15x15 blob; contrast rule uses noise sigma."""
    mask = np.zeros((128, 128), dtype=np.uint8)
    signed = np.zeros((128, 128), dtype=np.int16)

    # 1 px speck (area = 1, high contrast) -> rejected by size
    mask[10, 10] = 255
    signed[10, 10] = 80

    # 3 px speck (area = 3, high contrast) -> rejected by size
    mask[10, 30:33] = 255
    signed[10, 30:33] = 80

    # 15x15 weak blob (area = 225, contrast = 10.0) -> survives size, rejected by contrast (limit = 3 * 5 = 15.0)
    mask[40:55, 20:35] = 255
    signed[40:55, 20:35] = 10

    # 15x15 strong blob (area = 225, contrast = 45.0) -> accepted
    mask[40:55, 70:85] = 255
    signed[40:55, 70:85] = 45

    regions = segment_regions(mask, signed, wafer_mask=np.full((128, 128), 255, dtype=np.uint8))
    assert len(regions) == 4

    noise_sigma = 5.0
    cfg = NoiseFilterConfig(min_area_px=12, contrast_sigma_k=3.0)
    res = filter_regions(regions, noise_sigma=noise_sigma, config=cfg)

    assert len(res.accepted_regions) == 1
    assert res.accepted_regions[0].area_px == 225
    assert res.accepted_regions[0].mean_contrast == 45.0
    assert "45.00" in res.accepted_regions[0].decision_reason

    assert len(res.rejected_regions) == 3
    by_rule = {rr.rule: [] for rr in res.rejected_regions}
    for rr in res.rejected_regions:
        by_rule[rr.rule].append(rr)

    assert len(by_rule["size_filter"]) == 2
    assert len(by_rule["contrast_filter"]) == 1

    # Verify reason strings embed measured numbers
    size_reasons = [rr.reason for rr in by_rule["size_filter"]]
    assert any("1" in r and "12" in r for r in size_reasons)
    assert any("3" in r and "12" in r for r in size_reasons)

    contrast_reason = by_rule["contrast_filter"][0].reason
    assert "10.00" in contrast_reason
    assert "15.00" in contrast_reason
    assert "5.00" in contrast_reason

    # Verify stage_counts includes both rules that removed regions
    stages = [sc["stage"] for sc in res.stage_counts]
    assert stages == ["after_size_filter", "after_contrast_filter"]
    assert res.stage_counts[0]["candidate_count"] == 2
    assert res.stage_counts[1]["candidate_count"] == 1


# ---------------------------------------------------------------------------
# 4. Thin Scratch Survival Under Default Config & Explicit Aspect Ratio Removal
# ---------------------------------------------------------------------------
def test_thin_scratch_survives_default_and_removed_only_with_max_aspect_ratio():
    """Verify a thin scratch (aspect ratio > 20) survives default config and is filtered only if enabled."""
    mask = np.zeros((160, 160), dtype=np.uint8)
    signed = np.zeros((160, 160), dtype=np.int16)

    # Draw thin scratch: 80 px long, 2 px thick -> aspect ratio ~ 40 (> 20)
    mask[80:82, 40:120] = 255
    signed[80:82, 40:120] = 60

    regions = segment_regions(mask, signed, wafer_mask=np.full((160, 160), 255, dtype=np.uint8))
    assert len(regions) == 1
    measured_aspect = compute_region_aspect_ratio(regions[0])
    assert measured_aspect > 20.0

    # Default config: max_aspect_ratio is None (OFF) -> scratch MUST survive
    default_res = filter_regions(regions, noise_sigma=5.0, config=NoiseFilterConfig())
    assert len(default_res.accepted_regions) == 1
    assert len(default_res.rejected_regions) == 0

    # Explicit max_aspect_ratio=15.0 -> scratch is rejected with measured aspect ratio in reason
    strict_res = filter_regions(regions, noise_sigma=5.0, config=NoiseFilterConfig(max_aspect_ratio=15.0))
    assert len(strict_res.accepted_regions) == 0
    assert len(strict_res.rejected_regions) == 1
    assert strict_res.rejected_regions[0].rule == "aspect_ratio_filter"
    assert "15.00" in strict_res.rejected_regions[0].reason


# ---------------------------------------------------------------------------
# 5. Polarity Detection ("bright" vs "dark")
# ---------------------------------------------------------------------------
def test_polarity_bright_and_dark_defects():
    """Verify bright injected defects get polarity='bright' and dark defects get polarity='dark'."""
    defects = [
        {"type": "bright_blob", "center": (180.0, 200.0), "radius": 9.0, "contrast": 60.0},
        {"type": "dark_blob", "center": (320.0, 300.0), "radius": 9.0, "contrast": -60.0},
    ]
    ref_img, test_img, _ = create_standard_synthetic_pair(size=512, seed=303, defects=defects)

    result = run_full_pipeline(test_image=test_img, reference_image=ref_img)
    assert len(result.accepted_regions) == 2

    # Sorted by centroid y: (180, 200) is region #1 (bright), (320, 300) is region #2 (dark)
    assert result.accepted_regions[0].polarity == "bright"
    assert result.accepted_regions[1].polarity == "dark"


# ---------------------------------------------------------------------------
# 6. Determinism & Input Immutability
# ---------------------------------------------------------------------------
def test_segmentation_determinism_and_no_input_mutation():
    """Verify region IDs and masks are deterministic across runs and inputs are never mutated."""
    defects = [
        {"type": "bright_blob", "center": (300.0, 180.0), "radius": 7.0, "contrast": 55.0},
        {"type": "dark_blob", "center": (180.0, 260.0), "radius": 8.0, "contrast": -55.0},
        {"type": "bright_blob", "center": (260.0, 340.0), "radius": 7.0, "contrast": 55.0},
    ]
    ref_img, test_img, _ = create_standard_synthetic_pair(size=512, seed=404, defects=defects)
    ref_copy = ref_img.copy()
    test_copy = test_img.copy()

    res1 = run_full_pipeline(test_image=test_img, reference_image=ref_img)
    res2 = run_full_pipeline(test_image=test_img, reference_image=ref_img)

    # Inputs untouched
    assert np.array_equal(ref_img, ref_copy)
    assert np.array_equal(test_img, test_copy)

    # Identical region IDs, centroids, areas, and final masks
    assert [(r.id, r.centroid, r.area_px, r.polarity) for r in res1.accepted_regions] == [
        (r.id, r.centroid, r.area_px, r.polarity) for r in res2.accepted_regions
    ]
    assert np.array_equal(res1.filtered_mask, res2.filtered_mask)
    assert np.array_equal(res1.label_map, res2.label_map)


# ---------------------------------------------------------------------------
# 7. Wafer Rim Edge Flag
# ---------------------------------------------------------------------------
def test_wafer_rim_touches_edge_flag():
    """Verify defect on wafer rim has touches_edge=True (and is kept) while interior defect has False."""
    # Wafer center=(256, 256), true radius=225.28, effective mask radius (~2% margin) = 220.8.
    # Place rim defect centered at (256 + 216, 256) with radius 10 -> reaches 226 and touches wafer_mask border.
    defects = [
        {"type": "bright_blob", "center": (256.0, 256.0), "radius": 9.0, "contrast": 65.0},
        {"type": "bright_blob", "center": (472.0, 256.0), "radius": 10.0, "contrast": 65.0},
    ]
    ref_img, test_img, _ = create_standard_synthetic_pair(size=512, seed=505, defects=defects)

    result = run_full_pipeline(test_image=test_img, reference_image=ref_img)
    assert len(result.accepted_regions) == 2

    interior = min(result.accepted_regions, key=lambda r: abs(r.centroid[0] - 256.0))
    rim = min(result.accepted_regions, key=lambda r: abs(r.centroid[0] - 472.0))

    assert interior.touches_edge is False
    assert rim.touches_edge is True


# ---------------------------------------------------------------------------
# 8. Zero-Defect Image & Coordinate Scale Mapping
# ---------------------------------------------------------------------------
def test_zero_defect_image_returns_empty_accepted_list():
    """Verify clean wafer pair produces empty accepted list and empty filtered mask without error."""
    clean_ref, _ = generate_synthetic_wafer(size=512, seed=601, noise_std=5.0, defects=None)
    clean_test, _ = generate_synthetic_wafer(size=512, seed=602, noise_std=5.0, defects=None)

    result = run_full_pipeline(test_image=clean_test, reference_image=clean_ref)
    assert result.accepted_regions == []
    assert int(np.count_nonzero(result.filtered_mask)) == 0
    assert result.stage_counts[-1] == {"stage": "final", "candidate_count": 0}


def test_scale_factor_coordinate_mapping():
    """Verify map_region_to_original_coords accurately maps processed pixels back to original pixels."""
    orig_bbox, orig_centroid = map_region_to_original_coords(
        bbox=(50, 100, 10, 20),
        centroid=(55.0, 110.0),
        scale_factor=0.5,
    )
    assert orig_bbox == (100, 200, 20, 40)
    assert orig_centroid == (110.0, 220.0)


# ---------------------------------------------------------------------------
# 9. API Integration Tests
# ---------------------------------------------------------------------------
def _to_png_bytes(arr: np.ndarray) -> bytes:
    buf = BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def test_api_phase5_regions_rejected_and_image_kinds():
    """Verify POST /inspection/run returns regions, rejected_regions, full stage_counts, and serves new images."""
    defects = [
        {"type": "bright_blob", "center": (180.0, 190.0), "radius": 8.0, "contrast": 60.0},
        {"type": "dark_blob", "center": (320.0, 310.0), "radius": 8.0, "contrast": -60.0},
    ]
    ref_img, test_img, _ = create_standard_synthetic_pair(size=512, seed=707, defects=defects)

    upload_res = client.post(
        "/inspection/upload",
        files={
            "test_image": ("wafer_test.png", _to_png_bytes(test_img), "image/png"),
            "reference_image": ("wafer_ref.png", _to_png_bytes(ref_img), "image/png"),
        },
    )
    assert upload_res.status_code == 200
    upload_id = upload_res.json()["upload_id"]

    run_res = client.post("/inspection/run", json={"upload_id": upload_id})
    assert run_res.status_code == 200
    body = run_res.json()

    assert "regions" in body
    assert "rejected_regions" in body
    assert "stage_counts" in body
    assert len(body["regions"]) == 2
    assert body["stage_counts"][0]["stage"] == "raw_threshold"
    assert body["stage_counts"][-1]["stage"] == "final"
    assert body["stage_counts"][-1]["candidate_count"] == 2
    assert body["stage_counts"][-1]["note"] is None

    # Check required fields on each accepted region
    for r in body["regions"]:
        for key in ("id", "bbox", "area_px", "centroid", "polarity", "mean_contrast", "touches_edge"):
            assert key in r

    # Verify all Phase 5 image kinds return HTTP 200 PNG
    for kind in ("original", "preprocessed", "difference", "mask", "filtered_mask", "bboxes", "contours"):
        img_res = client.get(f"/inspection/run/{upload_id}/image/{kind}")
        assert img_res.status_code == 200, f"Expected 200 for kind='{kind}', got {img_res.status_code}"
        assert img_res.headers["content-type"] == "image/png"

    # Unknown image kind must return HTTP 400
    bad_res = client.get(f"/inspection/run/{upload_id}/image/unknown_kind")
    assert bad_res.status_code == 400
    assert bad_res.json()["detail"]["code"] == "INVALID_KIND"
