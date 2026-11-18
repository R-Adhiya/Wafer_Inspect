"""Backend tests for Phase 7 additions and Step 0 fixes.

Validates:
1. Simplified contour polygon present in original image coordinates (approxPolyDP, <= 200 pts,
   all points strictly within image bounds), both at scale=1.0 and scale=0.5.
2. GET /inspection/run/{upload_id}/result returns 404 (unknown/invalid UUID), 409 (run not finished),
   and 200 (validated InspectionRunResponse after run completes, including upload metadata).
3. Live stage status transitions through the exact real stages in order during a run
   (preparing_image -> preprocessing -> detecting -> segmenting -> filtering_noise -> measuring ->
   diagnosing_pattern -> generating_report -> completed) via callback and GET /status.
4. Step 0b: n = 3 or 4 adds "low defect count: interpret with caution" warning, and MC floor
   p-values format as "p < 0.001" rather than "0.0005".
"""

from io import BytesIO
import uuid
from fastapi.testclient import TestClient
import numpy as np
from PIL import Image

from app.cv.pattern_analysis import PatternConfig, analyze_spatial_pattern
from app.cv.pipeline import DetectionConfig, PreprocessingConfig, run_full_pipeline
from app.main import app
from app.services.status_store import PIPELINE_STAGES
from app.utils.synthetic import generate_layout_centroids, generate_layout_wafer

client = TestClient(app)


def _to_png_bytes(arr: np.ndarray) -> bytes:
    buf = BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def test_simplified_contour_present_capped_and_within_image_bounds():
    """Verify every measurement includes a simplified original-pixel contour (<= 200 pts, inside image)."""
    img, _, _ = generate_layout_wafer(
        defect_layout="edge_ring",
        count=5,
        size=512,
        seed=9101,
        grid_pitch=32,
        die_pattern="grid",
    )

    for max_side in (None, 256):
        prep_cfg = PreprocessingConfig(max_side=max_side)
        res = run_full_pipeline(
            test_image=img,
            reference_image=None,
            prep_config=prep_cfg,
            det_config=DetectionConfig(no_reference_mode="auto"),
        )
        assert len(res.measurements) == 5
        h, w = img.shape[:2]
        for m in res.measurements:
            assert len(m.contour) >= 3, f"Expected polygon with >= 3 points, got {len(m.contour)}"
            assert len(m.contour) <= 200, f"Contour exceeded 200-point cap: {len(m.contour)}"
            for px, py in m.contour:
                assert 0 <= px < w, f"Contour x={px} outside [0, {w})"
                assert 0 <= py < h, f"Contour y={py} outside [0, {h})"


def test_result_endpoint_404_409_and_200_validated():
    """Verify GET /inspection/run/{upload_id}/result handles 404, 409, and 200 correctly."""
    # 1. Invalid UUID -> 404
    bad_res = client.get("/inspection/run/not-a-uuid/result")
    assert bad_res.status_code == 404

    # 2. Unknown valid UUID -> 404
    unknown_id = str(uuid.uuid4())
    unk_res = client.get(f"/inspection/run/{unknown_id}/result")
    assert unk_res.status_code == 404

    # 3. Uploaded but NOT yet run -> 409 Conflict
    img, _, _ = generate_layout_wafer(
        defect_layout="donut",
        count=6,
        size=512,
        seed=9102,
        grid_pitch=32,
        die_pattern="grid",
    )
    up = client.post(
        "/inspection/upload",
        files={"test_image": ("donut_wafer.png", _to_png_bytes(img), "image/png")},
    )
    assert up.status_code == 200
    upload_id = up.json()["upload_id"]

    pending_res = client.get(f"/inspection/run/{upload_id}/result")
    assert pending_res.status_code == 409
    assert pending_res.json()["detail"]["code"] == "RUN_NOT_FINISHED"

    # 4. Run inspection, then GET /result -> 200 matching POST response schema
    run_res = client.post("/inspection/run", json={"upload_id": upload_id})
    assert run_res.status_code == 200
    post_body = run_res.json()

    get_res = client.get(f"/inspection/run/{upload_id}/result")
    assert get_res.status_code == 200
    get_body = get_res.json()

    assert get_body["upload_id"] == upload_id
    assert get_body["test_filename"] == "donut_wafer.png"
    assert get_body["reference_used"] is False
    assert get_body["summary"] == post_body["summary"]
    assert get_body["pattern"] == post_body["pattern"]
    assert len(get_body["regions"]) == len(post_body["regions"]) == 6
    for r in get_body["regions"]:
        assert isinstance(r["contour"], list) and len(r["contour"]) >= 3


def test_live_stage_status_transitions_in_real_order():
    """Verify pipeline callback and GET /status record the exact ordered real stages."""
    # A. Direct pipeline callback order
    img, _, _ = generate_layout_wafer(
        defect_layout="center_cluster",
        count=5,
        size=512,
        seed=9103,
        grid_pitch=32,
        die_pattern="grid",
    )
    observed_stages = []
    run_full_pipeline(
        test_image=img,
        reference_image=None,
        on_stage=lambda s: observed_stages.append(s),
    )
    assert observed_stages == [
        "preprocessing",
        "detecting",
        "segmenting",
        "filtering_noise",
        "measuring",
        "diagnosing_pattern",
    ]

    # B. API status lifecycle: 404 unknown -> not_started -> full ordered stage_history -> completed
    unknown_id = str(uuid.uuid4())
    assert client.get(f"/inspection/run/{unknown_id}/status").status_code == 404

    up = client.post(
        "/inspection/upload",
        files={"test_image": ("center_wafer.png", _to_png_bytes(img), "image/png")},
    )
    upload_id = up.json()["upload_id"]

    st_before = client.get(f"/inspection/run/{upload_id}/status")
    assert st_before.status_code == 200
    assert st_before.json()["status"] == "not_started"

    run_res = client.post("/inspection/run", json={"upload_id": upload_id})
    assert run_res.status_code == 200

    st_after = client.get(f"/inspection/run/{upload_id}/status")
    assert st_after.status_code == 200
    st_body = st_after.json()
    assert st_body["status"] == "completed"
    assert st_body["current_stage"] == "completed"
    assert st_body["completed_stages"] == PIPELINE_STAGES
    assert st_body["stage_history"] == PIPELINE_STAGES + ["completed"]


def test_step0b_low_count_warning_and_mc_floor_p_formatting():
    """Verify n=3 and n=4 emit low-count warning and MC floor formats as 'p < 0.001'."""
    center = (256.0, 256.0)
    radius = 225.28
    cfg = PatternConfig(mc_trials=2000, mc_seed=42)

    for n in (3, 4):
        pts, _ = generate_layout_centroids("center_cluster", count=n, center=center, radius=radius, seed=9105)
        recs = [
            {"id": i + 1, "centroid": pt, "area_px": 150, "length_px": 14.0, "elongation": 1.0, "severity": "MEDIUM"}
            for i, pt in enumerate(pts)
        ]
        res = analyze_spatial_pattern(recs, center, radius, (512, 512), config=cfg)
        assert any("low defect count: interpret with caution" in w for w in res.warnings)

    # Strong 8-point edge ring hits MC floor -> explanation says "p < 0.001", not "0.0005"
    pts_8, _ = generate_layout_centroids("edge_ring", count=8, center=center, radius=radius, seed=9106)
    recs_8 = [
        {"id": i + 1, "centroid": pt, "area_px": 150, "length_px": 14.0, "elongation": 1.0, "severity": "MEDIUM"}
        for i, pt in enumerate(pts_8)
    ]
    res_8 = analyze_spatial_pattern(recs_8, center, radius, (512, 512), config=cfg)
    assert res_8.type == "EDGE_RING"
    assert "p < 0.001" in res_8.explanation
    assert "0.0005" not in res_8.explanation
    assert not any("low defect count" in w for w in res_8.warnings)
