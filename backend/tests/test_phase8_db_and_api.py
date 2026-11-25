"""Comprehensive Phase 8 tests for SQLAlchemy 2.0 CRUD, Alembic migrations, PostgreSQL readiness,
database-backed API endpoints, history queries, dashboard stats, samples, and failure handling.
"""

from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from app.core.config import settings
from app.db import crud
from app.db.database import Base, build_engine
from app.db.migrations import upgrade_database_to_head
from app.db.models import Defect, Inspection, ProcessingRun
from app.services.sample_service import SYNTHETIC_DISCLAIMER
from app.utils.synthetic import generate_layout_wafer
from PIL import Image
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.schema import CreateIndex, CreateTable


def _png_bytes(arr: np.ndarray) -> bytes:
    buf = BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 1. CRUD Unit Tests (In-Memory SQLite): Create, Atomicity, Cascade, FK
# ---------------------------------------------------------------------------
def test_crud_create_fk_enforcement_cascade_and_atomicity(memory_db_session):
    """Verify create, FK rejection, cascade delete, and atomic rollback on mid-save failure."""
    db = memory_db_session
    # 1. Create pending inspection
    row = crud.create_inspection(
        db,
        upload_id="00000000-0000-0000-0000-000000000001",
        original_filename="wafer_test.png",
        image_width=512,
        image_height=512,
        image_mode="L",
        image_format="PNG",
        file_size=12345,
    )
    assert row.id > 0
    assert row.status == "pending"

    # 2. Foreign key enforcement: inserting a Defect with a nonexistent inspection_id MUST fail
    bad_defect = Defect(
        inspection_id=999999,
        region_no=1,
        area=50,
        detected_area=50,
        width=8,
        height=8,
        centroid_x=100.0,
        centroid_y=100.0,
        bbox_x=96,
        bbox_y=96,
        bbox_width=8,
        bbox_height=8,
        perimeter=25.0,
        equivalent_diameter=8.0,
        length=8.0,
        thickness=8.0,
        angle_deg=0.0,
        elongation=1.0,
        polarity="bright",
        mean_contrast=60.0,
        peak_contrast=70.0,
        contrast_to_noise=10.0,
        touches_edge=False,
        detected_by="golden",
        severity="LOW",
        why_flagged="test",
        contour=[[96, 96], [104, 96], [104, 104]],
    )
    db.add(bad_defect)
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()

    # 3. Atomicity of save_results: an error mid-transaction leaves NO partial rows
    valid_defect_dict = {
        "region_no": 1,
        "area": 100,
        "detected_area": 90,
        "width": 10,
        "height": 10,
        "centroid_x": 120.0,
        "centroid_y": 130.0,
        "bbox_x": 115,
        "bbox_y": 125,
        "bbox_width": 10,
        "bbox_height": 10,
        "perimeter": 35.0,
        "equivalent_diameter": 11.3,
        "length": 10.0,
        "thickness": 10.0,
        "angle_deg": 0.0,
        "elongation": 1.0,
        "polarity": "bright",
        "mean_contrast": 55.0,
        "peak_contrast": 65.0,
        "contrast_to_noise": 9.0,
        "touches_edge": False,
        "detected_by": "self_reference",
        "severity": "MEDIUM",
        "why_flagged": "test defect",
        "contour": [[115, 125], [125, 125], [125, 135]],
    }
    # Deliberately pass invalid NULL for non-nullable processing_run field to trigger mid-save failure
    bad_run_record = {
        "preprocessing_method": None,  # NOT NULL violation after defect was added
        "threshold_method": "otsu",
        "threshold_value": 25.0,
        "morphology_method": "open:3,close:3",
        "minimum_area": 12,
        "config": {},
        "stage_timings": {},
        "processing_time": 0.1,
    }
    with pytest.raises(IntegrityError):
        crud.save_results(
            db,
            row.id,
            inspection_fields={"defect_count": 1, "severity": "MEDIUM"},
            defect_records=[valid_defect_dict],
            processing_run_record=bad_run_record,
        )

    # Verify rollback left ZERO defects, ZERO runs, and original pending status
    reloaded = crud.get_inspection(db, row.id)
    assert reloaded is not None
    assert reloaded.status == "pending"
    assert reloaded.defect_count == 0
    assert len(reloaded.defects) == 0
    assert len(reloaded.processing_runs) == 0

    # 4. Valid save_results commits atomically, then cascade delete removes children
    good_run_record = {**bad_run_record, "preprocessing_method": "gray=True"}
    saved = crud.save_results(
        db,
        row.id,
        inspection_fields={"defect_count": 1, "total_defect_area": 100, "severity": "MEDIUM"},
        defect_records=[valid_defect_dict],
        processing_run_record=good_run_record,
    )
    assert saved.status == "completed"
    assert len(saved.defects) == 1
    assert len(saved.processing_runs) == 1

    deleted_uid = crud.delete_inspection(db, row.id)
    assert deleted_uid == row.upload_id
    assert db.query(Inspection).count() == 0
    assert db.query(Defect).count() == 0
    assert db.query(ProcessingRun).count() == 0


# ---------------------------------------------------------------------------
# 2. PostgreSQL Dialect Readiness & Alembic Migration Parity
# ---------------------------------------------------------------------------
def test_postgresql_dialect_compilation_and_alembic_schema_parity(tmp_path: Path):
    """Verify all tables/indexes compile on PostgreSQL dialect and alembic upgrade matches models."""
    pg_dialect = postgresql.dialect()
    for table in Base.metadata.sorted_tables:
        ddl = str(CreateTable(table).compile(dialect=pg_dialect))
        assert f"CREATE TABLE {table.name}" in ddl
        assert "AUTOINCREMENT" not in ddl.upper()
        for idx in table.indexes:
            idx_ddl = str(CreateIndex(idx).compile(dialect=pg_dialect))
            assert "CREATE" in idx_ddl and "INDEX" in idx_ddl

    # Run alembic upgrade head on a brand-new empty SQLite database
    empty_db = tmp_path / "alembic_parity.db"
    db_url = f"sqlite:///{empty_db.as_posix()}"
    upgrade_database_to_head(db_url)

    eng = build_engine(db_url)
    try:
        insp = sa_inspect(eng)
        migrated_tables = set(insp.get_table_names()) - {"alembic_version"}
        model_tables = set(Base.metadata.tables.keys())
        assert migrated_tables == model_tables

        for tname, table in Base.metadata.tables.items():
            migrated_cols = {c["name"] for c in insp.get_columns(tname)}
            model_cols = {c.name for c in table.columns}
            assert migrated_cols == model_cols, f"Column mismatch on '{tname}': {migrated_cols ^ model_cols}"
    finally:
        eng.dispose()


# ---------------------------------------------------------------------------
# 3. API End-to-End Parity (DB Report == Run Response) & Persistence Across Engine Disposal
# ---------------------------------------------------------------------------
def test_api_upload_run_db_parity_persistence_and_delete(isolated_api_client):
    """Verify pending row, DB-built GET /inspection/{id} == POST /run field-by-field, persistence, and delete."""
    client, eng, Session, db_file, db_url = isolated_api_client
    img, _, _ = generate_layout_wafer("edge_ring", count=6, size=512, seed=8501, grid_pitch=32)
    up = client.post(
        "/inspection/upload",
        files={"test_image": ("parity_edge_ring.png", _png_bytes(img), "image/png")},
    )
    assert up.status_code == 200
    up_body = up.json()
    insp_id = up_body["inspection_id"]
    upload_id = up_body["upload_id"]

    # Pending row exists in DB before run
    with Session() as s:
        pending_row = crud.get_inspection(s, insp_id)
        assert pending_row is not None
        assert pending_row.status == "pending"

    # Run using integer inspection_id
    run_res = client.post("/inspection/run", json={"inspection_id": insp_id})
    assert run_res.status_code == 200
    run_body = run_res.json()
    assert run_body["inspection_id"] == insp_id
    assert len(run_body["regions"]) == 6

    # Delete run.json to PROVE GET /inspection/{id} reads strictly from DB rows, never run.json!
    run_json_path = settings.results_path / upload_id / "run.json"
    assert run_json_path.is_file()
    run_json_path.unlink()
    assert not run_json_path.exists()

    # GET /inspection/{id} MUST equal POST /inspection/run field by field
    get_res = client.get(f"/inspection/{insp_id}")
    assert get_res.status_code == 200
    assert get_res.json() == run_body

    # GET /inspection/{id}/defects & /images & /image/{kind} & /status
    defects_res = client.get(f"/inspection/{insp_id}/defects")
    assert defects_res.status_code == 200
    assert defects_res.json() == run_body["regions"]

    imgs_res = client.get(f"/inspection/{insp_id}/images")
    assert imgs_res.status_code == 200
    imgs_map = imgs_res.json()["images"]
    assert "final_overlay" in imgs_map and "pattern_overlay" in imgs_map
    for kind, url in imgs_map.items():
        img_fetch = client.get(url)
        assert img_fetch.status_code == 200, f"Failed fetching {kind} from {url}"

    st_res = client.get(f"/inspection/{insp_id}/status")
    assert st_res.status_code == 200
    assert st_res.json()["status"] == "completed"

    # Persistence across engine disposal: dispose current engine, open fresh engine on same SQLite file
    eng.dispose()
    fresh_eng = build_engine(db_url)
    FreshSession = sessionmaker(bind=fresh_eng, expire_on_commit=False)
    try:
        with FreshSession() as fs:
            persisted = crud.get_inspection(fs, insp_id)
            assert persisted is not None
            assert persisted.defect_count == 6
            assert len(persisted.defects) == 6
            assert len(persisted.processing_runs) == 1
    finally:
        fresh_eng.dispose()

    # Delete removes DB rows AND upload/result files on disk; deleting unknown ID returns 404
    del_res = client.delete(f"/inspection/{insp_id}")
    assert del_res.status_code == 200
    assert del_res.json()["deleted"] is True
    assert not (settings.upload_path / upload_id).exists()
    assert not (settings.results_path / upload_id).exists()
    assert client.get(f"/inspection/{insp_id}").status_code == 404
    assert client.delete("/inspection/999999").status_code == 404


# ---------------------------------------------------------------------------
# 4. History Search, Severity Filter, All Sort Keys & Directions, Pagination
# ---------------------------------------------------------------------------
def test_history_search_filter_sort_and_pagination_over_12_rows(isolated_api_client):
    """Verify history search, severity filter, every sort key/order, and pagination on 12+ rows."""
    client, _, Session, _, _ = isolated_api_client
    severities = ["NONE", "LOW", "MEDIUM", "HIGH"]
    with Session() as s:
        for i in range(14):
            sev = severities[i % 4]
            row = crud.create_inspection(
                s,
                upload_id=f"00000000-0000-0000-0000-{i:012d}",
                original_filename=f"lot_{'alpha' if i % 2 == 0 else 'beta'}_wafer_{i:02d}.png",
                image_width=512,
                image_height=512,
                image_mode="L",
                image_format="PNG",
                file_size=10000 + i,
            )
            crud.save_results(
                s,
                row.id,
                inspection_fields={
                    "defect_count": i * 2,
                    "total_defect_area": i * 100,
                    "largest_defect_area": i * 50,
                    "average_defect_area": 50.0 if i > 0 else 0.0,
                    "severity": sev,
                    "pattern_type": "EDGE_RING" if sev == "HIGH" else "RANDOM_SCATTER",
                    "processing_time": round(0.1 + (14 - i) * 0.05, 3),
                },
                defect_records=[],
                processing_run_record={
                    "preprocessing_method": "gray=True",
                    "threshold_method": "otsu",
                    "threshold_value": 25.0,
                    "morphology_method": "open:3,close:3",
                    "minimum_area": 12,
                    "config": {},
                    "stage_timings": {},
                    "processing_time": round(0.1 + (14 - i) * 0.05, 3),
                },
            )

    # Search by filename
    r_search = client.get("/inspection/history?search=ALPHA&page_size=20")
    assert r_search.status_code == 200
    assert r_search.json()["total"] == 7
    assert all("alpha" in item["original_filename"] for item in r_search.json()["items"])

    # Filter by severity
    r_sev = client.get("/inspection/history?severity=HIGH&page_size=20")
    assert r_sev.status_code == 200
    assert r_sev.json()["total"] == 3  # indices 3, 7, 11
    assert all(item["severity"] == "HIGH" for item in r_sev.json()["items"])

    # Sort keys & directions
    for sort_key in ("created_at", "defect_count", "severity", "processing_time"):
        for order in ("asc", "desc"):
            r_sort = client.get(f"/inspection/history?sort={sort_key}&order={order}&page_size=20")
            assert r_sort.status_code == 200
            items = r_sort.json()["items"]
            assert len(items) == 14
            if sort_key == "defect_count":
                vals = [x["defect_count"] for x in items]
            elif sort_key == "processing_time":
                vals = [x["processing_time"] for x in items]
            elif sort_key == "severity":
                rank = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}
                vals = [rank[x["severity"]] for x in items]
            else:
                vals = [x["id"] for x in items]
            expected_vals = sorted(vals, reverse=(order == "desc"))
            assert vals == expected_vals, f"Sort failed for {sort_key} {order}: {vals}"

    # Pagination
    p1 = client.get("/inspection/history?page=1&page_size=5").json()
    p2 = client.get("/inspection/history?page=2&page_size=5").json()
    p3 = client.get("/inspection/history?page=3&page_size=5").json()
    assert p1["total"] == 14 and p1["total_pages"] == 3
    assert len(p1["items"]) == 5 and len(p2["items"]) == 5 and len(p3["items"]) == 4
    ids_all = [x["id"] for x in p1["items"] + p2["items"] + p3["items"]]
    assert len(set(ids_all)) == 14


# ---------------------------------------------------------------------------
# 5. Dashboard Stats: Empty Safe + Exact Hand-Computed SQL Aggregates
# ---------------------------------------------------------------------------
def test_dashboard_stats_empty_and_hand_computed_aggregates(isolated_api_client):
    """Verify empty DB returns zeros cleanly and populated DB matches exact hand-computed aggregates."""
    client, _, Session, _, _ = isolated_api_client
    # Empty DB
    empty = client.get("/dashboard/stats")
    assert empty.status_code == 200
    eb = empty.json()
    assert eb["total_inspections"] == 0
    assert eb["total_defects"] == 0
    assert eb["high_severity_inspections"] == 0
    assert eb["average_defect_area"] == 0.0
    assert eb["recent_inspections"] == []
    assert eb["recent_defect_counts"] == []

    # Insert 3 inspections with known defect areas:
    # Insp 1: 0 defects, NONE, INSUFFICIENT_DATA
    # Insp 2: 2 defects (areas 40, 120), MEDIUM, RANDOM_SCATTER
    # Insp 3: 3 defects (areas 200, 450, 700), HIGH, EDGE_RING
    spec = [
        ("clean.png", "NONE", "INSUFFICIENT_DATA", []),
        ("med.png", "MEDIUM", "RANDOM_SCATTER", [40, 120]),
        ("high.png", "HIGH", "EDGE_RING", [200, 450, 700]),
    ]
    with Session() as s:
        for idx, (fname, sev, pat, areas) in enumerate(spec, start=1):
            row = crud.create_inspection(
                s,
                upload_id=f"10000000-0000-0000-0000-{idx:012d}",
                original_filename=fname,
                image_width=512,
                image_height=512,
                image_mode="L",
                image_format="PNG",
                file_size=5000,
            )
            def_recs = [
                {
                    "region_no": j + 1,
                    "area": a,
                    "detected_area": a,
                    "width": 10,
                    "height": 10,
                    "centroid_x": 100.0,
                    "centroid_y": 100.0,
                    "bbox_x": 95,
                    "bbox_y": 95,
                    "bbox_width": 10,
                    "bbox_height": 10,
                    "perimeter": 35.0,
                    "equivalent_diameter": 10.0,
                    "length": 10.0,
                    "thickness": 10.0,
                    "angle_deg": 0.0,
                    "elongation": 1.0,
                    "polarity": "bright",
                    "mean_contrast": 60.0,
                    "peak_contrast": 70.0,
                    "contrast_to_noise": 10.0,
                    "touches_edge": False,
                    "detected_by": "self_reference",
                    "severity": "HIGH" if a >= 400 else "MEDIUM",
                    "why_flagged": "test",
                    "contour": [[95, 95], [105, 95], [105, 105]],
                }
                for j, a in enumerate(areas)
            ]
            crud.save_results(
                s,
                row.id,
                inspection_fields={
                    "defect_count": len(areas),
                    "total_defect_area": sum(areas),
                    "largest_defect_area": max(areas, default=0),
                    "average_defect_area": round(sum(areas) / len(areas), 2) if areas else 0.0,
                    "severity": sev,
                    "pattern_type": pat,
                    "processing_time": 0.25,
                },
                defect_records=def_recs,
                processing_run_record={
                    "preprocessing_method": "gray=True",
                    "threshold_method": "otsu",
                    "threshold_value": 25.0,
                    "morphology_method": "open:3,close:3",
                    "minimum_area": 12,
                    "config": {},
                    "stage_timings": {},
                    "processing_time": 0.25,
                },
            )

    stats = client.get("/dashboard/stats").json()
    assert stats["total_inspections"] == 3
    assert stats["total_defects"] == 5
    assert stats["high_severity_inspections"] == 1
    # Average defect area = (40 + 120 + 200 + 450 + 700) / 5 = 302.0
    assert stats["average_defect_area"] == 302.0
    assert len(stats["recent_inspections"]) == 3
    assert len(stats["recent_defect_counts"]) == 3

    sev_map = {x["severity"]: x["count"] for x in stats["severity_distribution"]}
    assert sev_map == {"NONE": 1, "LOW": 0, "MEDIUM": 1, "HIGH": 1}

    hist_map = {x["bin"]: x["count"] for x in stats["defect_area_histogram"]}
    assert hist_map == {"0-50": 1, "51-150": 1, "151-300": 1, "301-600": 1, "601+": 1}

    pat_map = {x["pattern_type"]: x["count"] for x in stats["pattern_distribution"]}
    assert pat_map == {"EDGE_RING": 1, "INSUFFICIENT_DATA": 1, "RANDOM_SCATTER": 1}


# ---------------------------------------------------------------------------
# 6. Failure Handling: Missing Image -> Failed Row; Commit Failure -> 503
# ---------------------------------------------------------------------------
def test_missing_image_marks_failed_and_db_commit_failure_returns_503(isolated_api_client):
    """Verify missing image marks inspection failed with sanitized msg; DB failure returns 503 with no stack trace."""
    client, _, Session, _, _ = isolated_api_client
    # A. Missing image file on disk
    img, _, _ = generate_layout_wafer("center_cluster", count=5, size=512, seed=8601, grid_pitch=32)
    up = client.post(
        "/inspection/upload",
        files={"test_image": ("missing_test.png", _png_bytes(img), "image/png")},
    )
    insp_id = up.json()["inspection_id"]
    upload_id = up.json()["upload_id"]

    # Remove the stored test image on disk before running
    for f in (settings.upload_path / upload_id).glob("test.*"):
        f.unlink()

    run_fail = client.post("/inspection/run", json={"inspection_id": insp_id})
    assert run_fail.status_code == 404
    assert "Traceback" not in run_fail.text

    with Session() as s:
        failed_row = crud.get_inspection(s, insp_id)
        assert failed_row is not None
        assert failed_row.status == "failed"
        assert failed_row.error_message is not None
        assert "missing" in failed_row.error_message.lower()
        assert "Traceback" not in failed_row.error_message

    # B. Injected DB commit failure -> HTTP 503 with code DATABASE_ERROR and no stack trace
    up2 = client.post(
        "/inspection/upload",
        files={"test_image": ("db_fail.png", _png_bytes(img), "image/png")},
    )
    insp_id2 = up2.json()["inspection_id"]

    with patch("app.db.crud.save_results", side_effect=OperationalError("stmt", {}, Exception("disk full"))):
        res_503 = client.post("/inspection/run", json={"inspection_id": insp_id2})
        assert res_503.status_code == 503
        body_503 = res_503.json()
        assert body_503["detail"]["code"] == "DATABASE_ERROR"
        assert "Traceback" not in res_503.text
        assert "disk full" not in res_503.text


# ---------------------------------------------------------------------------
# 7. Synthetic Demonstration Samples API
# ---------------------------------------------------------------------------
def test_samples_catalog_and_every_sample_runs_cleanly(isolated_api_client):
    """Verify GET /samples labels all items synthetic and every standard sample uploads and runs cleanly."""
    client, _, _, _, _ = isolated_api_client
    cat_res = client.get("/samples")
    assert cat_res.status_code == 200
    cat = cat_res.json()
    assert cat["disclaimer"] == SYNTHETIC_DISCLAIMER
    names = {s["name"] for s in cat["samples"]}
    required_names = {
        "clean",
        "bright_defect",
        "dark_defect",
        "multiple_defects",
        "scratch",
        "edge_ring",
        "center_cluster",
        "noise_only",
        "illumination_gradient",
    }
    assert required_names.issubset(names)
    assert all(s["disclaimer"] == SYNTHETIC_DISCLAIMER for s in cat["samples"])

    # Test every required 512px sample end to end (upload via POST /samples/{name} -> run -> DB report)
    for sample_name in sorted(required_names):
        up_res = client.post(f"/samples/{sample_name}")
        assert up_res.status_code == 200, f"Sample upload failed for {sample_name}: {up_res.text}"
        up_body = up_res.json()
        assert up_body["disclaimer"] == SYNTHETIC_DISCLAIMER
        insp_id = up_body["inspection_id"]

        run_res = client.post("/inspection/run", json={"inspection_id": insp_id})
        assert run_res.status_code == 200, f"Sample run failed for {sample_name}: {run_res.text}"
        assert run_res.json()["inspection_id"] == insp_id

    assert client.post("/samples/does_not_exist").status_code == 404


# ---------------------------------------------------------------------------
# 8. Default max_side=2048 Policy: Images <= 2048 Unchanged, 4096 Scaled to 0.5
# ---------------------------------------------------------------------------
def test_default_max_side_leaves_2048_unchanged_and_scales_4096():
    """Confirm default PreprocessingConfig.max_side == 2048 leaves <=2048 images untouched and scales 4096."""
    from app.cv.preprocessing import PreprocessingConfig, resize_preserving_aspect

    cfg = PreprocessingConfig()
    assert cfg.max_side == 2048

    # 512x512 and 2048x2048 must remain completely unchanged (scale_factor == 1.0, identical pixels)
    img_512 = np.arange(512 * 512, dtype=np.uint8).reshape(512, 512)
    out_512, scale_512 = resize_preserving_aspect(img_512, cfg.max_side)
    assert scale_512 == 1.0
    assert out_512.shape == (512, 512)
    assert np.array_equal(out_512, img_512)

    img_2048 = np.full((2048, 2048), 140, dtype=np.uint8)
    img_2048[100:120, 200:220] = 220
    out_2048, scale_2048 = resize_preserving_aspect(img_2048, cfg.max_side)
    assert scale_2048 == 1.0
    assert out_2048.shape == (2048, 2048)
    assert np.array_equal(out_2048, img_2048)

    # 4096x4096 downscales cleanly to 2048x2048 with scale_factor == 0.5
    img_4096 = np.full((4096, 4096), 140, dtype=np.uint8)
    out_4096, scale_4096 = resize_preserving_aspect(img_4096, cfg.max_side)
    assert scale_4096 == 0.5
    assert out_4096.shape == (2048, 2048)
