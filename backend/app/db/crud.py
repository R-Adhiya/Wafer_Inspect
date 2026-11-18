"""Pure SQLAlchemy 2.0 repository / CRUD functions for WaferInspect.

Contains database read/write/aggregate operations only (no CV or HTTP business logic).
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple
from sqlalchemy import asc, case, desc, func, select
from sqlalchemy.orm import Session, selectinload

from app.db.models import Defect, Inspection, ProcessingRun

SEVERITY_ORDER = ("NONE", "LOW", "MEDIUM", "HIGH")

AREA_HISTOGRAM_BINS: List[Tuple[str, int, Optional[int]]] = [
    ("0-50", 0, 50),
    ("51-150", 51, 150),
    ("151-300", 151, 300),
    ("301-600", 301, 600),
    ("601+", 601, None),
]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def create_inspection(
    db: Session,
    *,
    upload_id: str,
    original_filename: str,
    image_width: int,
    image_height: int,
    image_mode: str,
    image_format: str,
    file_size: int,
    has_reference: bool = False,
    reference_filename: Optional[str] = None,
    status: str = "pending",
) -> Inspection:
    """Insert a new Inspection row (default status 'pending') and commit."""
    now = _utc_now()
    row = Inspection(
        upload_id=upload_id,
        status=status,
        original_filename=original_filename,
        image_width=int(image_width),
        image_height=int(image_height),
        image_mode=image_mode,
        image_format=image_format,
        file_size=int(file_size),
        has_reference=bool(has_reference),
        reference_filename=reference_filename,
        created_at=now,
        updated_at=now,
    )
    try:
        db.add(row)
        db.commit()
        db.refresh(row)
        return row
    except Exception:
        db.rollback()
        raise


def set_status(
    db: Session,
    inspection_id: int,
    status: str,
    error_message: Optional[str] = None,
) -> Optional[Inspection]:
    """Update inspection lifecycle status and optional sanitized error message."""
    row = db.get(Inspection, inspection_id)
    if row is None:
        return None
    row.status = status
    row.error_message = error_message
    row.updated_at = _utc_now()
    try:
        db.commit()
        db.refresh(row)
        return row
    except Exception:
        db.rollback()
        raise


def save_results(
    db: Session,
    inspection_id: int,
    *,
    inspection_fields: Dict[str, Any],
    defect_records: Sequence[Dict[str, Any]],
    processing_run_record: Dict[str, Any],
) -> Inspection:
    """Atomically update Inspection fields, replace Defects, and insert ProcessingRun in ONE transaction."""
    try:
        row = db.get(Inspection, inspection_id)
        if row is None:
            raise ValueError(f"Inspection id={inspection_id} does not exist.")

        for key, value in inspection_fields.items():
            setattr(row, key, value)
        row.status = "completed"
        row.error_message = None
        row.updated_at = _utc_now()

        # Clear any prior defects for idempotent re-runs on the same inspection
        db.query(Defect).filter(Defect.inspection_id == inspection_id).delete(synchronize_session="fetch")

        for rec in defect_records:
            defect_obj = Defect(inspection_id=inspection_id, **rec)
            db.add(defect_obj)

        run_obj = ProcessingRun(inspection_id=inspection_id, **processing_run_record)
        db.add(run_obj)

        db.commit()
        db.refresh(row)
        return get_inspection(db, inspection_id) or row
    except Exception:
        db.rollback()
        raise


def get_inspection(db: Session, inspection_id: int) -> Optional[Inspection]:
    """Fetch a single Inspection by primary key with defects and processing runs eagerly loaded."""
    stmt = (
        select(Inspection)
        .options(selectinload(Inspection.defects), selectinload(Inspection.processing_runs))
        .where(Inspection.id == inspection_id)
    )
    return db.execute(stmt).scalar_one_or_none()


def get_inspection_by_upload_id(db: Session, upload_id: str) -> Optional[Inspection]:
    """Fetch a single Inspection by UUID upload_id with defects and processing runs eagerly loaded."""
    stmt = (
        select(Inspection)
        .options(selectinload(Inspection.defects), selectinload(Inspection.processing_runs))
        .where(Inspection.upload_id == upload_id)
    )
    return db.execute(stmt).scalar_one_or_none()


def list_defects_for_inspection(db: Session, inspection_id: int) -> List[Defect]:
    """Return ordered Defect rows for a given inspection_id."""
    stmt = (
        select(Defect)
        .where(Defect.inspection_id == inspection_id)
        .order_by(Defect.region_no.asc(), Defect.id.asc())
    )
    return list(db.execute(stmt).scalars().all())


def list_history(
    db: Session,
    *,
    search: Optional[str] = None,
    severity: Optional[str] = None,
    sort: str = "created_at",
    order: str = "desc",
    page: int = 1,
    page_size: int = 20,
) -> Tuple[List[Inspection], int]:
    """Query paginated inspection history with filename search, severity filter, and sorting."""
    filters = []
    if search and search.strip():
        pattern = f"%{search.strip()}%"
        filters.append(Inspection.original_filename.ilike(pattern))

    if severity and severity.strip():
        sev_clean = severity.strip().upper()
        if sev_clean in SEVERITY_ORDER:
            filters.append(Inspection.severity == sev_clean)

    count_stmt = select(func.count(Inspection.id))
    if filters:
        count_stmt = count_stmt.where(*filters)
    total = int(db.execute(count_stmt).scalar_one() or 0)

    severity_rank_expr = case(
        (Inspection.severity == "NONE", 0),
        (Inspection.severity == "LOW", 1),
        (Inspection.severity == "MEDIUM", 2),
        (Inspection.severity == "HIGH", 3),
        else_=-1,
    )

    sort_map = {
        "created_at": Inspection.created_at,
        "defect_count": Inspection.defect_count,
        "severity": severity_rank_expr,
        "processing_time": func.coalesce(Inspection.processing_time, 0.0),
    }
    sort_col = sort_map.get(sort, Inspection.created_at)
    direction = asc if order.strip().lower() == "asc" else desc

    safe_page = max(1, int(page))
    safe_size = max(1, min(100, int(page_size)))
    offset = (safe_page - 1) * safe_size

    data_stmt = (
        select(Inspection)
        .where(*filters)
        .order_by(direction(sort_col), direction(Inspection.id))
        .offset(offset)
        .limit(safe_size)
    )
    items = list(db.execute(data_stmt).scalars().all())
    return items, total


def delete_inspection(db: Session, inspection_id: int) -> Optional[str]:
    """Delete an Inspection (cascading defects & processing_runs) and return its upload_id, or None."""
    row = db.get(Inspection, inspection_id)
    if row is None:
        return None
    upload_id = row.upload_id
    try:
        db.delete(row)
        db.commit()
        return upload_id
    except Exception:
        db.rollback()
        raise


def get_dashboard_stats(db: Session) -> Dict[str, Any]:
    """Compute dashboard summary KPIs and chart series strictly via SQL aggregates (empty-safe)."""
    # 1. Core KPI aggregates
    kpi_stmt = select(
        func.count(Inspection.id),
        func.coalesce(func.sum(Inspection.defect_count), 0),
        func.coalesce(
            func.sum(case((Inspection.severity == "HIGH", 1), else_=0)),
            0,
        ),
    )
    total_inspections, total_defects, high_severity_inspections = db.execute(kpi_stmt).one()

    avg_area_stmt = select(func.coalesce(func.avg(Defect.area), 0.0))
    avg_defect_area = float(db.execute(avg_area_stmt).scalar_one() or 0.0)

    # 2. 5 most recent inspections
    recent_5_stmt = (
        select(Inspection)
        .order_by(Inspection.created_at.desc(), Inspection.id.desc())
        .limit(5)
    )
    recent_5 = list(db.execute(recent_5_stmt).scalars().all())

    # 3. Defect counts for the last 20 inspections (oldest -> newest for chart readability)
    recent_20_stmt = (
        select(
            Inspection.id,
            Inspection.original_filename,
            Inspection.defect_count,
            Inspection.severity,
            Inspection.created_at,
        )
        .order_by(Inspection.created_at.desc(), Inspection.id.desc())
        .limit(20)
    )
    recent_20_rows = list(reversed(db.execute(recent_20_stmt).all()))
    recent_defect_counts = [
        {
            "inspection_id": int(r.id),
            "filename": r.original_filename,
            "defect_count": int(r.defect_count),
            "severity": r.severity or "NONE",
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in recent_20_rows
    ]

    # 4. Severity distribution via GROUP BY
    sev_stmt = (
        select(Inspection.severity, func.count(Inspection.id))
        .where(Inspection.severity.is_not(None))
        .group_by(Inspection.severity)
    )
    sev_map = {str(sev): int(cnt) for sev, cnt in db.execute(sev_stmt).all() if sev is not None}
    severity_distribution = [
        {"severity": sev, "count": sev_map.get(sev, 0)} for sev in SEVERITY_ORDER
    ]

    # 5. Pattern-type distribution via GROUP BY
    pat_stmt = (
        select(Inspection.pattern_type, func.count(Inspection.id))
        .where(Inspection.pattern_type.is_not(None))
        .group_by(Inspection.pattern_type)
        .order_by(func.count(Inspection.id).desc(), Inspection.pattern_type.asc())
    )
    pattern_distribution = [
        {"pattern_type": str(pat), "count": int(cnt)}
        for pat, cnt in db.execute(pat_stmt).all()
        if pat is not None
    ]

    # 6. Defect-area histogram bins via SQL CASE aggregation
    bucket_expr = case(
        (Defect.area <= 50, "0-50"),
        (Defect.area <= 150, "51-150"),
        (Defect.area <= 300, "151-300"),
        (Defect.area <= 600, "301-600"),
        else_="601+",
    )
    hist_stmt = select(bucket_expr.label("bucket"), func.count(Defect.id)).group_by(bucket_expr)
    hist_map = {str(b): int(cnt) for b, cnt in db.execute(hist_stmt).all()}
    defect_area_histogram = [
        {"bin": label, "count": hist_map.get(label, 0)}
        for label, _, _ in AREA_HISTOGRAM_BINS
    ]

    return {
        "total_inspections": int(total_inspections or 0),
        "total_defects": int(total_defects or 0),
        "high_severity_inspections": int(high_severity_inspections or 0),
        "average_defect_area": round(avg_defect_area, 2),
        "recent_inspections": recent_5,
        "recent_defect_counts": recent_defect_counts,
        "severity_distribution": severity_distribution,
        "defect_area_histogram": defect_area_histogram,
        "pattern_distribution": pattern_distribution,
    }
