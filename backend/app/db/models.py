"""SQLAlchemy 2.0 ORM models for inspections, defects, and processing runs.

Uses strictly database-portable SQLAlchemy column types (Integer, String, Float,
Boolean, Text, DateTime(timezone=True), JSON) so the schema runs identically on
SQLite and PostgreSQL without SQLite-only constructs.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Inspection(Base):
    """Master inspection record storing upload metadata, summary metrics, and pattern diagnosis."""

    __tablename__ = "inspections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    upload_id: Mapped[str] = mapped_column(String(36), unique=True, index=True, nullable=False)
    status: Mapped[str] = mapped_column(String(32), index=True, nullable=False, default="pending")
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Upload metadata
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    image_width: Mapped[int] = mapped_column(Integer, nullable=False)
    image_height: Mapped[int] = mapped_column(Integer, nullable=False)
    image_mode: Mapped[str] = mapped_column(String(32), nullable=False)
    image_format: Mapped[str] = mapped_column(String(32), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    has_reference: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    reference_filename: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Detection & summary metrics
    detection_path: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    detection_label: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    defect_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_defect_area: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    largest_defect_area: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    average_defect_area: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    severity: Mapped[Optional[str]] = mapped_column(String(16), index=True, nullable=True)
    severity_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    calibration_status: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    pixel_size_um: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Wafer-level spatial pattern diagnosis
    pattern_type: Mapped[Optional[str]] = mapped_column(String(64), index=True, nullable=True)
    pattern_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pattern_p_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pattern_explanation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    pattern_metrics: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    pattern_supporting_ids: Mapped[Optional[List[int]]] = mapped_column(JSON, nullable=True)
    pattern_warnings: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)

    # Pipeline diagnostics & reproducibility metadata
    stage_counts: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSON, nullable=True)
    rejected_regions: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSON, nullable=True)
    warnings: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)
    original_stats: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    preprocessed_stats: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    step_log: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSON, nullable=True)
    detection_details: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)

    # Model & timing
    model_name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    model_version: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    detection_method: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    processing_time: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), index=True, nullable=False, default=_utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utc_now, onupdate=_utc_now
    )

    defects: Mapped[List["Defect"]] = relationship(
        "Defect",
        back_populates="inspection",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Defect.region_no",
    )
    processing_runs: Mapped[List["ProcessingRun"]] = relationship(
        "ProcessingRun",
        back_populates="inspection",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="ProcessingRun.created_at",
    )


Index("ix_inspections_status_created_at", Inspection.status, Inspection.created_at)
Index("ix_inspections_filename", Inspection.original_filename)


class Defect(Base):
    """Per-defect geometric, photometric, severity, and contour record linked to an inspection."""

    __tablename__ = "defects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    inspection_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("inspections.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    region_no: Mapped[int] = mapped_column(Integer, nullable=False)
    area: Mapped[int] = mapped_column(Integer, nullable=False)
    detected_area: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    centroid_x: Mapped[float] = mapped_column(Float, nullable=False)
    centroid_y: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_x: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_y: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_width: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_height: Mapped[int] = mapped_column(Integer, nullable=False)
    perimeter: Mapped[float] = mapped_column(Float, nullable=False)
    equivalent_diameter: Mapped[float] = mapped_column(Float, nullable=False)
    length: Mapped[float] = mapped_column(Float, nullable=False)
    thickness: Mapped[float] = mapped_column(Float, nullable=False)
    angle_deg: Mapped[float] = mapped_column(Float, nullable=False)
    elongation: Mapped[float] = mapped_column(Float, nullable=False)
    polarity: Mapped[str] = mapped_column(String(16), nullable=False)
    mean_contrast: Mapped[float] = mapped_column(Float, nullable=False)
    peak_contrast: Mapped[float] = mapped_column(Float, nullable=False)
    contrast_to_noise: Mapped[float] = mapped_column(Float, nullable=False)
    touches_edge: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    detected_by: Mapped[str] = mapped_column(String(32), nullable=False)
    severity: Mapped[str] = mapped_column(String(16), index=True, nullable=False)

    # Nullable ML fields populated in future ML phase
    defect_type: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    why_flagged: Mapped[str] = mapped_column(Text, nullable=False)
    contour: Mapped[Optional[List[List[int]]]] = mapped_column(JSON, nullable=True)
    reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    area_um2: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    length_um: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    thickness_um: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    inspection: Mapped["Inspection"] = relationship("Inspection", back_populates="defects")


Index("ix_defects_inspection_region", Defect.inspection_id, Defect.region_no)


class ProcessingRun(Base):
    """Execution parameters, configuration snapshot, and stage timings for an inspection run."""

    __tablename__ = "processing_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    inspection_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("inspections.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    preprocessing_method: Mapped[str] = mapped_column(String(128), nullable=False)
    threshold_method: Mapped[str] = mapped_column(String(64), nullable=False)
    threshold_value: Mapped[float] = mapped_column(Float, nullable=False)
    morphology_method: Mapped[str] = mapped_column(String(128), nullable=False)
    minimum_area: Mapped[int] = mapped_column(Integer, nullable=False)
    config: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    stage_timings: Mapped[Dict[str, float]] = mapped_column(JSON, nullable=False)
    processing_time: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utc_now
    )

    inspection: Mapped["Inspection"] = relationship("Inspection", back_populates="processing_runs")
