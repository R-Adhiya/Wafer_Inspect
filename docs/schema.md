# WaferInspect Relational Database Schema

WaferInspect uses **SQLAlchemy 2.0** and **Alembic** migrations with strictly database-portable column types (`Integer`, `String`, `Float`, `Boolean`, `Text`, `DateTime(timezone=True)`, `JSON`). It runs on SQLite out of the box (`data/waferinspect.db`) and switches cleanly to PostgreSQL via `DATABASE_URL`.

## Entity-Relationship Diagram

```mermaid
erDiagram
    INSPECTIONS ||--o{ DEFECTS : "has many (ON DELETE CASCADE)"
    INSPECTIONS ||--o{ PROCESSING_RUNS : "has many (ON DELETE CASCADE)"

    INSPECTIONS {
        int id PK
        string upload_id UK "UUID4, indexed"
        string status "pending | running | completed | failed"
        text error_message "sanitized error, nullable"
        string original_filename "indexed"
        int image_width
        int image_height
        string image_mode
        string image_format
        int file_size
        boolean has_reference
        string reference_filename
        string detection_path
        string detection_label
        int defect_count
        int total_defect_area
        int largest_defect_area
        float average_defect_area
        string severity "NONE | LOW | MEDIUM | HIGH (indexed)"
        text severity_reason
        string calibration_status
        string pattern_type "indexed"
        float pattern_confidence
        float pattern_p_value
        text pattern_explanation
        json pattern_metrics
        json stage_counts
        json rejected_regions
        json warnings
        string model_name
        string model_version
        string detection_method
        float processing_time
        datetime created_at "timezone-aware, indexed"
        datetime updated_at "timezone-aware"
    }

    DEFECTS {
        int id PK
        int inspection_id FK "indexed, ON DELETE CASCADE"
        int region_no "1..N per inspection"
        int area "refined area in original px"
        int detected_area "pre-refinement area in original px"
        int width
        int height
        float centroid_x
        float centroid_y
        int bbox_x
        int bbox_y
        int bbox_width
        int bbox_height
        float perimeter
        float equivalent_diameter
        float length
        float thickness
        float angle_deg
        float elongation
        string polarity "bright | dark"
        float mean_contrast
        float peak_contrast
        float contrast_to_noise
        boolean touches_edge
        string detected_by "golden | self_reference | background"
        string severity "LOW | MEDIUM | HIGH (indexed)"
        string defect_type "nullable, reserved for ML phase"
        float confidence "nullable, reserved for ML phase"
        text why_flagged
        json contour "simplified polygon <= 200 pts"
    }

    PROCESSING_RUNS {
        int id PK
        int inspection_id FK "indexed, ON DELETE CASCADE"
        string preprocessing_method
        string threshold_method
        float threshold_value
        string morphology_method
        int minimum_area
        json config "full pipeline config snapshot"
        json stage_timings "per-stage ms timings"
        float processing_time
        datetime created_at "timezone-aware"
    }
```

## Indexes & Constraints

- **Foreign Keys**: `defects.inspection_id` and `processing_runs.inspection_id` reference `inspections.id` with `ON DELETE CASCADE`. SQLite enforces `PRAGMA foreign_keys=ON` and `PRAGMA journal_mode=WAL` on every connection.
- **Indexes**:
  - `inspections.upload_id` (unique)
  - `inspections.status`, `inspections.severity`, `inspections.pattern_type`, `inspections.created_at`, `inspections.original_filename`, composite `(status, created_at)`
  - `defects.inspection_id`, `defects.severity`, composite `(inspection_id, region_no)`
  - `processing_runs.inspection_id`

## Running Migrations

```bash
# Apply all Alembic migrations to the configured DATABASE_URL
alembic -c backend/alembic.ini upgrade head
```

Development startup (`backend/app/main.py`) invokes the exact same Alembic migration path automatically.
