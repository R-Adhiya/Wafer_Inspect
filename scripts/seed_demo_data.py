"""Seed WaferInspect with >= 15 real synthetic demonstration inspections.

Runs every sample through the REAL upload validation, CV pipeline, and SQLAlchemy
persistence service layer (never inserting fake rows directly), so the Dashboard,
History table, charts, and Inspection Reports are backed by real images and metrics.
"""

import asyncio
from pathlib import Path
import sys
from typing import List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings  # noqa: E402
from app.db.database import SessionLocal, engine  # noqa: E402
from app.db.migrations import upgrade_database_to_head  # noqa: E402
from app.services.inspection_service import run_inspection_pipeline  # noqa: E402
from app.services.sample_service import create_sample_upload  # noqa: E402

# 16 varied real synthetic runs covering all layouts, severities (NONE/LOW/MEDIUM/HIGH), and modes
SEED_PLAN: List[Tuple[str, int, str]] = [
    ("clean", 0, "lotA_w01_clean.png"),
    ("noise_only", 1, "lotA_w02_noise_only.png"),
    ("illumination_gradient", 2, "lotA_w03_gradient_only.png"),
    ("bright_defect", 3, "lotA_w04_bright_particle.png"),
    ("dark_defect", 4, "lotA_w05_dark_pit.png"),
    ("scratch", 5, "lotB_w01_diagonal_scratch.png"),
    ("multiple_defects", 6, "lotB_w02_golden_4defects.png"),
    ("edge_ring", 7, "lotB_w03_edge_ring_a.png"),
    ("center_cluster", 8, "lotB_w04_center_cluster_a.png"),
    ("donut", 9, "lotC_w01_donut_ring_a.png"),
    ("local_cluster", 10, "lotC_w02_local_cluster_a.png"),
    ("edge_ring", 21, "lotC_w03_edge_ring_b.png"),
    ("center_cluster", 22, "lotC_w04_center_cluster_b.png"),
    ("donut", 23, "lotD_w01_donut_ring_b.png"),
    ("multiple_defects", 24, "lotD_w02_golden_mixed.png"),
    ("bright_defect", 25, "lotD_w03_single_particle.png"),
]


async def seed_all() -> None:
    upgrade_database_to_head(settings.DATABASE_URL)
    db = SessionLocal()
    try:
        print(f"Seeding {len(SEED_PLAN)} real synthetic demonstration inspections into {settings.DATABASE_URL}...")
        for idx, (sample_name, seed_offset, filename) in enumerate(SEED_PLAN, start=1):
            upload_res = await create_sample_upload(
                name=sample_name,
                db=db,
                seed_offset=seed_offset,
                filename_override=filename,
            )
            run_res = run_inspection_pipeline(
                inspection_id=upload_res.inspection_id,
                db=db,
            )
            sev = run_res.summary.severity if run_res.summary else "NONE"
            pat = run_res.pattern.type if run_res.pattern else "NONE"
            cnt = run_res.summary.defect_count if run_res.summary else 0
            print(
                f"  [{idx:02d}/{len(SEED_PLAN)}] id={run_res.inspection_id:<3} | {filename:<32} | "
                f"defects={cnt:<2} | sev={sev:<6} | pattern={pat:<17} | {run_res.processing_time_s:.3f}s"
            )
        print("Demo seed complete.")
    finally:
        db.close()
        engine.dispose()


if __name__ == "__main__":
    asyncio.run(seed_all())
