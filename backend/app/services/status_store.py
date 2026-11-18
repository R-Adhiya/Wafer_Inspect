"""In-memory inspection pipeline stage status store (single-process).

Tracks live pipeline stage progression keyed by `upload_id` so the frontend can
poll `GET /inspection/run/{upload_id}/status` while synchronous `POST /inspection/run`
executes inside FastAPI's worker threadpool.

LIMITATION:
This store is strictly in-memory and single-process (protected by a threading.Lock).
Status history is not persisted across server restarts and is not shared across
multi-worker Uvicorn/Gunicorn processes. Completed inspection results themselves
are persisted to `data/results/<upload_id>/run.json`.
"""

from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, List, Literal, Optional

PIPELINE_STAGES: List[str] = [
    "preparing_image",
    "preprocessing",
    "detecting",
    "segmenting",
    "filtering_noise",
    "measuring",
    "diagnosing_pattern",
    "generating_report",
]

RunLifecycleStatus = Literal["not_started", "running", "completed", "failed"]


class InspectionStatusStore:
    """Thread-safe single-process in-memory status tracker for inspection runs."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._records: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def start_run(self, upload_id: str) -> None:
        """Initialize or reset run status for `upload_id` at the first stage."""
        with self._lock:
            self._records[upload_id] = {
                "upload_id": upload_id,
                "status": "running",
                "current_stage": PIPELINE_STAGES[0],
                "completed_stages": [],
                "stage_history": [PIPELINE_STAGES[0]],
                "stages": list(PIPELINE_STAGES),
                "error": None,
                "updated_at": self._now_iso(),
            }

    def set_stage(self, upload_id: str, stage: str) -> None:
        """Transition `upload_id` to `stage`, marking all prior stages completed."""
        with self._lock:
            rec = self._records.get(upload_id)
            if rec is None:
                self._records[upload_id] = {
                    "upload_id": upload_id,
                    "status": "running",
                    "current_stage": stage,
                    "completed_stages": [],
                    "stage_history": [stage],
                    "stages": list(PIPELINE_STAGES),
                    "error": None,
                    "updated_at": self._now_iso(),
                }
                return

            prev = rec.get("current_stage")
            if prev and prev != stage and prev in PIPELINE_STAGES and prev not in rec["completed_stages"]:
                rec["completed_stages"].append(prev)

            # Also mark any earlier stages in canonical order as completed
            if stage in PIPELINE_STAGES:
                target_idx = PIPELINE_STAGES.index(stage)
                for s in PIPELINE_STAGES[:target_idx]:
                    if s not in rec["completed_stages"]:
                        rec["completed_stages"].append(s)

            rec["status"] = "running"
            rec["current_stage"] = stage
            if not rec["stage_history"] or rec["stage_history"][-1] != stage:
                rec["stage_history"].append(stage)
            rec["updated_at"] = self._now_iso()

    def complete_run(self, upload_id: str) -> None:
        """Mark `upload_id` as completed across all stages."""
        with self._lock:
            rec = self._records.get(upload_id)
            if rec is None:
                rec = {
                    "upload_id": upload_id,
                    "stage_history": list(PIPELINE_STAGES),
                    "stages": list(PIPELINE_STAGES),
                }
                self._records[upload_id] = rec
            rec["status"] = "completed"
            rec["current_stage"] = "completed"
            rec["completed_stages"] = list(PIPELINE_STAGES)
            if not rec["stage_history"] or rec["stage_history"][-1] != "completed":
                rec["stage_history"].append("completed")
            rec["error"] = None
            rec["updated_at"] = self._now_iso()

    def fail_run(self, upload_id: str, error_message: str) -> None:
        """Mark `upload_id` as failed with an error message."""
        with self._lock:
            rec = self._records.get(upload_id)
            if rec is None:
                rec = {
                    "upload_id": upload_id,
                    "completed_stages": [],
                    "stage_history": [],
                    "stages": list(PIPELINE_STAGES),
                }
                self._records[upload_id] = rec
            rec["status"] = "failed"
            rec["current_stage"] = "failed"
            if not rec["stage_history"] or rec["stage_history"][-1] != "failed":
                rec["stage_history"].append("failed")
            rec["error"] = error_message
            rec["updated_at"] = self._now_iso()

    def get_status(self, upload_id: str) -> Optional[Dict[str, Any]]:
        """Return a shallow copy of the current status record, or None if not tracked."""
        with self._lock:
            rec = self._records.get(upload_id)
            if rec is None:
                return None
            return {
                "upload_id": rec["upload_id"],
                "status": rec["status"],
                "current_stage": rec["current_stage"],
                "completed_stages": list(rec["completed_stages"]),
                "stage_history": list(rec["stage_history"]),
                "stages": list(rec["stages"]),
                "error": rec["error"],
                "updated_at": rec["updated_at"],
            }


status_store = InspectionStatusStore()
