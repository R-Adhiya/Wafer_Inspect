"""API route handlers for CV experiments and noise filtering sweeps."""

from fastapi import APIRouter, HTTPException

from app.cv.noise_experiment import run_noise_sweep
from app.db.schemas import NoiseSweepRequest, NoiseSweepResponse

router = APIRouter(prefix="/experiments", tags=["Experiments"])


@router.post("/noise-sweep", response_model=NoiseSweepResponse)
def post_noise_sweep(
    payload: NoiseSweepRequest,
) -> NoiseSweepResponse:
    """Execute synchronous multi-seed noise sweep across Gaussian noise sigma levels.

    Enforces hard safety caps (<= 8 noise levels, <= 10 seeds, size <= 1024),
    returning 422 Unprocessable Entity if violated.
    """
    try:
        data = run_noise_sweep(
            layout=payload.layout,  # type: ignore[arg-type]
            path=payload.path,
            noise_levels=payload.noise_levels,
            seeds=payload.seeds,
            size=payload.size,
            tolerance_px=payload.tolerance_px,
            defect_count=payload.defect_count,
        )
        return NoiseSweepResponse.model_validate(data)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "INVALID_SWEEP_CONFIG", "message": str(exc)},
        )
