"""Synthetic demonstration wafer sample catalog and generator service.

Every sample returned or uploaded by this module is a deterministic synthetic
demonstration image generated on demand via `backend/app/utils/synthetic.py`
and routed through the EXACT same upload validation, storage, and database
creation path as a normal user upload.
"""

from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple
from fastapi import HTTPException, UploadFile
import numpy as np
from PIL import Image
from sqlalchemy.orm import Session

from app.db.schemas import SampleItemResponse, SampleListResponse, UploadInspectionResponse
from app.services.image_service import process_and_store_upload
from app.utils.synthetic import (
    create_standard_synthetic_pair,
    generate_layout_wafer,
    generate_synthetic_wafer,
)

SYNTHETIC_DISCLAIMER = "Synthetic demonstration image"

SAMPLE_CATALOG: List[Dict[str, Any]] = [
    {
        "name": "clean",
        "title": "Clean Patterned Wafer",
        "description": "Defect-free periodic die-grid wafer with realistic sensor noise (expects 0 defects, NONE).",
        "category": "baseline",
        "has_reference": False,
        "image_size": 512,
    },
    {
        "name": "bright_defect",
        "title": "Single Bright Particle",
        "description": "Patterned wafer containing a single bright particle anomaly.",
        "category": "single_defect",
        "has_reference": False,
        "image_size": 512,
    },
    {
        "name": "dark_defect",
        "title": "Single Dark Pit Defect",
        "description": "Patterned wafer containing a single dark pit anomaly.",
        "category": "single_defect",
        "has_reference": False,
        "image_size": 512,
    },
    {
        "name": "multiple_defects",
        "title": "Multiple Mixed Defects (Golden Pair)",
        "description": "Test + golden reference pair with bright blob, dark blob, scratch across streets, and rim defect.",
        "category": "golden_reference",
        "has_reference": True,
        "image_size": 512,
    },
    {
        "name": "scratch",
        "title": "Diagonal Scratch Across Die Streets",
        "description": "Elongated diagonal scratch spanning multiple die streets (tests fragment merge & shape SCRATCH_LINE).",
        "category": "pattern",
        "has_reference": False,
        "image_size": 512,
    },
    {
        "name": "edge_ring",
        "title": "Edge Ring Pattern (8 Defects)",
        "description": "8 bright defects distributed around the outer 20% wafer edge ring.",
        "category": "pattern",
        "has_reference": False,
        "image_size": 512,
    },
    {
        "name": "center_cluster",
        "title": "Center Cluster Pattern (8 Defects)",
        "description": "8 defects concentrated within r <= 0.3 of the wafer center.",
        "category": "pattern",
        "has_reference": False,
        "image_size": 512,
    },
    {
        "name": "donut",
        "title": "Donut Ring Pattern (8 Defects)",
        "description": "8 defects distributed around the mid-radius annulus (0.4 <= r <= 0.7).",
        "category": "pattern",
        "has_reference": False,
        "image_size": 512,
    },
    {
        "name": "local_cluster",
        "title": "Off-Center Local Cluster (8 Defects)",
        "description": "8 defects tightly clustered in a single off-center wafer quadrant.",
        "category": "pattern",
        "has_reference": False,
        "image_size": 512,
    },
    {
        "name": "noise_only",
        "title": "High Sensor Noise Only",
        "description": "Clean periodic wafer with elevated Gaussian noise (tests noise ceiling false-positive immunity).",
        "category": "robustness",
        "has_reference": False,
        "image_size": 512,
    },
    {
        "name": "illumination_gradient",
        "title": "Strong Illumination Gradient Only",
        "description": "Clean periodic wafer with strong diagonal lighting gradient (expects 0 defects).",
        "category": "robustness",
        "has_reference": False,
        "image_size": 512,
    },
    {
        "name": "high_res_2048",
        "title": "High-Resolution 2048×2048 Edge Ring",
        "description": "2048×2048 patterned wafer with 8 edge-ring defects for full-resolution performance & live status testing.",
        "category": "performance",
        "has_reference": False,
        "image_size": 2048,
    },
]

_SAMPLE_BY_NAME: Dict[str, Dict[str, Any]] = {item["name"]: item for item in SAMPLE_CATALOG}


def list_samples() -> SampleListResponse:
    """Return catalog of available synthetic demonstration wafer samples."""
    return SampleListResponse(
        disclaimer=SYNTHETIC_DISCLAIMER,
        samples=[SampleItemResponse(**item, disclaimer=SYNTHETIC_DISCLAIMER) for item in SAMPLE_CATALOG],
    )


def _generate_sample_arrays(name: str, seed_offset: int = 0) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Deterministically generate (test_array, optional_ref_array) for a named sample."""
    seed = 4000 + seed_offset

    if name == "clean":
        img, _ = generate_synthetic_wafer(size=512, seed=seed + 1, grid_pitch=32, die_pattern="grid", noise_std=5.0)
        return img, None

    if name == "bright_defect":
        defects = [{"type": "bright_blob", "center": (220.0, 210.0), "radius": 9.0, "contrast": 70.0}]
        img, _ = generate_synthetic_wafer(size=512, seed=seed + 2, grid_pitch=32, die_pattern="grid", noise_std=5.0, defects=defects)
        return img, None

    if name == "dark_defect":
        defects = [{"type": "dark_blob", "center": (300.0, 290.0), "radius": 10.0, "contrast": -70.0}]
        img, _ = generate_synthetic_wafer(size=512, seed=seed + 3, grid_pitch=32, die_pattern="grid", noise_std=5.0, defects=defects)
        return img, None

    if name == "multiple_defects":
        defects = [
            {"type": "bright_blob", "center": (165.0, 175.0), "radius": 8.0, "contrast": 65.0},
            {"type": "dark_blob", "center": (345.0, 185.0), "radius": 9.0, "contrast": -65.0},
            {"type": "scratch", "start": (190.0, 315.0), "end": (260.0, 345.0), "width": 3, "contrast": 70.0},
            {"type": "bright_blob", "center": (472.0, 256.0), "radius": 10.0, "contrast": 65.0},
        ]
        ref_img, test_img, _ = create_standard_synthetic_pair(size=512, seed=seed + 4, defects=defects, grid_pitch=32, die_pattern="grid")
        return test_img, ref_img

    if name == "scratch":
        defects = [{"type": "scratch", "start": (175.0, 215.0), "end": (325.0, 295.0), "width": 4, "contrast": 75.0}]
        img, _ = generate_synthetic_wafer(size=512, seed=seed + 5, grid_pitch=32, die_pattern="grid", noise_std=5.0, defects=defects)
        return img, None

    if name == "edge_ring":
        img, _, _ = generate_layout_wafer("edge_ring", count=8, size=512, seed=seed + 6, grid_pitch=32, die_pattern="grid", radius=7.0, contrast=65.0)
        return img, None

    if name == "center_cluster":
        img, _, _ = generate_layout_wafer("center_cluster", count=8, size=512, seed=seed + 7, grid_pitch=32, die_pattern="grid", radius=7.0, contrast=65.0)
        return img, None

    if name == "donut":
        img, _, _ = generate_layout_wafer("donut", count=8, size=512, seed=seed + 8, grid_pitch=32, die_pattern="grid", radius=7.0, contrast=65.0)
        return img, None

    if name == "local_cluster":
        img, _, _ = generate_layout_wafer("local_cluster", count=8, size=512, seed=seed + 9, grid_pitch=32, die_pattern="grid", radius=7.0, contrast=65.0)
        return img, None

    if name == "noise_only":
        img, _ = generate_synthetic_wafer(size=512, seed=seed + 10, grid_pitch=32, die_pattern="grid", noise_std=10.0, defects=None)
        return img, None

    if name == "illumination_gradient":
        img, _ = generate_synthetic_wafer(
            size=512,
            seed=seed + 11,
            grid_pitch=32,
            die_pattern="grid",
            noise_std=5.0,
            has_gradient=True,
            gradient_strength=40.0,
            defects=None,
        )
        return img, None

    if name == "high_res_2048":
        img, _, _ = generate_layout_wafer(
            "edge_ring",
            count=8,
            size=2048,
            seed=seed + 12,
            grid_pitch=128,
            die_pattern="grid",
            noise_std=5.0,
            radius=24.0,
            contrast=65.0,
        )
        return img, None

    raise HTTPException(
        status_code=404,
        detail={"code": "SAMPLE_NOT_FOUND", "message": f"Unknown synthetic sample '{name}'."},
    )


def _array_to_upload_file(arr: np.ndarray, filename: str) -> UploadFile:
    buf = BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG")
    buf.seek(0)
    return UploadFile(file=buf, filename=filename)


async def create_sample_upload(
    name: str,
    db: Session,
    seed_offset: int = 0,
    filename_override: Optional[str] = None,
) -> UploadInspectionResponse:
    """Generate a deterministic synthetic sample and route it through the normal upload path."""
    clean_name = name.strip().lower()
    if clean_name not in _SAMPLE_BY_NAME:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "SAMPLE_NOT_FOUND",
                "message": f"Unknown sample '{name}'. Available: {sorted(_SAMPLE_BY_NAME.keys())}.",
            },
        )

    test_arr, ref_arr = _generate_sample_arrays(clean_name, seed_offset=seed_offset)
    test_fname = filename_override or f"synthetic_demo_{clean_name}.png"
    test_file = _array_to_upload_file(test_arr, filename=test_fname)
    ref_file = _array_to_upload_file(ref_arr, filename=f"synthetic_demo_{clean_name}_ref.png") if ref_arr is not None else None

    return await process_and_store_upload(
        test_image_file=test_file,
        reference_image_file=ref_file,
        db=db,
        disclaimer=SYNTHETIC_DISCLAIMER,
    )
