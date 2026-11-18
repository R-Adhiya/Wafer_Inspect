"""Generate synthetic wafer images for computer vision inspection testing.

Wraps the shared synthetic wafer generator in backend/app/utils/synthetic.py.
Supports generating either a single noisy wafer or a deterministic (reference, test)
pair with 4 injected synthetic defects (bright blob, dark blob, scratch, rim defect).
All data produced is purely synthetic for inspection validation.
"""

import json
import math
from pathlib import Path
import sys
from typing import Optional
from PIL import Image

# Ensure backend root is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.utils.synthetic import (  # noqa: E402
    create_standard_synthetic_pair,
    generate_layout_wafer,
    generate_synthetic_wafer,
)


def generate_noisy_wafer(output_path: str = "noisy_wafer.png", size: int = 512) -> None:
    """Generate and save a single synthetic noisy wafer image with illumination gradient."""
    image_uint8, _ = generate_synthetic_wafer(
        size=size,
        seed=42,
        wafer_radius_ratio=0.44,
        noise_std=18.0,
        has_gradient=True,
        gradient_strength=40.0,
        defects=None,
    )

    img = Image.fromarray(image_uint8, mode="L")
    img.save(output_path)
    print(
        f"Generated synthetic wafer at '{output_path}' "
        f"({size}x{size}, mean={image_uint8.mean():.2f}, std={image_uint8.std():.2f})"
    )


def generate_four_defect_pair(
    ref_path: str,
    test_path: str,
    size: int = 512,
    gt_path: Optional[str] = None,
) -> None:
    """Generate a synthetic golden reference and test wafer pair with 4 injected defects."""
    defects = [
        {"type": "bright_blob", "center": (165.0, 175.0), "radius": 8.0, "contrast": 60.0},
        {"type": "dark_blob", "center": (345.0, 185.0), "radius": 9.0, "contrast": -60.0},
        {"type": "scratch", "start": (190.0, 315.0), "end": (260.0, 345.0), "width": 3, "contrast": 65.0},
        {"type": "bright_blob", "center": (472.0, 256.0), "radius": 10.0, "contrast": 65.0},
    ]
    ref_img, test_img, gt = create_standard_synthetic_pair(size=size, seed=42, defects=defects)
    Image.fromarray(ref_img, mode="L").save(ref_path)
    Image.fromarray(test_img, mode="L").save(test_path)

    if gt_path:
        enriched_gt = []
        for idx, d in enumerate(gt, start=1):
            item = dict(d)
            item["gt_id"] = idx
            if d["type"] == "scratch":
                sx, sy = d["start"]
                ex, ey = d["end"]
                length = float(math.hypot(ex - sx, ey - sy))
                angle = float(math.degrees(math.atan2(ey - sy, ex - sx)) % 180.0)
                item["true_length_px"] = round(length, 2)
                item["true_thickness_px"] = float(d["width"])
                item["true_angle_deg"] = round(angle, 2)
                item["true_area_px"] = round(length * float(d["width"]), 1)
            else:
                r = float(d["radius"])
                item["true_diameter_px"] = round(2.0 * r, 2)
                item["true_area_px"] = round(math.pi * r * r, 1)
            enriched_gt.append(item)
        Path(gt_path).write_text(json.dumps(enriched_gt, indent=2), encoding="utf-8")

    print(
        f"Generated synthetic reference '{ref_path}' and 4-defect test wafer '{test_path}' "
        f"({len(gt)} ground-truth defects)."
    )


def generate_pattern_layout_image(layout: str, output_path: str, count: int = 8, seed: int = 8305) -> None:
    """Generate a synthetic patterned wafer with defects arranged in `layout`."""
    img, gt, label = generate_layout_wafer(
        defect_layout=layout,  # type: ignore[arg-type]
        count=count,
        size=512,
        seed=seed,
        grid_pitch=32,
        die_pattern="grid",
        noise_std=5.0,
        radius=7.0,
        contrast=65.0,
    )
    Image.fromarray(img, mode="L").save(output_path)
    print(f"Generated layout '{layout}' (GT={label}, n={len(gt)}) at '{output_path}'")


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--pair":
        gt_out = sys.argv[5] if len(sys.argv) >= 6 and sys.argv[4] == "--gt" else None
        generate_four_defect_pair(sys.argv[2], sys.argv[3], gt_path=gt_out)
    elif len(sys.argv) >= 4 and sys.argv[1] == "--layout":
        generate_pattern_layout_image(sys.argv[2], sys.argv[3])
    else:
        out_file = sys.argv[1] if len(sys.argv) > 1 else "noisy_wafer.png"
        generate_noisy_wafer(out_file)
