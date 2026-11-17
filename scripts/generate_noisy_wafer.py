"""Generate a synthetic noisy wafer image for computer vision testing.

Produces a 512x512 grayscale image simulating a silicon wafer disk with:
1. Substrate disk and dark wafer background.
2. Subtle die grid pattern.
3. Spatial illumination gradient (triggers adaptive CLAHE).
4. Zero-mean Gaussian noise (triggers denoising filter).
"""

import sys
import numpy as np
from PIL import Image


def generate_noisy_wafer(output_path: str = "noisy_wafer.png", size: int = 512) -> None:
    """Generate and save synthetic noisy wafer image with illumination gradient."""
    np.random.seed(42)

    # 1. Base circular wafer disk
    y, x = np.ogrid[:size, :size]
    center_y, center_x = size / 2.0, size / 2.0
    radius = size * 0.42
    wafer_mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2

    # Background is dark (~20), wafer substrate base is medium gray (~135)
    image = np.full((size, size), 20.0, dtype=np.float32)
    image[wafer_mask] = 135.0

    # 2. Add die grid pattern
    grid_spacing = 32
    grid_mask = ((x % grid_spacing < 2) | (y % grid_spacing < 2)) & wafer_mask
    image[grid_mask] = 110.0

    # 3. Add illumination gradient (bright at top-left, darker at bottom-right)
    gradient = ((size - x) + (size - y)) / (2.0 * size) * 80.0
    image[wafer_mask] += gradient[wafer_mask] - 40.0

    # 4. Add Gaussian noise (std ~ 18.0)
    noise = np.random.normal(0, 18.0, (size, size))
    image += noise

    # Clip intensity to valid 8-bit range
    image_uint8 = np.clip(image, 0, 255).astype(np.uint8)

    # Save to disk
    img = Image.fromarray(image_uint8, mode="L")
    img.save(output_path)
    print(
        f"Generated synthetic wafer at '{output_path}' "
        f"({size}x{size}, mean={image_uint8.mean():.2f}, std={image_uint8.std():.2f})"
    )


if __name__ == "__main__":
    out_file = sys.argv[1] if len(sys.argv) > 1 else "noisy_wafer.png"
    generate_noisy_wafer(out_file)
