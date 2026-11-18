"""Synthetic wafer image generator for testing and validation.

Generates deterministic, synthetic grayscale images simulating silicon wafer
imagery with circular geometry, die patterns, sensor noise, illumination gradients,
and ground-truth synthetic defects (bright blobs, dark blobs, scratches).

All data produced by this module is purely synthetic and intended solely
for computer vision inspection and algorithm accuracy validation.
"""

from typing import Any, Dict, List, Literal, Optional, Tuple
import math
import cv2
import numpy as np

DefectLayout = Literal[
    "random",
    "edge_ring",
    "center_cluster",
    "donut",
    "scratch_line",
    "local_cluster",
]

LAYOUT_TO_PATTERN_LABEL: Dict[str, str] = {
    "random": "RANDOM_SCATTER",
    "edge_ring": "EDGE_RING",
    "center_cluster": "CENTER_CLUSTER",
    "donut": "DONUT",
    "scratch_line": "SCRATCH_LINE",
    "local_cluster": "LOCAL_CLUSTER",
}


def generate_layout_centroids(
    layout: DefectLayout,
    count: int = 8,
    center: Tuple[float, float] = (256.0, 256.0),
    radius: float = 225.28,
    seed: int = 42,
    min_separation_px: float = 0.0,
) -> Tuple[List[Tuple[float, float]], str]:
    """Generate deterministic seeded defect centroids for a named wafer spatial layout.

    Args:
        layout: One of "random", "edge_ring", "center_cluster", "donut",
                "scratch_line", or "local_cluster".
        count: Number of defect centroids to generate.
        center: (cx, cy) wafer center coordinates in pixels.
        radius: Wafer radius in pixels.
        seed: Deterministic NumPy RNG seed.
        min_separation_px: Optional minimum Euclidean separation between centroids
                           (useful when injecting finite-radius blobs into images so
                           distinct defects do not overlap or merge).

    Returns:
        Tuple[List[Tuple[float, float]], str]: (centroids, ground_truth_pattern_label)
    """
    if layout not in LAYOUT_TO_PATTERN_LABEL:
        raise ValueError(f"Unsupported defect_layout '{layout}'. Allowed: {list(LAYOUT_TO_PATTERN_LABEL.keys())}")

    rng = np.random.default_rng(seed)
    cx, cy = float(center[0]), float(center[1])
    R = float(radius)
    pts: List[Tuple[float, float]] = []

    def _far_enough(px: float, py: float, sep: float) -> bool:
        if sep <= 0.0:
            return True
        return all(math.hypot(px - ox, py - oy) >= sep for ox, oy in pts)

    if count <= 0:
        return [], LAYOUT_TO_PATTERN_LABEL[layout]

    if layout == "edge_ring":
        # Place evenly around outer band [0.83, 0.92] * R with seeded angular/radial jitter
        base_angle = float(rng.uniform(0.0, 2.0 * math.pi))
        step = (2.0 * math.pi) / float(count)
        for i in range(count):
            for attempt in range(40):
                jitter_scale = 0.28 if attempt < 20 else 0.05
                theta = base_angle + i * step + float(rng.uniform(-jitter_scale, jitter_scale) * step)
                r = float(rng.uniform(0.84, 0.91) * R)
                px = cx + r * math.cos(theta)
                py = cy + r * math.sin(theta)
                if _far_enough(px, py, min_separation_px):
                    pts.append((round(px, 2), round(py, 2)))
                    break
            else:
                theta = base_angle + i * step
                r = 0.875 * R
                pts.append((round(cx + r * math.cos(theta), 2), round(cy + r * math.sin(theta), 2)))

    elif layout == "donut":
        # Place evenly around mid annulus [0.44, 0.66] * R with seeded jitter
        base_angle = float(rng.uniform(0.0, 2.0 * math.pi))
        step = (2.0 * math.pi) / float(count)
        for i in range(count):
            for attempt in range(40):
                jitter_scale = 0.28 if attempt < 20 else 0.05
                theta = base_angle + i * step + float(rng.uniform(-jitter_scale, jitter_scale) * step)
                r = float(rng.uniform(0.45, 0.65) * R)
                px = cx + r * math.cos(theta)
                py = cy + r * math.sin(theta)
                if _far_enough(px, py, min_separation_px):
                    pts.append((round(px, 2), round(py, 2)))
                    break
            else:
                theta = base_angle + i * step
                r = 0.55 * R
                pts.append((round(cx + r * math.cos(theta), 2), round(cy + r * math.sin(theta), 2)))

    elif layout == "center_cluster":
        # Place within inner disk r <= 0.27 * R (well inside 0.30 * R)
        max_r = 0.265 * R
        base_angle = float(rng.uniform(0.0, 2.0 * math.pi))
        if min_separation_px > 0.0 and count > 1:
            # Structured jittered placement inside center disk so finite blobs never merge
            # Put 1 near center (if count >= 4) and remainder on an inner ring at ~0.22 * R
            ring_start = 0
            if count >= 5:
                r0 = float(rng.uniform(0.0, 0.05) * R)
                t0 = float(rng.uniform(0.0, 2.0 * math.pi))
                pts.append((round(cx + r0 * math.cos(t0), 2), round(cy + r0 * math.sin(t0), 2)))
                ring_start = 1
            ring_n = count - ring_start
            step = (2.0 * math.pi) / float(max(ring_n, 1))
            for i in range(ring_n):
                for attempt in range(40):
                    theta = base_angle + i * step + float(rng.uniform(-0.18, 0.18) * step)
                    r = float(rng.uniform(0.18, 0.265) * R)
                    px = cx + r * math.cos(theta)
                    py = cy + r * math.sin(theta)
                    if _far_enough(px, py, min_separation_px):
                        pts.append((round(px, 2), round(py, 2)))
                        break
                else:
                    theta = base_angle + i * step
                    r = 0.23 * R
                    pts.append((round(cx + r * math.cos(theta), 2), round(cy + r * math.sin(theta), 2)))
        else:
            for _ in range(count):
                r = max_r * math.sqrt(float(rng.uniform(0.0, 1.0)))
                theta = float(rng.uniform(0.0, 2.0 * math.pi))
                pts.append((round(cx + r * math.cos(theta), 2), round(cy + r * math.sin(theta), 2)))

    elif layout == "scratch_line":
        # Collinear points along a seeded chord spanning 55%-72% of wafer diameter (>= 40%)
        angle = float(rng.uniform(0.0, math.pi))
        offset_perp = float(rng.uniform(-0.25, 0.25) * R)
        ux, uy = math.cos(angle), math.sin(angle)
        nx, ny = -uy, ux
        half_span = float(rng.uniform(0.58, 0.72) * R)  # total span = 1.16..1.44 * R = 58%..72% of diameter
        max_chord = math.sqrt(max(0.0, (0.90 * R) ** 2 - offset_perp ** 2))
        half_span = min(half_span, max_chord)
        ts = np.linspace(-half_span, half_span, count) if count > 1 else np.array([0.0])
        for t in ts:
            t_j = float(t) + float(rng.uniform(-2.0, 2.0))
            p_j = float(rng.uniform(-2.0, 2.0))
            px = cx + t_j * ux + (offset_perp + p_j) * nx
            py = cy + t_j * uy + (offset_perp + p_j) * ny
            pts.append((round(px, 2), round(py, 2)))

    elif layout == "local_cluster":
        # Tight cluster centered off-center (r_c in [0.46, 0.62] * R) so it is NOT CENTER_CLUSTER
        c_theta = float(rng.uniform(0.0, 2.0 * math.pi))
        c_r = float(rng.uniform(0.46, 0.62) * R)
        cl_x = cx + c_r * math.cos(c_theta)
        cl_y = cy + c_r * math.sin(c_theta)
        # Cluster radius scales cleanly so R_clark_evans stays << 0.6 while respecting min_separation_px
        cluster_r = max(0.14 * R, min_separation_px * max(1.1, math.sqrt(count) * 0.55))
        cluster_r = min(cluster_r, 0.24 * R)
        base_angle = float(rng.uniform(0.0, 2.0 * math.pi))

        if min_separation_px > 0.0 and count > 1:
            # Place 1 near cluster center + remaining on compact ring(s) around cluster center
            pts.append((round(cl_x + float(rng.uniform(-3.0, 3.0)), 2), round(cl_y + float(rng.uniform(-3.0, 3.0)), 2)))
            rem = count - 1
            step = (2.0 * math.pi) / float(rem)
            ring_r = max(min_separation_px * 1.15, 0.12 * R)
            for i in range(rem):
                for attempt in range(40):
                    theta = base_angle + i * step + float(rng.uniform(-0.18, 0.18) * step)
                    r_loc = float(rng.uniform(0.88, 1.12) * ring_r)
                    px = cl_x + r_loc * math.cos(theta)
                    py = cl_y + r_loc * math.sin(theta)
                    if math.hypot(px - cx, py - cy) <= 0.90 * R and _far_enough(px, py, min_separation_px):
                        pts.append((round(px, 2), round(py, 2)))
                        break
                else:
                    theta = base_angle + i * step
                    px = cl_x + ring_r * math.cos(theta)
                    py = cl_y + ring_r * math.sin(theta)
                    pts.append((round(px, 2), round(py, 2)))
        else:
            for _ in range(count):
                r_loc = cluster_r * math.sqrt(float(rng.uniform(0.0, 1.0)))
                theta = float(rng.uniform(0.0, 2.0 * math.pi))
                px = cl_x + r_loc * math.cos(theta)
                py = cl_y + r_loc * math.sin(theta)
                # Keep inside wafer disk
                d_center = math.hypot(px - cx, py - cy)
                if d_center > 0.94 * R:
                    scale = (0.94 * R) / d_center
                    px = cx + (px - cx) * scale
                    py = cy + (py - cy) * scale
                pts.append((round(px, 2), round(py, 2)))

    else:  # "random"
        max_r = 0.92 * R if min_separation_px > 0.0 else R
        for _ in range(count):
            for _attempt in range(100):
                r = max_r * math.sqrt(float(rng.uniform(0.0, 1.0)))
                theta = float(rng.uniform(0.0, 2.0 * math.pi))
                px = cx + r * math.cos(theta)
                py = cy + r * math.sin(theta)
                if _far_enough(px, py, min_separation_px):
                    pts.append((round(px, 2), round(py, 2)))
                    break
            else:
                r = max_r * math.sqrt(float(rng.uniform(0.0, 1.0)))
                theta = float(rng.uniform(0.0, 2.0 * math.pi))
                pts.append((round(cx + r * math.cos(theta), 2), round(cy + r * math.sin(theta), 2)))

    return pts, LAYOUT_TO_PATTERN_LABEL[layout]


def generate_layout_defects(
    layout: DefectLayout,
    count: int = 8,
    size: int = 512,
    wafer_radius_ratio: float = 0.44,
    seed: int = 42,
    radius: float = 7.0,
    contrast: float = 65.0,
) -> Tuple[List[Dict[str, Any]], str]:
    """Generate synthetic blob defect specs arranged in the requested wafer spatial layout."""
    center = (size / 2.0, size / 2.0)
    wafer_r = size * wafer_radius_ratio
    # Keep minimum center-to-center distance >= 2*radius + 7 px so adjacent blobs stay
    # strictly separated beyond the default merge_gap_px=3 radius.
    min_sep = 2.0 * float(radius) + 7.0
    centroids, label = generate_layout_centroids(
        layout=layout,
        count=count,
        center=center,
        radius=wafer_r,
        seed=seed,
        min_separation_px=min_sep,
    )
    defects = [
        {
            "type": "bright_blob",
            "center": (px, py),
            "radius": float(radius),
            "contrast": float(contrast),
        }
        for px, py in centroids
    ]
    return defects, label


def _build_synthetic_circuit_tile(
    pitch: int,
    substrate_intensity: float,
    seed: Optional[int],
) -> np.ndarray:
    """Build a deterministic synthetic pitch x pitch circuit-like block tile.

    Generates random rectangular blocks spanning a wide intensity range (60 to 210)
    plus subtle die street border so every tiled die is identical before noise.
    """
    tile_rng = np.random.default_rng(seed)
    tile = np.full((pitch, pitch), substrate_intensity, dtype=np.float32)

    # Outer die street border (1 px)
    tile[0, :] = 80.0
    tile[:, 0] = 80.0

    # Random rectangular circuit blocks spanning wide intensity range [60, 210]
    num_blocks = max(6, pitch // 3)
    for _ in range(num_blocks):
        bw = int(tile_rng.integers(max(2, pitch // 10), max(3, pitch // 3)))
        bh = int(tile_rng.integers(max(2, pitch // 10), max(3, pitch // 3)))
        bx = int(tile_rng.integers(1, max(2, pitch - bw)))
        by = int(tile_rng.integers(1, max(2, pitch - bh)))
        intensity = float(tile_rng.uniform(60.0, 210.0))
        tile[by : by + bh, bx : bx + bw] = intensity

    return tile


def generate_synthetic_wafer(
    size: int = 512,
    seed: Optional[int] = 42,
    wafer_radius_ratio: float = 0.44,
    substrate_intensity: float = 140.0,
    background_intensity: float = 20.0,
    grid_pitch: int = 32,
    grid_line_width: int = 2,
    grid_intensity: float = 90.0,
    noise_std: float = 5.0,
    has_gradient: bool = False,
    gradient_strength: float = 35.0,
    defects: Optional[List[Dict[str, Any]]] = None,
    die_pattern: Literal["grid", "circuit"] = "grid",
    pattern_seed: Optional[int] = None,
    defect_layout: Optional[DefectLayout] = None,
    layout_defect_count: int = 8,
    layout_defect_radius: float = 7.0,
    layout_defect_contrast: float = 65.0,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Generate a deterministic synthetic wafer image with ground-truth defect annotations.

    Args:
        size: Width and height of the square synthetic canvas in pixels.
        seed: Random seed for numpy default_rng to ensure reproducibility.
        wafer_radius_ratio: Ratio of wafer radius to canvas dimension.
        substrate_intensity: Nominal pixel intensity for clean wafer silicon substrate.
        background_intensity: Nominal pixel intensity for outer background chuck.
        grid_pitch: Die pitch in pixels (used for both 'grid' and 'circuit' die patterns; 0 disables pattern).
        grid_line_width: Width in pixels of die streets/grid lines (for 'grid' pattern).
        grid_intensity: Pixel intensity along die grid streets (for 'grid' pattern).
        noise_std: Standard deviation of additive zero-mean Gaussian sensor noise.
        has_gradient: Whether to synthesize an uneven spatial illumination gradient.
        gradient_strength: Peak-to-peak amplitude of the illumination gradient.
        defects: Optional list of synthetic defect specifications. Each item is a dict:
                 - "type": "bright_blob" | "dark_blob" | "scratch"
                 - "center": (x, y) in canvas coordinates
                 - "radius": float (for blobs) or "size": float
                 - "contrast": float (positive intensity delta for bright, negative for dark)
                 - for "scratch": "start": (x1, y1), "end": (x2, y2), "width": int
        die_pattern: Repeating die pattern style:
                     - "grid": classic periodic die street grid (default)
                     - "circuit": seeded random multi-intensity block tile repeated at `grid_pitch`
        pattern_seed: Optional separate RNG seed for the circuit die tile (defaults to `seed`).

    Returns:
        Tuple[np.ndarray, List[Dict[str, Any]]]:
            - image_uint8: 2D uint8 NumPy array of the synthetic wafer.
            - ground_truth: List of ground-truth defect metadata dicts with verified locations.
    """
    rng = np.random.default_rng(seed)

    # 1. Coordinate grids
    y, x = np.ogrid[:size, :size]
    center_y, center_x = size / 2.0, size / 2.0
    radius = size * wafer_radius_ratio
    wafer_mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2

    # 2. Base canvas (dark background and wafer substrate)
    canvas = np.full((size, size), background_intensity, dtype=np.float32)
    canvas[wafer_mask] = substrate_intensity

    # 3. Fabricated periodic die pattern texture
    if grid_pitch > 0:
        if die_pattern == "grid":
            if grid_line_width > 0:
                grid_mask = ((x % grid_pitch < grid_line_width) | (y % grid_pitch < grid_line_width)) & wafer_mask
                canvas[grid_mask] = grid_intensity
        elif die_pattern == "circuit":
            tile_seed = seed if pattern_seed is None else pattern_seed
            tile = _build_synthetic_circuit_tile(grid_pitch, substrate_intensity, tile_seed)
            reps = (size // grid_pitch) + 1
            tiled = np.tile(tile, (reps, reps))[:size, :size]
            canvas[wafer_mask] = tiled[wafer_mask]
        else:
            raise ValueError(f"Unsupported die_pattern '{die_pattern}'. Allowed: 'grid', 'circuit'")

    # 4. Optional spatial illumination gradient across wafer
    if has_gradient and gradient_strength > 0:
        # Diagonal linear gradient across wafer: bright at top-left, darker at bottom-right
        grad = ((size - x) + (size - y)) / (2.0 * size) * (gradient_strength * 2.0) - gradient_strength
        canvas[wafer_mask] += grad[wafer_mask]

    # 5. Additive Gaussian noise
    if noise_std > 0:
        noise = rng.normal(0.0, noise_std, (size, size)).astype(np.float32)
        canvas += noise

    # 6. Inject synthetic defects
    ground_truth: List[Dict[str, Any]] = []
    layout_label: Optional[str] = None
    effective_defects = defects
    if effective_defects is None and defect_layout is not None:
        layout_seed = (seed if seed is not None else 42) + 10000
        effective_defects, layout_label = generate_layout_defects(
            layout=defect_layout,
            count=layout_defect_count,
            size=size,
            wafer_radius_ratio=wafer_radius_ratio,
            seed=layout_seed,
            radius=layout_defect_radius,
            contrast=layout_defect_contrast,
        )

    if effective_defects:
        for defect in effective_defects:
            d_type = defect.get("type", "bright_blob")
            contrast = float(defect.get("contrast", 40.0))

            if d_type in ("bright_blob", "dark_blob"):
                cx, cy = defect.get("center", (size / 2.0, size / 2.0))
                r = float(defect.get("radius", defect.get("size", 8.0) / 2.0))
                dist_sq = (x - cx) ** 2 + (y - cy) ** 2
                blob_mask = (dist_sq <= r ** 2) & wafer_mask
                true_area_visible_px = int(np.count_nonzero(blob_mask))

                if d_type == "bright_blob":
                    canvas[blob_mask] += abs(contrast)
                    contrast_val = abs(contrast)
                else:
                    canvas[blob_mask] -= abs(contrast)
                    contrast_val = -abs(contrast)

                gt_entry: Dict[str, Any] = {
                    "type": d_type,
                    "center": (float(cx), float(cy)),
                    "radius": float(r),
                    "size": float(2.0 * r),
                    "contrast": contrast_val,
                    "true_area_visible_px": true_area_visible_px,
                }
                if layout_label is not None:
                    gt_entry["layout"] = defect_layout
                    gt_entry["layout_label"] = layout_label
                ground_truth.append(gt_entry)

            elif d_type == "scratch":
                start = defect.get("start", (int(center_x - 30), int(center_y - 30)))
                end = defect.get("end", (int(center_x + 30), int(center_y + 30)))
                line_width = int(defect.get("width", 2))

                scratch_layer = np.zeros((size, size), dtype=np.uint8)
                cv2.line(
                    scratch_layer,
                    (int(start[0]), int(start[1])),
                    (int(end[0]), int(end[1])),
                    color=255,
                    thickness=line_width,
                )
                scratch_mask = (scratch_layer > 0) & wafer_mask
                true_area_visible_px = int(np.count_nonzero(scratch_mask))
                canvas[scratch_mask] += contrast

                mid_x = (start[0] + end[0]) / 2.0
                mid_y = (start[1] + end[1]) / 2.0
                length = float(np.hypot(end[0] - start[0], end[1] - start[1]))

                # Measure true rendered mask length & thickness via minAreaRect (matches pixel coverage)
                ys_s, xs_s = np.nonzero(scratch_mask)
                if xs_s.size >= 2:
                    pts_s = np.column_stack((xs_s, ys_s)).astype(np.float32)
                    (_, _), (rw_s, rh_s), _ = cv2.minAreaRect(pts_s)
                    true_mask_length_px = round(max(float(rw_s), float(rh_s)) + 1.0, 2)
                    true_mask_thickness_px = round(min(float(rw_s), float(rh_s)) + 1.0, 2)
                else:
                    true_mask_length_px = round(length, 2)
                    true_mask_thickness_px = float(line_width)

                gt_entry = {
                    "type": "scratch",
                    "center": (float(mid_x), float(mid_y)),
                    "start": (float(start[0]), float(start[1])),
                    "end": (float(end[0]), float(end[1])),
                    "size": length,
                    "radius": length / 2.0,
                    "width": line_width,
                    "contrast": contrast,
                    "true_area_visible_px": true_area_visible_px,
                    "true_length_px": round(length, 2),
                    "true_thickness_px": float(line_width),
                    "true_mask_length_px": true_mask_length_px,
                    "true_mask_thickness_px": true_mask_thickness_px,
                }
                if layout_label is not None:
                    gt_entry["layout"] = defect_layout
                    gt_entry["layout_label"] = layout_label
                ground_truth.append(gt_entry)

    # 7. Clip to valid 8-bit intensity range and cast to uint8
    image_uint8 = np.clip(canvas, 0, 255).astype(np.uint8)
    return image_uint8, ground_truth


def generate_layout_wafer(
    defect_layout: DefectLayout,
    count: int = 8,
    size: int = 512,
    seed: int = 42,
    grid_pitch: int = 32,
    die_pattern: Literal["grid", "circuit"] = "grid",
    noise_std: float = 5.0,
    radius: float = 7.0,
    contrast: float = 65.0,
) -> Tuple[np.ndarray, List[Dict[str, Any]], str]:
    """Generate a synthetic patterned wafer with defects placed in `defect_layout` and return its GT label."""
    img, gt = generate_synthetic_wafer(
        size=size,
        seed=seed,
        grid_pitch=grid_pitch,
        die_pattern=die_pattern,
        noise_std=noise_std,
        defect_layout=defect_layout,
        layout_defect_count=count,
        layout_defect_radius=radius,
        layout_defect_contrast=contrast,
    )
    return img, gt, LAYOUT_TO_PATTERN_LABEL[defect_layout]


def create_standard_synthetic_pair(
    size: int = 512,
    seed: int = 42,
    shift: Tuple[float, float] = (0.0, 0.0),
    defects: Optional[List[Dict[str, Any]]] = None,
    grid_pitch: int = 32,
    die_pattern: Literal["grid", "circuit"] = "grid",
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Create a matching synthetic golden reference and test image pair.

    The golden reference has no defects and is clean.
    The test image contains the injected defects and optional translational shift.

    Args:
        size: Dimensions of square image.
        seed: Random seed for deterministic reproducibility.
        shift: (dx, dy) translation shift to apply to the test image.
        defects: List of synthetic defects injected into the test image.
        grid_pitch: Die pitch in pixels (default 32).
        die_pattern: Repeating die pattern style ("grid" or "circuit").

    Returns:
        Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
            (reference_image, test_image, ground_truth_defects)
    """
    # Clean reference image
    reference_image, _ = generate_synthetic_wafer(
        size=size,
        seed=seed,
        grid_pitch=grid_pitch,
        has_gradient=False,
        defects=None,
        die_pattern=die_pattern,
        pattern_seed=seed,
    )

    # Test image with defects (same pattern layout, independent noise realization)
    test_image, ground_truth = generate_synthetic_wafer(
        size=size,
        seed=seed + 1,
        grid_pitch=grid_pitch,
        has_gradient=False,
        defects=defects,
        die_pattern=die_pattern,
        pattern_seed=seed,
    )

    # Apply translation if requested
    dx, dy = shift
    if abs(dx) > 1e-4 or abs(dy) > 1e-4:
        M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        test_image = cv2.warpAffine(
            test_image,
            M,
            (size, size),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=20,
        )
        # Shift ground-truth coordinates
        shifted_gt = []
        for d in ground_truth:
            d_shifted = d.copy()
            cx, cy = d["center"]
            d_shifted["center"] = (cx + dx, cy + dy)
            if "start" in d:
                sx, sy = d["start"]
                ex, ey = d["end"]
                d_shifted["start"] = (sx + dx, sy + dy)
                d_shifted["end"] = (ex + dx, ey + dy)
            shifted_gt.append(d_shifted)
        ground_truth = shifted_gt

    return reference_image, test_image, ground_truth
