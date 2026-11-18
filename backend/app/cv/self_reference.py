"""Die-to-die self-reference defect detection for periodic patterned wafers.

Estimates repeating die pitch along x and y via 2D FFT autocorrelation and constructs
a defect-free local reference image by taking the per-pixel median across neighbouring
dies. Because a defect in one die is an outlier among 3 to 8 neighbouring dies, the
median suppresses defects from contaminating the reference.

LIMITATIONS:
- Assumes a regular, periodic rectangular die layout with **integer pixel pitch**
  (pitch_x, pitch_y) and at least 3 full die repeats across the wafer (8 <= pitch <= size / 3).
- Subpixel/fractional die pitches are not currently resampled; instead, an optional 3x3
  shift-tolerance window absorbs up to +/- 1 px edge misregistration at pattern boundaries.
- Pixels near the wafer rim with fewer than `min_valid_neighbors` (default 3) in-wafer
  neighbouring die samples are marked unverifiable and explicitly excluded from detection
  (reported via `coverage` and `unverifiable_fraction`).
"""

from dataclasses import dataclass
import logging
from typing import Optional, Tuple
import cv2
import numpy as np

from app.cv.detection import estimate_noise_sigma

logger = logging.getLogger("waferinspect.cv.self_reference")


@dataclass
class PitchEstimate:
    """Result of 2D FFT autocorrelation die-pitch estimation."""

    periodic: bool
    pitch_x: Optional[int]
    pitch_y: Optional[int]
    peak_strength_x: float
    peak_strength_y: float
    peak_strength: float
    reason: str


@dataclass
class SelfReferenceResult:
    """Artifacts and coverage metrics from die-to-die self-reference construction."""

    reference_image: np.ndarray  # 2D uint8
    verifiable_mask: np.ndarray  # 2D uint8 (255 where verifiable, 0 where unverifiable/outside wafer)
    difference_image: np.ndarray  # 2D uint8
    signed_difference: np.ndarray  # 2D int16
    pitch_x: int
    pitch_y: int
    coverage: float
    unverifiable_fraction: float
    shift_tolerance_applied: bool
    noise_sigma: float = 0.0


def _next_pow2(n: int) -> int:
    """Return smallest power of two >= n for fast zero-padded FFT."""
    return 1 << (int(n) - 1).bit_length()


def _find_fundamental_peak(
    profile: np.ndarray,
    min_pitch: int,
    max_pitch: int,
) -> Tuple[Optional[int], float]:
    """Find the fundamental periodic peak lag and its valley-referenced prominence in [min_pitch, max_pitch].

    Uses local maximum prominence above the preceding valley (between lag min_pitch//2 and k)
    so smooth illumination gradients (which create a monotonically decaying central lobe)
    do not produce false peaks or mask genuine periodic peaks.
    """
    if max_pitch < min_pitch or len(profile) <= max_pitch + 1:
        return None, 0.0

    valley_start = max(2, min_pitch // 2)
    candidates = []

    for lag in range(min_pitch, max_pitch + 1):
        val = float(profile[lag])
        if val > float(profile[lag - 1]) and val >= float(profile[lag + 1]):
            left_valley = float(np.min(profile[valley_start:lag]))
            prominence = val - left_valley
            if prominence > 0.0:
                candidates.append((lag, prominence, val))

    if not candidates:
        return None, 0.0

    # Primary selection: peak with highest prominence above preceding valley
    best_lag, best_prom, _ = max(candidates, key=lambda item: item[1])

    # Guard against picking a harmonic (2*p, 3*p) over the fundamental period p:
    # if an integer sub-multiple >= min_pitch is also a strong local peak (>= 60% prominence),
    # prefer the smallest such fundamental lag.
    cand_map = {lag: prom for lag, prom, _ in candidates}
    for divisor in range(2, (best_lag // min_pitch) + 1):
        if best_lag % divisor == 0:
            sub_lag = best_lag // divisor
            if sub_lag >= min_pitch and sub_lag in cand_map:
                if cand_map[sub_lag] >= 0.60 * best_prom:
                    best_lag = sub_lag
                    best_prom = cand_map[sub_lag]

    return int(best_lag), float(best_prom)


def estimate_die_pitch(
    image: np.ndarray,
    wafer_mask: np.ndarray,
    min_pitch: int = 8,
    min_peak_strength: float = 0.10,
) -> PitchEstimate:
    """Estimate integer die pitch (pitch_x, pitch_y) via 2D FFT linear autocorrelation.

    Mean-subtracts the image strictly inside `wafer_mask` (zero outside), computes the
    zero-padded 2D linear autocorrelation via FFT, and locates the strongest non-zero-lag
    periodic peaks along the x and y axes within `[min_pitch, size // 3]`.

    Args:
        image: 2D uint8 grayscale wafer image.
        wafer_mask: 2D uint8 binary wafer mask (non-zero inside wafer).
        min_pitch: Minimum valid die pitch in pixels (default 8).
        min_peak_strength: Minimum normalized peak prominence required to classify as periodic.

    Returns:
        PitchEstimate indicating periodicity, integer pitch_x/pitch_y, normalized strengths,
        and a diagnostic reason string containing the measured values.
    """
    if not isinstance(image, np.ndarray) or image.ndim != 2:
        raise ValueError("estimate_die_pitch requires a 2D grayscale image.")
    if not isinstance(wafer_mask, np.ndarray) or wafer_mask.shape != image.shape:
        raise ValueError("wafer_mask must match image shape.")

    valid = wafer_mask > 0
    if not np.any(valid):
        return PitchEstimate(
            periodic=False,
            pitch_x=None,
            pitch_y=None,
            peak_strength_x=0.0,
            peak_strength_y=0.0,
            peak_strength=0.0,
            reason="not periodic: empty wafer mask",
        )

    h, w = image.shape
    max_pitch_x = w // 3
    max_pitch_y = h // 3
    if max_pitch_x < min_pitch or max_pitch_y < min_pitch:
        return PitchEstimate(
            periodic=False,
            pitch_x=None,
            pitch_y=None,
            peak_strength_x=0.0,
            peak_strength_y=0.0,
            peak_strength=0.0,
            reason=f"not periodic: image ({w}x{h}) too small for min_pitch={min_pitch} with >=3 repeats",
        )

    # Mean-subtract strictly inside wafer mask, zero outside
    img_f = image.astype(np.float64)
    mean_val = float(np.mean(img_f[valid]))
    centered = np.where(valid, img_f - mean_val, 0.0)

    energy = float(np.sum(centered * centered))
    if energy <= 1e-9:
        return PitchEstimate(
            periodic=False,
            pitch_x=None,
            pitch_y=None,
            peak_strength_x=0.0,
            peak_strength_y=0.0,
            peak_strength=0.0,
            reason="not periodic: zero intensity variance inside wafer mask",
        )

    # Zero-padded 2D FFT linear autocorrelation (avoids circular wrap-around)
    fft_h = _next_pow2(2 * h)
    fft_w = _next_pow2(2 * w)
    spectrum = np.fft.rfft2(centered, s=(fft_h, fft_w))
    power = spectrum * np.conj(spectrum)
    autocorr = np.fft.irfft2(power, s=(fft_h, fft_w))
    norm_ac = autocorr / max(float(autocorr[0, 0]), 1e-12)

    profile_x = norm_ac[0, : max_pitch_x + 2]
    profile_y = norm_ac[: max_pitch_y + 2, 0]

    pitch_x, strength_x = _find_fundamental_peak(profile_x, min_pitch=min_pitch, max_pitch=max_pitch_x)
    pitch_y, strength_y = _find_fundamental_peak(profile_y, min_pitch=min_pitch, max_pitch=max_pitch_y)

    combined_strength = min(strength_x, strength_y)

    if (
        pitch_x is None
        or pitch_y is None
        or strength_x < min_peak_strength
        or strength_y < min_peak_strength
    ):
        reason = (
            f"not periodic (peak_strength_x={strength_x:.4f} at lag={pitch_x}, "
            f"peak_strength_y={strength_y:.4f} at lag={pitch_y}, "
            f"threshold={min_peak_strength:.2f})"
        )
        logger.info("Die pitch estimation: %s", reason)
        return PitchEstimate(
            periodic=False,
            pitch_x=None,
            pitch_y=None,
            peak_strength_x=round(strength_x, 4),
            peak_strength_y=round(strength_y, 4),
            peak_strength=round(combined_strength, 4),
            reason=reason,
        )

    reason = (
        f"Periodic die structure detected: pitch=({pitch_x}, {pitch_y}) px, "
        f"peak_strength_x={strength_x:.4f}, peak_strength_y={strength_y:.4f}"
    )
    logger.info("Die pitch estimation: %s", reason)
    return PitchEstimate(
        periodic=True,
        pitch_x=pitch_x,
        pitch_y=pitch_y,
        peak_strength_x=round(strength_x, 4),
        peak_strength_y=round(strength_y, 4),
        peak_strength=round(combined_strength, 4),
        reason=reason,
    )


def _shift_array_with_mask(
    arr: np.ndarray,
    mask_bool: np.ndarray,
    dx: int,
    dy: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample `arr(y + dy, x + dx)` and `mask_bool(y + dy, x + dx)` with out-of-bounds marked invalid."""
    h, w = arr.shape
    out_vals = np.full((h, w), np.nan, dtype=np.float32)
    out_valid = np.zeros((h, w), dtype=bool)

    src_y0 = max(0, dy)
    src_y1 = min(h, h + dy)
    src_x0 = max(0, dx)
    src_x1 = min(w, w + dx)

    if src_y1 <= src_y0 or src_x1 <= src_x0:
        return out_vals, out_valid

    dst_y0 = src_y0 - dy
    dst_y1 = src_y1 - dy
    dst_x0 = src_x0 - dx
    dst_x1 = src_x1 - dx

    valid_slice = mask_bool[src_y0:src_y1, src_x0:src_x1]
    out_valid[dst_y0:dst_y1, dst_x0:dst_x1] = valid_slice
    vals_slice = np.where(valid_slice, arr[src_y0:src_y1, src_x0:src_x1].astype(np.float32), np.nan)
    out_vals[dst_y0:dst_y1, dst_x0:dst_x1] = vals_slice
    return out_vals, out_valid


def build_self_reference(
    image: np.ndarray,
    pitch_x: int,
    pitch_y: int,
    wafer_mask: np.ndarray,
    min_valid_neighbors: int = 3,
    use_shift_tolerance: bool = True,
) -> SelfReferenceResult:
    """Construct a median die-to-die self-reference and compute difference maps.

    For each pixel inside `wafer_mask`, gathers the corresponding pixel from the 8
    neighbouring dies at `(+-pitch_x, 0)`, `(0, +-pitch_y)`, and `(+-pitch_x, +-pitch_y)`
    whose shifted coordinates fall inside `wafer_mask`. Taking the **median** across
    neighbours prevents a defect in one die from contaminating its neighbours' references.

    Pixels with fewer than `min_valid_neighbors` (default 3) valid neighbours are marked
    unverifiable and excluded from the detection mask; `coverage` and `unverifiable_fraction`
    report the exact fraction of wafer pixels verified.

    Registration tolerance (`use_shift_tolerance=True`, default):
        Computes the minimum absolute difference over a 3x3 shift window `{-1, 0, +1}^2`
        of the reference image (preserving the signed difference at the minimizing shift).
        Trade-off: absorbs up to +/- 1 px pattern-edge misregistration (preventing false
        alarms on high-contrast die streets and circuit block edges), at the cost of
        slightly reducing sensitivity to ultra-thin (1-2 px) features lying directly
        parallel to high-contrast pattern edges.

    LIMITATION: Supports integer die pitches `(pitch_x, pitch_y)` only.

    Args:
        image: 2D uint8 grayscale wafer image.
        pitch_x: Integer horizontal die pitch in pixels.
        pitch_y: Integer vertical die pitch in pixels.
        wafer_mask: 2D uint8 wafer domain mask (non-zero inside wafer).
        min_valid_neighbors: Minimum valid neighbouring dies required (default 3).
        use_shift_tolerance: Whether to apply 3x3 shift-window minimum difference.

    Returns:
        SelfReferenceResult with reference image, verifiable mask, difference maps,
        and coverage metrics.
    """
    if pitch_x <= 0 or pitch_y <= 0:
        raise ValueError(f"pitch_x and pitch_y must be positive integers, got ({pitch_x}, {pitch_y})")
    if not isinstance(image, np.ndarray) or image.ndim != 2:
        raise ValueError("build_self_reference requires a 2D grayscale image.")
    if not isinstance(wafer_mask, np.ndarray) or wafer_mask.shape != image.shape:
        raise ValueError("wafer_mask must match image shape.")

    h, w = image.shape
    mask_bool = wafer_mask > 0
    wafer_px_count = int(np.count_nonzero(mask_bool))

    offsets = [
        (-pitch_x, 0),
        (pitch_x, 0),
        (0, -pitch_y),
        (0, pitch_y),
        (-pitch_x, -pitch_y),
        (-pitch_x, pitch_y),
        (pitch_x, -pitch_y),
        (pitch_x, pitch_y),
    ]

    samples = np.full((len(offsets), h, w), np.nan, dtype=np.float32)
    valid_count = np.zeros((h, w), dtype=np.int32)

    for idx, (dx, dy) in enumerate(offsets):
        shifted_vals, shifted_valid = _shift_array_with_mask(image, mask_bool, dx, dy)
        samples[idx] = shifted_vals
        valid_count += shifted_valid.astype(np.int32)

    verifiable_bool = mask_bool & (valid_count >= min_valid_neighbors)
    verifiable_px_count = int(np.count_nonzero(verifiable_bool))
    coverage = float(verifiable_px_count) / float(wafer_px_count) if wafer_px_count > 0 else 0.0
    unverifiable_fraction = max(0.0, 1.0 - coverage)

    reference_image = np.zeros((h, w), dtype=np.uint8)
    if verifiable_px_count > 0:
        # Vectorized median strictly on verifiable pixels (avoids all-NaN warnings)
        verifiable_samples = samples[:, verifiable_bool]
        med_vals = np.nanmedian(verifiable_samples, axis=0)
        reference_image[verifiable_bool] = np.clip(np.round(med_vals), 0, 255).astype(np.uint8)

    verifiable_mask = verifiable_bool.astype(np.uint8) * 255

    # Compute difference (with optional 3x3 shift tolerance window)
    test_i16 = image.astype(np.int16)
    ref_i16 = reference_image.astype(np.int16)

    if use_shift_tolerance:
        best_abs = np.full((h, w), 32767, dtype=np.int16)
        best_signed = np.zeros((h, w), dtype=np.int16)

        for sy in (-1, 0, 1):
            for sx in (-1, 0, 1):
                shifted_ref_f, shifted_verif = _shift_array_with_mask(reference_image, verifiable_bool, sx, sy)
                valid_pair = verifiable_bool & shifted_verif
                if not np.any(valid_pair):
                    continue
                shifted_ref_i16 = np.where(valid_pair, shifted_ref_f, 0.0).astype(np.int16)
                cand_signed = test_i16 - shifted_ref_i16
                cand_abs = np.abs(cand_signed)
                better = valid_pair & (cand_abs < best_abs)
                best_abs = np.where(better, cand_abs, best_abs)
                best_signed = np.where(better, cand_signed, best_signed)

        signed_diff = np.where(verifiable_bool, best_signed, np.int16(0))
        diff_uint8 = np.where(verifiable_bool, np.clip(best_abs, 0, 255), 0).astype(np.uint8)
    else:
        signed_diff = np.where(verifiable_bool, test_i16 - ref_i16, np.int16(0))
        diff_uint8 = np.where(verifiable_bool, np.clip(np.abs(signed_diff), 0, 255), 0).astype(np.uint8)

    # Compute physical noise sigma via Phase 4 MAD estimator on unshifted difference
    # (so 3x3 shift-tolerance minimum does not artificially deflate the noise sigma estimate)
    unshifted_abs = np.where(verifiable_bool, np.clip(np.abs(test_i16 - ref_i16), 0, 255), 0).astype(np.uint8)
    _, unshifted_sigma = estimate_noise_sigma(unshifted_abs, verifiable_mask)

    logger.info(
        "Self-reference built: pitch=(%d, %d), coverage=%.1f%%, unverifiable=%.1f%%, shift_tol=%s, noise_sigma=%.2f",
        pitch_x,
        pitch_y,
        coverage * 100.0,
        unverifiable_fraction * 100.0,
        use_shift_tolerance,
        unshifted_sigma,
    )

    return SelfReferenceResult(
        reference_image=reference_image,
        verifiable_mask=verifiable_mask,
        difference_image=diff_uint8,
        signed_difference=signed_diff,
        pitch_x=pitch_x,
        pitch_y=pitch_y,
        coverage=round(coverage, 4),
        unverifiable_fraction=round(unverifiable_fraction, 4),
        shift_tolerance_applied=use_shift_tolerance,
        noise_sigma=round(unshifted_sigma, 4),
    )
