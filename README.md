# WaferInspect

WaferInspect is an automated wafer defect inspection and measurement prototype system designed to detect surface anomalies, segment defect regions, perform geometric measurements, and diagnose spatial defect patterns using wafer-like grayscale imagery.

## Die-to-Die Self-Reference Detection

When no golden reference wafer is supplied, WaferInspect runs in `no_reference_mode="auto"` by default:
1. **Periodicity & Pitch Estimation**: Mean-subtracts the image inside the wafer mask and computes a zero-padded 2D FFT linear autocorrelation to detect repeating die pitch `(pitch_x, pitch_y)` with at least 3 repeats (`8 <= pitch <= image_size / 3`) and peak prominence above threshold.
2. **Median Neighbourhood Reference**: For each pixel, constructs a local defect-free reference from the **median** of up to 8 neighbouring die positions `(±pitch_x, 0)`, `(0, ±pitch_y)`, `(±pitch_x, ±pitch_y)` lying inside the wafer mask. Taking the median prevents a defect in one die from contaminating its neighbours' references.
3. **Fallback**: If no periodic die pattern is detected (e.g. blanket/unpatterned wafers), auto mode falls back to masked Gaussian background subtraction and emits a warning containing the measured autocorrelation peak strengths.

### Assumptions & Limitations
- **Regular periodic pattern & integer pitch only**: Assumes a uniform rectangular die array with integer pixel pitch `(pitch_x, pitch_y)`. Non-integer/fractional pitches or rotational skew are not resampled; instead, an optional 3×3 shift-tolerance window (enabled by default) absorbs up to ±1 px misregistration at pattern edges at the trade-off of slightly reduced sensitivity to thin features parallel to or crossing high-contrast pattern edges.
- **Unverifiable border band**: Pixels with fewer than 3 valid in-wafer neighbouring dies are marked unverifiable and explicitly excluded from detection rather than guessed. Both verifiable `coverage` and `unverifiable_fraction` are reported in the detection summary and API response.

## Scripts & Utilities

- `scripts/generate_noisy_wafer.py`: Generates synthetic wafer test images (512x512 grayscale) via `backend/app/utils/synthetic.py`, simulating silicon wafer substrate disks, die grids or circuit block tiles, additive Gaussian sensor noise, and illumination gradients.
- `scripts/evaluate_detection.py`: Accuracy evaluation benchmark that runs the full classical CV pipeline (preprocessing, detection, segmentation, noise filtering) over 30 seeded synthetic wafers per (diameter, contrast) cell across golden reference, die-to-die self-reference (`grid` and `circuit`, with and without 3×3 shift tolerance), and Gaussian background paths, reporting post-filter recall, false positives per image, mean funnel counts, and coverage.
- `scripts/test_flow.ps1`: Automated PowerShell script to verify the full-stack inspection workflow end-to-end. Launches both backend and frontend servers, generates a synthetic patterned wafer with 4 injected defects, runs inspection via API, prints path label, pitch, coverage, stage funnel counts, and accepted/rejected regions, and verifies all image artifact endpoints.
