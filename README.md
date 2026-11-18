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

## Wafer-Level Spatial Pattern Diagnosis

After measuring and classifying individual defects, WaferInspect runs wafer-level spatial pattern diagnosis (`backend/app/cv/pattern_analysis.py`) on normalized polar defect coordinates `(r = d / R, theta)` mapped back to original image coordinates:

- **Supported patterns**: `EDGE_RING`, `CENTER_CLUSTER`, `DONUT`, `LOCAL_CLUSTER`, `SCRATCH_LINE`, `RANDOM_SCATTER`, and `INSUFFICIENT_DATA`.
- **Single-defect shape exception**: Fewer than 3 non-elongated defects return `INSUFFICIENT_DATA`, unless a single region has elongation `>= 8.0` and length `>= 15%` of the wafer diameter, which is diagnosed as `SCRATCH_LINE` from shape evidence.
- **Seeded Monte Carlo significance testing**: For `n >= 3` defects, each spatial test statistic (band fraction + largest empty angular gap, total-least-squares explained variance + span, and Clark-Evans nearest-neighbour ratio `R`) is compared against `2,000` seeded uniform-in-disc random trials. A non-random pattern is reported **only** when both its geometric rule passes and `p_value <= 0.01` (with deterministic tie-break priority `SCRATCH_LINE → EDGE_RING → CENTER_CLUSTER → DONUT → LOCAL_CLUSTER`); otherwise the wafer is reported as `RANDOM_SCATTER` naming the closest candidate pattern and its p-value.
- **Evidence strength vs. probability**: Reported `confidence = 1 - p_value` is a **rule-based statistical evidence strength**, NOT a calibrated machine-learning probability. Spatial pattern diagnosis never alters inspection severity.
- **Limits of Monte Carlo diagnosis**: At very low defect counts (e.g. `n = 3`), uniform random placement has non-trivial probability of placing all 3 points inside a single radial band (`0.36^3 ≈ 4.7%` for the outer 20% ring), so the `p <= 0.01` gate deliberately suppresses low-n band calls to avoid false alarms. Full confusion matrices and false-pattern rates are recorded in [`docs/pattern_accuracy.md`](docs/pattern_accuracy.md), and half-maximum measurement refinement errors in [`docs/measurement_accuracy.md`](docs/measurement_accuracy.md).

## Evaluation Limits

- **Synthetic evaluation only**: All detection, segmentation, fragment-merging, measurement, and spatial pattern algorithms were developed and evaluated on a deterministic synthetic wafer generator (`backend/app/utils/synthetic.py`) with exactly periodic dies and integer pixel pitch. See [`docs/accuracy.md`](docs/accuracy.md), [`docs/measurement_accuracy.md`](docs/measurement_accuracy.md), and [`docs/pattern_accuracy.md`](docs/pattern_accuracy.md) for the full benchmark tables.
- **Rim defect area under-measurement**: Defects touching the wafer bevel measure roughly **20% low** in area because the 2% inward wafer-edge margin mask clips the outermost rim pixels to prevent bevel false alarms.
- **Thin low-contrast scratches**: Very narrow, low-contrast scratches (e.g. `T = 2 px`, contrast `30`) are measured poorly because 3×3 denoising attenuates sub-kernel line peaks and breaks low-SNR segments.
- **Low-count spatial pattern calls**: Spatial pattern diagnoses with **fewer than 5 defects** (`n = 3` or `4`) are tentative and carry an explicit `"low defect count: interpret with caution"` warning.
- **Real wafer images untested**: Performance on real brightfield/darkfield optical or SEM fab wafer imagery (with non-integer die pitch, optical distortion, lens vignetting, or wafer-edge bevel glare) has not been tested.
- **Uncalibrated prototype**: Measurements are reported in uncalibrated original image pixels unless the user explicitly provides `pixel_size_um`, and severity tiers (`NONE`/`LOW`/`MEDIUM`/`HIGH`) are a documented prototype heuristic rather than an industrial fab standard.

## Database & Switching to PostgreSQL

WaferInspect persists inspections, defects, and processing run configurations via **SQLAlchemy 2.0** and **Alembic** migrations (see [`docs/schema.md`](docs/schema.md) for the full ER diagram).

- **Default (SQLite)**: Resolves automatically to `sqlite:///<repo root>/data/waferinspect.db` with `PRAGMA foreign_keys=ON` and `PRAGMA journal_mode=WAL`.
- **Alembic CLI**: Run `alembic -c backend/alembic.ini upgrade head` to initialize or upgrade any target database. Application startup runs the exact same Alembic upgrade path.
- **Switching to PostgreSQL**: All ORM columns use generic SQLAlchemy types (`JSON`, `Float`, `DateTime(timezone=True)`, `Text`) with zero SQLite-only constructs. To run on PostgreSQL:
  1. Install a PostgreSQL driver such as `psycopg[binary]` (or `psycopg2-binary`).
  2. Set `DATABASE_URL` in `.env` or your environment:
     ```bash
     DATABASE_URL=postgresql+psycopg://waferinspect:password@localhost:5432/waferinspect
     ```
  3. Apply migrations:
     ```bash
     alembic -c backend/alembic.ini upgrade head
     ```
- **Synthetic Demonstration Seeding**: Run `python scripts/seed_demo_data.py` to execute 16 varied synthetic demonstration wafers through the real CV pipeline and persist them into the database for the Dashboard and History views.

## Scripts & Utilities

- `scripts/generate_noisy_wafer.py`: Generates synthetic wafer test images (512x512 grayscale) via `backend/app/utils/synthetic.py`, simulating silicon wafer substrate disks, die grids or circuit block tiles, additive Gaussian sensor noise, and illumination gradients.
- `scripts/evaluate_detection.py`: Accuracy evaluation benchmark that runs the full classical CV pipeline (preprocessing, detection, segmentation, noise filtering) over 30 seeded synthetic wafers per (diameter, contrast) cell across golden reference, die-to-die self-reference (`grid` and `circuit`, with and without 3×3 shift tolerance), and Gaussian background paths, reporting post-filter recall, false positives per image, mean funnel counts, and coverage.
- `scripts/test_flow.ps1`: Automated PowerShell script to verify the full-stack inspection workflow end-to-end. Launches both backend and frontend servers, generates a synthetic patterned wafer with 4 injected defects, runs inspection via API, prints path label, pitch, coverage, stage funnel counts, and accepted/rejected regions, and verifies all image artifact endpoints.
