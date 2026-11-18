# WaferInspect

WaferInspect is an automated wafer defect inspection and measurement prototype system designed to detect surface anomalies, segment defect regions, perform geometric measurements, and diagnose spatial defect patterns using wafer-like grayscale imagery.

## Scripts & Utilities

- `scripts/generate_noisy_wafer.py`: Generates synthetic wafer test images (512x512 grayscale) via `backend/app/utils/synthetic.py`, simulating silicon wafer substrate disks, die grids, additive Gaussian sensor noise, and illumination gradients.
- `scripts/evaluate_detection.py`: Accuracy evaluation benchmark that runs the full classical CV pipeline (preprocessing, detection, segmentation, noise filtering) over 30 seeded synthetic wafers per (diameter, contrast) cell for both golden-reference and experimental no-reference paths, reporting post-filter recall, false positives per image, and mean funnel counts.
- `scripts/test_flow.ps1`: Automated PowerShell script to verify the full-stack inspection workflow end-to-end. Launches both backend and frontend servers, generates a synthetic wafer with 4 injected defects, runs golden and no-reference inspection via API, prints stage funnel counts and accepted/rejected regions, and verifies all image artifact endpoints.
