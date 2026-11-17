# WaferInspect

WaferInspect is an automated wafer defect inspection and measurement prototype system designed to detect surface anomalies, segment defect regions, perform geometric measurements, and diagnose spatial defect patterns using wafer-like grayscale imagery.

## Scripts & Utilities

- `scripts/generate_noisy_wafer.py`: Generates synthetic wafer test images (512x512 grayscale) simulating silicon wafer substrate disks, die grids, additive Gaussian sensor noise, and radial/linear illumination gradients to exercise denoising and adaptive contrast enhancement.
- `scripts/test_flow.ps1`: Automated PowerShell script to verify the full backend inspection workflow end-to-end. Starts the FastAPI server, runs the health check, generates and uploads a synthetic test wafer, executes the computer vision preprocessing pipeline, logs step-by-step metrics and timing, and downloads the resulting preprocessed artifact.
