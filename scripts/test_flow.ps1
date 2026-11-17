# Automated End-to-End Test Script for WaferInspect Pipeline
# Spins up the backend server, tests upload and preprocessing, and verifies output.

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path "$ScriptDir\.."
$BackendDir = "$ProjectRoot\backend"
$PythonExe = "$ProjectRoot\venv\Scripts\python.exe"

Write-Host "============================================================"
Write-Host "WaferInspect: Starting End-to-End Pipeline Verification"
Write-Host "============================================================"

# 1. Generate synthetic noisy wafer
$TempImg = "$ScriptDir\temp_noisy_wafer.png"
Write-Host "`n[Step 1] Generating synthetic noisy wafer image..."
& $PythonExe "$ScriptDir\generate_noisy_wafer.py" $TempImg

# 2. Start backend server
Write-Host "`n[Step 2] Launching backend server on http://127.0.0.1:8000..."
$backendProc = Start-Process -FilePath $PythonExe `
    -ArgumentList "-m uvicorn app.main:app --host 127.0.0.1 --port 8000" `
    -WorkingDirectory $BackendDir `
    -PassThru

try {
    # Wait for backend health check
    $ready = $false
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Milliseconds 500
        try {
            $health = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method Get -TimeoutSec 2
            if ($health.status -eq "ok") {
                $ready = $true
                break
            }
        } catch { }
    }

    if (-not $ready) {
        throw "Backend server failed to respond on http://127.0.0.1:8000/health within 15 seconds."
    }
    Write-Host "Backend healthy: $($health | ConvertTo-Json -Compress)"

    # 3. Upload image
    Write-Host "`n[Step 3] Uploading test wafer image via POST /inspection/upload..."
    $uploadOutput = curl.exe -s -F "test_image=@$TempImg" http://127.0.0.1:8000/inspection/upload
    Write-Host "Upload response: $uploadOutput"
    $uploadJson = $uploadOutput | ConvertFrom-Json
    $uploadId = $uploadJson.upload_id

    if (-not $uploadId) {
        throw "Upload failed or upload_id missing in response."
    }

    # 4. Run preprocessing pipeline
    Write-Host "`n[Step 4] Running preprocessing pipeline via POST /inspection/run..."
    $payloadFile = "$ScriptDir\temp_payload.json"
    $payloadData = @{
        upload_id = $uploadId
        config = @{
            denoise_enabled = $true
            denoise_method = "gaussian"
            denoise_kernel_size = 5
            contrast_method = "auto"
            low_contrast_std_threshold = 70.0
            normalize_enabled = $true
            normalize_mode = "0-255"
        }
    }
    $payloadData | ConvertTo-Json -Depth 5 | Set-Content $payloadFile -Encoding utf8
    $runOutput = curl.exe -s -X POST http://127.0.0.1:8000/inspection/run `
        -H "Content-Type: application/json" `
        --data-binary "@$payloadFile"
    
    $runJson = $runOutput | ConvertFrom-Json

    Write-Host "`n=== Preprocessing Results Summary ==="
    Write-Host "Upload ID:           $($runJson.upload_id)"
    Write-Host "Original Stats:      $($runJson.original_stats.width)x$($runJson.original_stats.height) | Mean: $($runJson.original_stats.mean) | Std: $($runJson.original_stats.std)"
    Write-Host "Preprocessed Stats:  $($runJson.preprocessed_stats.width)x$($runJson.preprocessed_stats.height) | Mean: $($runJson.preprocessed_stats.mean) | Std: $($runJson.preprocessed_stats.std)"
    
    Write-Host "`n=== Real Pipeline Step Log ==="
    $runJson.step_log | Format-Table -Property step, applied, duration_ms, mean, std, reason -AutoSize | Out-String | Write-Host

    # 5. Fetch preprocessed image
    Write-Host "[Step 5] Fetching preprocessed output image..."
    $downloadedImg = "$ScriptDir\temp_preprocessed.png"
    $statusCode = curl.exe -s -o $downloadedImg -w "%{http_code}" "http://127.0.0.1:8000/inspection/run/$uploadId/image/preprocessed"
    if ($statusCode -ne "200") {
        throw "Failed to fetch preprocessed image. HTTP Status: $statusCode"
    }
    $fileSize = (Get-Item $downloadedImg).Length
    Write-Host "Successfully retrieved preprocessed image (HTTP 200, $fileSize bytes)"

    Write-Host "`n============================================================"
    Write-Host "Pipeline End-to-End Verification PASSED!"
    Write-Host "============================================================"

} finally {
    if ($backendProc -and -not $backendProc.HasExited) {
        Write-Host "`nStopping backend server (PID $($backendProc.Id))..."
        Stop-Process -Id $backendProc.Id -Force
    }
    # Clean up temporary artifacts
    Remove-Item -Force -ErrorAction SilentlyContinue $TempImg, $payloadFile, $downloadedImg
}
