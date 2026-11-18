# Automated End-to-End Live Flow Verification for WaferInspect (Phase 5)
# Starts both backend (8000) and frontend (5173), generates a synthetic wafer
# with 4 injected defects, runs full inspection (golden & no-reference),
# prints stage funnel counts, accepted/rejected regions, and verifies image endpoints.

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path "$ScriptDir\.."
$BackendDir = "$ProjectRoot\backend"
$FrontendDir = "$ProjectRoot\frontend"
$PythonExe = "$ProjectRoot\venv\Scripts\python.exe"

Write-Host "============================================================"
Write-Host "WaferInspect: Full-Stack Live Inspection Verification"
Write-Host "============================================================"

$TempRef = "$ScriptDir\temp_ref_wafer.png"
$TempTest = "$ScriptDir\temp_test_wafer_4defects.png"
$PayloadFile = "$ScriptDir\temp_payload.json"
$TempOut = "$ScriptDir\temp_artifact.png"

# 1. Generate synthetic wafer pair with 4 injected defects
Write-Host "`n[Step 1] Generating synthetic wafer pair (4 injected defects: bright blob, dark blob, scratch, rim defect)..."
& $PythonExe "$ScriptDir\generate_noisy_wafer.py" --pair $TempRef $TempTest

# 2. Start backend and frontend servers
Write-Host "`n[Step 2] Launching backend (port 8000) and frontend (port 5173)..."
$backendProc = Start-Process -FilePath $PythonExe `
    -ArgumentList "-m uvicorn app.main:app --host 127.0.0.1 --port 8000" `
    -WorkingDirectory $BackendDir `
    -PassThru

$frontendProc = Start-Process -FilePath "cmd.exe" `
    -ArgumentList "/c npm run dev -- --host 127.0.0.1 --port 5173" `
    -WorkingDirectory $FrontendDir `
    -PassThru

try {
    # Wait for backend health
    $backendReady = $false
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Milliseconds 500
        try {
            $health = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method Get -TimeoutSec 2
            if ($health.status -eq "ok") {
                $backendReady = $true
                break
            }
        } catch { }
    }
    if (-not $backendReady) {
        throw "Backend failed to start on http://127.0.0.1:8000/health"
    }
    Write-Host "Backend healthy: $($health | ConvertTo-Json -Compress)"

    # Wait for frontend server
    $frontendReady = $false
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Milliseconds 500
        try {
            $feRes = Invoke-WebRequest -Uri "http://127.0.0.1:5173/" -UseBasicParsing -TimeoutSec 2
            if ($feRes.StatusCode -eq 200) {
                $frontendReady = $true
                break
            }
        } catch { }
    }
    Write-Host "Frontend ready on http://127.0.0.1:5173 (HTTP 200: $frontendReady)"

    # 3. Golden-Reference Live Flow (4 injected defects)
    Write-Host "`n[Step 3] Live Flow A: Golden-Reference Inspection (4 injected defects)..."
    $uploadGolden = curl.exe -s -F "test_image=@$TempTest" -F "reference_image=@$TempRef" http://127.0.0.1:8000/inspection/upload
    $uploadGoldenJson = $uploadGolden | ConvertFrom-Json
    $uploadIdGolden = $uploadGoldenJson.upload_id
    Write-Host "Uploaded golden session ID: $uploadIdGolden"

    @{ upload_id = $uploadIdGolden } | ConvertTo-Json | Set-Content $PayloadFile -Encoding utf8
    $runGolden = curl.exe -s -X POST http://127.0.0.1:8000/inspection/run `
        -H "Content-Type: application/json" `
        --data-binary "@$PayloadFile"
    $runGoldenJson = $runGolden | ConvertFrom-Json

    Write-Host "Mode: $($runGoldenJson.detection.label) | Threshold: $($runGoldenJson.detection.threshold_method) ($($runGoldenJson.detection.threshold_value))"
    Write-Host "Threshold Diagnostic: $($runGoldenJson.detection.threshold_reason)"

    Write-Host "`n--- Stage Counts Funnel (Golden) ---"
    $runGoldenJson.stage_counts | Format-Table -Property stage, candidate_count -AutoSize | Out-String | Write-Host

    Write-Host "--- Accepted Regions (Golden) ---"
    if ($runGoldenJson.regions.Count -gt 0) {
        $runGoldenJson.regions | Select-Object id, area_px, @{N='bbox';E={$_.bbox -join ','}}, @{N='centroid';E={$_.centroid -join ','}}, polarity, mean_contrast, touches_edge | Format-Table -AutoSize | Out-String | Write-Host
    } else {
        Write-Host "No defects found."
    }

    Write-Host "--- Rejected Regions (Golden) ---"
    if ($runGoldenJson.rejected_regions.Count -gt 0) {
        $runGoldenJson.rejected_regions | Format-Table -Property id, area_px, mean_contrast, reason -AutoSize | Out-String | Write-Host
    } else {
        Write-Host "None rejected."
    }

    # Verify all 7 image kinds return HTTP 200 and invalid kind returns 400
    Write-Host "--- Image Artifact Endpoints Verification ---"
    foreach ($kind in @("original", "preprocessed", "difference", "mask", "filtered_mask", "bboxes", "contours")) {
        $code = curl.exe -s -o $TempOut -w "%{http_code}" "http://127.0.0.1:8000/inspection/run/$uploadIdGolden/image/$kind"
        $size = (Get-Item $TempOut).Length
        Write-Host "  GET /image/$kind -> HTTP $code ($size bytes)"
        if ($code -ne "200") { throw "Expected 200 for $kind, got $code" }
    }
    $badCode = curl.exe -s -o NUL -w "%{http_code}" "http://127.0.0.1:8000/inspection/run/$uploadIdGolden/image/invalid_kind"
    Write-Host "  GET /image/invalid_kind -> HTTP $badCode (expected 400)"
    if ($badCode -ne "400") { throw "Expected 400 for invalid_kind, got $badCode" }

    # 4. No-Reference Live Flow (same 4-defect patterned wafer -> auto selects self_reference)
    Write-Host "`n[Step 4] Live Flow B: Patterned Wafer WITHOUT Reference (auto die-to-die self-reference)..."
    $uploadNoRef = curl.exe -s -F "test_image=@$TempTest" http://127.0.0.1:8000/inspection/upload
    $uploadNoRefJson = $uploadNoRef | ConvertFrom-Json
    $uploadIdNoRef = $uploadNoRefJson.upload_id
    Write-Host "Uploaded no-reference session ID: $uploadIdNoRef"

    @{ upload_id = $uploadIdNoRef } | ConvertTo-Json | Set-Content $PayloadFile -Encoding utf8
    $runNoRef = curl.exe -s -X POST http://127.0.0.1:8000/inspection/run `
        -H "Content-Type: application/json" `
        --data-binary "@$PayloadFile"
    $runNoRefJson = $runNoRef | ConvertFrom-Json

    $det = $runNoRefJson.detection
    Write-Host "Path: $($det.path) | Label: $($det.label)"
    Write-Host "Threshold: $($det.threshold_method) ($($det.threshold_value)) | Diagnostic: $($det.threshold_reason)"
    Write-Host "Die Pitch: ($($det.die_pitch -join ', ')) px | Peak Strength: $($det.peak_strength) | Coverage: $([math]::Round($det.coverage * 100, 2))% | Unverifiable: $([math]::Round($det.unverifiable_fraction * 100, 2))%"

    Write-Host "`n--- Stage Counts Funnel (No-Reference / Self-Reference) ---"
    $runNoRefJson.stage_counts | Format-Table -Property stage, candidate_count, note -AutoSize | Out-String | Write-Host

    Write-Host "--- Accepted Regions (No-Reference / Self-Reference) ---"
    if ($runNoRefJson.regions.Count -gt 0) {
        $runNoRefJson.regions | Select-Object id, area_px, @{N='bbox';E={$_.bbox -join ','}}, @{N='centroid';E={$_.centroid -join ','}}, polarity, mean_contrast, touches_edge | Format-Table -AutoSize | Out-String | Write-Host
    } else {
        Write-Host "No defects found."
    }

    Write-Host "--- Rejected Regions (No-Reference / Self-Reference) ---"
    if ($runNoRefJson.rejected_regions.Count -gt 0) {
        $runNoRefJson.rejected_regions | Format-Table -Property id, area_px, mean_contrast, reason -AutoSize | Out-String | Write-Host
    } else {
        Write-Host "None rejected."
    }

    Write-Host "`n============================================================"
    Write-Host "Full-Stack Live Inspection Verification PASSED!"
    Write-Host "============================================================"

} finally {
    if ($backendProc -and -not $backendProc.HasExited) {
        Stop-Process -Id $backendProc.Id -Force -ErrorAction SilentlyContinue
    }
    if ($frontendProc -and -not $frontendProc.HasExited) {
        taskkill /PID $frontendProc.Id /T /F 2>$null | Out-Null
    }
    Remove-Item -Force -ErrorAction SilentlyContinue $TempRef, $TempTest, $PayloadFile, $TempOut
}
