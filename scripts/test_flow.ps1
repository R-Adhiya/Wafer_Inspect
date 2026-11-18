# Automated End-to-End Live Flow Verification for WaferInspect (Phase 6)
# Starts both backend (8000) and frontend (5173), generates a patterned synthetic
# wafer with 4 injected defects (bright blob, dark blob, scratch crossing streets,
# rim defect), runs inspection WITHOUT a reference and WITH a reference, and prints
# the summary, measurement table vs ground truth, stage funnel, and artifact checks.

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path "$ScriptDir\.."
$BackendDir = "$ProjectRoot\backend"
$FrontendDir = "$ProjectRoot\frontend"
$PythonExe = "$ProjectRoot\venv\Scripts\python.exe"

Write-Host "============================================================"
Write-Host "WaferInspect Phase 6: Full-Stack Live Inspection Verification"
Write-Host "============================================================"

$TempRef = "$ScriptDir\temp_ref_wafer.png"
$TempTest = "$ScriptDir\temp_test_wafer_4defects.png"
$TempGt = "$ScriptDir\temp_gt_4defects.json"
$PayloadFile = "$ScriptDir\temp_payload.json"
$TempOut = "$ScriptDir\temp_artifact.png"

function Show-MeasurementComparison {
    param(
        [string]$RunLabel,
        $RunJson,
        $GtList
    )
    $s = $RunJson.summary
    $m = $RunJson.model
    Write-Host "`n--- Summary ($RunLabel) ---"
    Write-Host ("Model: {0} v{1} ({2}) | Processing Time: {3} s" -f $m.name, $m.version, $m.detection_method, $RunJson.processing_time_s)
    Write-Host ("Defects: {0} | Total Area: {1} px | Largest: {2} px | Average: {3} px | Severity: {4}" -f $s.defect_count, $s.total_defect_area_px, $s.largest_defect_area_px, $s.average_defect_area_px, $s.severity)
    Write-Host ("Severity Reason: {0}" -f $s.severity_reason)
    Write-Host ("Calibration: {0}" -f $s.calibration_status)

    Write-Host "`n--- Stage Counts Funnel ($RunLabel) ---"
    $RunJson.stage_counts | Format-Table -Property stage, candidate_count, note -AutoSize | Out-String | Write-Host

    Write-Host "--- Measurements vs Ground Truth ($RunLabel) ---"
    $rows = @()
    foreach ($reg in $RunJson.regions) {
        $bestGt = $null
        $bestDist = 1e9
        foreach ($g in $GtList) {
            $dx = [double]$reg.centroid[0] - [double]$g.center[0]
            $dy = [double]$reg.centroid[1] - [double]$g.center[1]
            $d = [math]::Sqrt($dx * $dx + $dy * $dy)
            if ($d -lt $bestDist) {
                $bestDist = $d
                $bestGt = $g
            }
        }
        $trueDesc = if ($bestGt.type -eq "scratch") {
            "L=$($bestGt.true_length_px), T=$($bestGt.true_thickness_px), A~$($bestGt.true_area_px)"
        } else {
            "D=$($bestGt.true_diameter_px), A=$($bestGt.true_area_px)"
        }
        $rows += [PSCustomObject]@{
            ID        = "#$($reg.id)"
            GT_Type   = $bestGt.type
            True_Size = $trueDesc
            Area_px   = $reg.area_px
            WxH       = "$($reg.width_px)x$($reg.height_px)"
            Length    = $reg.length_px
            Thick     = $reg.thickness_px
            Perim     = $reg.perimeter_px
            CNR       = "$($reg.contrast_to_noise)x"
            Severity  = $reg.severity
            Rim       = $reg.touches_edge
            Dist_GT   = [math]::Round($bestDist, 2)
        }
    }
    $rows | Format-Table -AutoSize | Out-String | Write-Host

    foreach ($reg in $RunJson.regions) {
        Write-Host ("  #{0} [{1}] why_flagged: {2}" -f $reg.id, $reg.severity, $reg.why_flagged)
    }
}

# 1. Generate patterned synthetic wafer pair with 4 injected defects + ground truth JSON
Write-Host "`n[Step 1] Generating patterned synthetic wafer pair (4 injected defects: bright blob, dark blob, scratch crossing streets, rim defect)..."
& $PythonExe "$ScriptDir\generate_noisy_wafer.py" --pair $TempRef $TempTest --gt $TempGt
$gtList = Get-Content $TempGt -Raw | ConvertFrom-Json

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
    if (-not $backendReady) { throw "Backend failed to start on http://127.0.0.1:8000/health" }
    Write-Host "Backend healthy: $($health | ConvertTo-Json -Compress)"

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

    # 3. Live Flow A: Patterned 4-defect wafer WITHOUT Reference (die-to-die self-reference)
    Write-Host "`n============================================================"
    Write-Host "[Step 3] Live Flow A: WITHOUT Reference (Self-Reference)"
    Write-Host "============================================================"
    $uploadNoRef = curl.exe -s -F "test_image=@$TempTest" http://127.0.0.1:8000/inspection/upload
    $uploadIdNoRef = ($uploadNoRef | ConvertFrom-Json).upload_id
    Write-Host "Uploaded no-reference session ID: $uploadIdNoRef"

    @{ upload_id = $uploadIdNoRef } | ConvertTo-Json | Set-Content $PayloadFile -Encoding utf8
    $runNoRef = curl.exe -s -X POST http://127.0.0.1:8000/inspection/run `
        -H "Content-Type: application/json" `
        --data-binary "@$PayloadFile"
    $runNoRefJson = $runNoRef | ConvertFrom-Json

    $det = $runNoRefJson.detection
    Write-Host "Path: $($det.path) | Label: $($det.label)"
    Write-Host "Die Pitch: ($($det.die_pitch -join ', ')) px | Peak Strength: $($det.peak_strength) | Coverage: $([math]::Round($det.coverage * 100, 2))%"
    Show-MeasurementComparison -RunLabel "No-Reference / Self-Reference" -RunJson $runNoRefJson -GtList $gtList

    # 4. Live Flow B: Same patterned 4-defect wafer WITH Reference (golden comparison)
    Write-Host "`n============================================================"
    Write-Host "[Step 4] Live Flow B: WITH Reference (Golden Comparison)"
    Write-Host "============================================================"
    $uploadGolden = curl.exe -s -F "test_image=@$TempTest" -F "reference_image=@$TempRef" http://127.0.0.1:8000/inspection/upload
    $uploadIdGolden = ($uploadGolden | ConvertFrom-Json).upload_id
    Write-Host "Uploaded golden session ID: $uploadIdGolden"

    @{ upload_id = $uploadIdGolden } | ConvertTo-Json | Set-Content $PayloadFile -Encoding utf8
    $runGolden = curl.exe -s -X POST http://127.0.0.1:8000/inspection/run `
        -H "Content-Type: application/json" `
        --data-binary "@$PayloadFile"
    $runGoldenJson = $runGolden | ConvertFrom-Json

    Write-Host "Path: $($runGoldenJson.detection.path) | Label: $($runGoldenJson.detection.label)"
    Show-MeasurementComparison -RunLabel "Golden Reference" -RunJson $runGoldenJson -GtList $gtList

    # 5. Verify all 8 image endpoints (including final_overlay) and 400 on invalid kind
    Write-Host "`n--- Image Artifact Endpoints Verification ---"
    foreach ($kind in @("original", "preprocessed", "difference", "mask", "filtered_mask", "bboxes", "contours", "final_overlay")) {
        $code = curl.exe -s -o $TempOut -w "%{http_code}" "http://127.0.0.1:8000/inspection/run/$uploadIdGolden/image/$kind"
        $size = (Get-Item $TempOut).Length
        Write-Host "  GET /image/$kind -> HTTP $code ($size bytes)"
        if ($code -ne "200") { throw "Expected 200 for $kind, got $code" }
    }
    $badCode = curl.exe -s -o NUL -w "%{http_code}" "http://127.0.0.1:8000/inspection/run/$uploadIdGolden/image/invalid_kind"
    Write-Host "  GET /image/invalid_kind -> HTTP $badCode (expected 400)"
    if ($badCode -ne "400") { throw "Expected 400 for invalid_kind, got $badCode" }

    Write-Host "`n============================================================"
    Write-Host "Phase 6 Full-Stack Live Inspection Verification PASSED!"
    Write-Host "============================================================"

} finally {
    if ($backendProc -and -not $backendProc.HasExited) {
        Stop-Process -Id $backendProc.Id -Force -ErrorAction SilentlyContinue
    }
    if ($frontendProc -and -not $frontendProc.HasExited) {
        taskkill /PID $frontendProc.Id /T /F 2>$null | Out-Null
    }
    Remove-Item -Force -ErrorAction SilentlyContinue $TempRef, $TempTest, $TempGt, $PayloadFile, $TempOut
}
