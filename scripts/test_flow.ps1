# Automated End-to-End Live Flow Verification for WaferInspect (Phase 6B)
# Starts both backend (8000) and frontend (5173):
#   1. Runs live flow on all 6 spatial layouts (patterned wafer, NO reference),
#      printing pattern type, confidence, p-value, explanation, and verifying pattern_overlay.
#   2. Runs measurement refinement OFF vs ON comparison on the 4-defect patterned wafer
#      (no-reference and golden) and verifies all 9 image artifact endpoints.

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path "$ScriptDir\.."
$BackendDir = "$ProjectRoot\backend"
$FrontendDir = "$ProjectRoot\frontend"
$PythonExe = "$ProjectRoot\venv\Scripts\python.exe"

Write-Host "================================================================="
Write-Host "WaferInspect Phase 6B: Full-Stack Live Verification (6 Layouts)"
Write-Host "================================================================="

$TempRef = "$ScriptDir\temp_ref_wafer.png"
$TempTest = "$ScriptDir\temp_test_wafer_4defects.png"
$TempLayout = "$ScriptDir\temp_layout_wafer.png"
$TempGt = "$ScriptDir\temp_gt_4defects.json"
$PayloadFile = "$ScriptDir\temp_payload.json"
$TempOut = "$ScriptDir\temp_artifact.png"

# Start backend and frontend servers
Write-Host "`n[Step 1] Launching backend (port 8000) and frontend (port 5173)..."
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
            if ($health.status -eq "ok") { $backendReady = $true; break }
        } catch { }
    }
    if (-not $backendReady) { throw "Backend failed to start on http://127.0.0.1:8000/health" }
    Write-Host "Backend healthy: $($health | ConvertTo-Json -Compress)"

    $frontendReady = $false
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Milliseconds 500
        try {
            $feRes = Invoke-WebRequest -Uri "http://127.0.0.1:5173/" -UseBasicParsing -TimeoutSec 2
            if ($feRes.StatusCode -eq 200) { $frontendReady = $true; break }
        } catch { }
    }
    Write-Host "Frontend ready on http://127.0.0.1:5173 (HTTP 200: $frontendReady)"

    # -----------------------------------------------------------------
    # Step 2: Live Flow for ALL 6 Spatial Layouts (patterned, NO ref)
    # -----------------------------------------------------------------
    Write-Host "`n================================================================="
    Write-Host "[Step 2] Live Flow: All 6 Spatial Layouts (Patterned, No Ref)"
    Write-Host "================================================================="

    $layouts = @("random", "edge_ring", "center_cluster", "donut", "scratch_line", "local_cluster")
    $layoutSummary = @()
    $lastUploadId = $null

    foreach ($lay in $layouts) {
        & $PythonExe "$ScriptDir\generate_noisy_wafer.py" --layout $lay $TempLayout | Out-Null
        $upRes = curl.exe -s -F "test_image=@$TempLayout" http://127.0.0.1:8000/inspection/upload | ConvertFrom-Json
        $uid = $upRes.upload_id
        $lastUploadId = $uid

        @{ upload_id = $uid } | ConvertTo-Json | Set-Content $PayloadFile -Encoding utf8
        $runJson = curl.exe -s -X POST http://127.0.0.1:8000/inspection/run `
            -H "Content-Type: application/json" `
            --data-binary "@$PayloadFile" | ConvertFrom-Json

        $pat = $runJson.pattern
        $ovCode = curl.exe -s -o $TempOut -w "%{http_code}" "http://127.0.0.1:8000/inspection/run/$uid/image/pattern_overlay"
        if ($ovCode -ne "200") { throw "Expected HTTP 200 for pattern_overlay on $lay, got $ovCode" }

        $confPct = [math]::Round([double]$pat.confidence * 100.0, 1)
        $pStr = if ($null -ne $pat.p_value) { "{0:F4}" -f [double]$pat.p_value } else { "N/A" }

        $layoutSummary += [PSCustomObject]@{
            Layout     = $lay
            Defects    = $runJson.summary.defect_count
            Pattern    = $pat.type
            Conf_Pct   = "$confPct%"
            p_value    = $pStr
            Overlay    = "HTTP $ovCode"
        }
        Write-Host ("  [{0,-14}] -> {1,-15} (conf={2,5}%, p={3}) | {4}" -f $lay, $pat.type, $confPct, $pStr, $pat.explanation)
    }

    Write-Host "`n--- 6-Layout Live Flow Summary ---"
    $layoutSummary | Format-Table -AutoSize | Out-String | Write-Host

    # -----------------------------------------------------------------
    # Step 3: Measurement Refinement OFF vs ON on 4-Defect Patterned Wafer
    # -----------------------------------------------------------------
    Write-Host "================================================================="
    Write-Host "[Step 3] Measurement Refinement OFF vs ON (4-Defect Wafer)"
    Write-Host "================================================================="
    & $PythonExe "$ScriptDir\generate_noisy_wafer.py" --pair $TempRef $TempTest --gt $TempGt
    $gtList = Get-Content $TempGt -Raw | ConvertFrom-Json

    $upNoRef = curl.exe -s -F "test_image=@$TempTest" http://127.0.0.1:8000/inspection/upload | ConvertFrom-Json
    $uidNoRef = $upNoRef.upload_id

    @{ upload_id = $uidNoRef; measurement_config = @{ refine_measurements = $false } } | ConvertTo-Json -Depth 4 | Set-Content $PayloadFile -Encoding utf8
    $runOff = curl.exe -s -X POST http://127.0.0.1:8000/inspection/run -H "Content-Type: application/json" --data-binary "@$PayloadFile" | ConvertFrom-Json

    @{ upload_id = $uidNoRef; measurement_config = @{ refine_measurements = $true } } | ConvertTo-Json -Depth 4 | Set-Content $PayloadFile -Encoding utf8
    $runOn = curl.exe -s -X POST http://127.0.0.1:8000/inspection/run -H "Content-Type: application/json" --data-binary "@$PayloadFile" | ConvertFrom-Json

    Write-Host "`n--- No-Reference (Self-Reference) Measurement Refinement OFF vs ON ---"
    $compRows = @()
    foreach ($regOn in $runOn.regions) {
        $regOff = $runOff.regions | Where-Object { $_.id -eq $regOn.id } | Select-Object -First 1
        $bestGt = $null
        $bestDist = 1e9
        foreach ($g in $gtList) {
            $dx = [double]$regOn.centroid[0] - [double]$g.center[0]
            $dy = [double]$regOn.centroid[1] - [double]$g.center[1]
            $d = [math]::Sqrt($dx * $dx + $dy * $dy)
            if ($d -lt $bestDist) { $bestDist = $d; $bestGt = $g }
        }
        $trueArea = [double]$bestGt.true_area_visible_px
        $errOff = [math]::Round([math]::Abs([double]$regOff.area_px - $trueArea) / $trueArea * 100.0, 1)
        $errOn = [math]::Round([math]::Abs([double]$regOn.area_px - $trueArea) / $trueArea * 100.0, 1)
        $compRows += [PSCustomObject]@{
            ID        = "#$($regOn.id)"
            GT_Type   = $bestGt.type
            True_Area = $bestGt.true_area_visible_px
            Area_OFF  = $regOff.area_px
            Err_OFF   = "$errOff%"
            Area_ON   = $regOn.area_px
            Err_ON    = "$errOn%"
            Det_Area  = $regOn.detected_area_px
            Length_ON = $regOn.length_px
            Thick_ON  = $regOn.thickness_px
        }
    }
    $compRows | Format-Table -AutoSize | Out-String | Write-Host

    # Verify all 9 image artifact endpoints + 400 on invalid kind
    Write-Host "--- All 9 Image Artifact Endpoints Verification ---"
    foreach ($kind in @("original", "preprocessed", "difference", "mask", "filtered_mask", "bboxes", "contours", "final_overlay", "pattern_overlay")) {
        $code = curl.exe -s -o $TempOut -w "%{http_code}" "http://127.0.0.1:8000/inspection/run/$uidNoRef/image/$kind"
        $size = (Get-Item $TempOut).Length
        Write-Host "  GET /image/$kind -> HTTP $code ($size bytes)"
        if ($code -ne "200") { throw "Expected 200 for $kind, got $code" }
    }
    $badCode = curl.exe -s -o NUL -w "%{http_code}" "http://127.0.0.1:8000/inspection/run/$uidNoRef/image/invalid_kind"
    Write-Host "  GET /image/invalid_kind -> HTTP $badCode (expected 400)"
    if ($badCode -ne "400") { throw "Expected 400 for invalid_kind, got $badCode" }

    Write-Host "`n================================================================="
    Write-Host "Phase 6B Full-Stack Live Verification PASSED!"
    Write-Host "================================================================="

} finally {
    if ($backendProc -and -not $backendProc.HasExited) {
        Stop-Process -Id $backendProc.Id -Force -ErrorAction SilentlyContinue
    }
    if ($frontendProc -and -not $frontendProc.HasExited) {
        taskkill /PID $frontendProc.Id /T /F 2>$null | Out-Null
    }
    Remove-Item -Force -ErrorAction SilentlyContinue $TempRef, $TempTest, $TempLayout, $TempGt, $PayloadFile, $TempOut
}
