param([switch]$Fresh)
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$video = Join-Path $projectRoot "data\lee_axelsen_rally.mp4"
$cache = Join-Path $projectRoot "results\m5-lee-rally-final"
$stamp = Get-Date -Format "yyyyMMdd_HHmmss_fff"
$output = Join-Path $projectRoot "results\m5_lee_rally_$stamp"
foreach ($path in @($python, $video, (Join-Path $projectRoot "configs\lee_axelsen_rally_court.json"), (Join-Path $projectRoot "configs\lee_axelsen_rally_roi.json"))) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Required file is missing: $path" }
}
$arguments = @((Join-Path $projectRoot "run.py"), "--milestone", "m5", "--video", $video,
    "--court-calibration", (Join-Path $projectRoot "configs\lee_axelsen_rally_court.json"),
    "--player-roi", (Join-Path $projectRoot "configs\lee_axelsen_rally_roi.json"),
    "--players-mode", "singles", "--pose-nms-iou", "0.5", "--device", "cpu", "--threads", "2", "--out", $output)
if (-not $Fresh -and (Test-Path -LiteralPath (Join-Path $cache "players.json")) -and (Test-Path -LiteralPath (Join-Path $cache "shuttle.json"))) {
    $arguments += @("--shuttle-results", $cache, "--player-results", $cache)
}
& $python @arguments
if ($LASTEXITCODE -ne 0) { throw "Analysis failed. See the error above and retained project diagnostics." }
Write-Host "Completed rally review: $output"