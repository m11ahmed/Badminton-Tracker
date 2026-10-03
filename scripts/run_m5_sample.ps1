param([switch]$Fresh)
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$video = Join-Path $projectRoot "data\sample.mp4"
$cache = Join-Path $projectRoot "results\m3-sample"
$stamp = Get-Date -Format "yyyyMMdd_HHmmss_fff"
$output = Join-Path $projectRoot "results\m5_sample_$stamp"
$arguments = @((Join-Path $projectRoot "run.py"), "--milestone", "m5", "--video", $video,
    "--court-calibration", (Join-Path $projectRoot "configs\sample_court.json"),
    "--player-roi", (Join-Path $projectRoot "configs\sample_player_roi.json"),
    "--players-mode", "singles", "--device", "cpu", "--threads", "2", "--out", $output)
if (-not $Fresh -and (Test-Path -LiteralPath (Join-Path $cache "players.json")) -and (Test-Path -LiteralPath (Join-Path $cache "shuttle.json"))) {
    $arguments += @("--shuttle-results", $cache, "--player-results", $cache)
}
& $python @arguments
if ($LASTEXITCODE -ne 0) { throw "Analysis failed. See the error above and retained project diagnostics." }
Write-Host "Completed review: $output"