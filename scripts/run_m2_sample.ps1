param([string]$ShuttleResults = "")
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$env:TEMP = Join-Path $projectRoot "working"
$env:TMP = $env:TEMP
$env:PYTHONDONTWRITEBYTECODE = "1"
$pythonExecutable = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonExecutable)) { throw "Run scripts\setup.ps1 first." }
foreach ($downloadScript in @("download_models.py", "download_pose_model.py", "download_sample.py")) {
    & $pythonExecutable (Join-Path $PSScriptRoot $downloadScript)
    if ($LASTEXITCODE -ne 0) { throw "Asset verification failed: $downloadScript" }
}
$sampleOutput = Join-Path $projectRoot ("results\m2_sample_" + (Get-Date -Format "yyyyMMdd_HHmmss_fff"))
$runArguments = @((Join-Path $projectRoot "run.py"), "--milestone", "m2", "--video", (Join-Path $projectRoot "data\sample.mp4"), "--player-roi", (Join-Path $projectRoot "configs\sample_player_roi.json"), "--players-mode", "singles", "--out", $sampleOutput)
if ($ShuttleResults) { $runArguments += @("--shuttle-results", $ShuttleResults) }
& $pythonExecutable @runArguments
if ($LASTEXITCODE -ne 0) { throw "M2 sample analysis failed." }
Write-Host "Open: $(Join-Path $sampleOutput 'annotated.mp4')"
