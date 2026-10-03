param(
    [switch]$Fresh,
    [string]$ShuttleResults = "",
    [string]$PlayerResults = ""
)
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$env:TEMP = Join-Path $projectRoot "working"
$env:TMP = $env:TEMP
$env:PIP_CACHE_DIR = Join-Path $projectRoot "working\pip-cache"
$env:YOLO_CONFIG_DIR = Join-Path $projectRoot "working\ultralytics"
$env:TORCH_HOME = Join-Path $projectRoot "working\torch"
$env:MPLCONFIGDIR = Join-Path $projectRoot "working\matplotlib"
$env:XDG_CACHE_HOME = Join-Path $projectRoot "working\cache"
$env:PYTHONDONTWRITEBYTECODE = "1"
New-Item -ItemType Directory -Path $env:TEMP -Force | Out-Null
$pythonExecutable = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonExecutable -PathType Leaf)) { throw "Run scripts\setup.ps1 first." }
$courtCalibration = Join-Path $projectRoot "configs\sample_court.json"
$playerRoi = Join-Path $projectRoot "configs\sample_player_roi.json"
if (-not (Test-Path -LiteralPath $courtCalibration -PathType Leaf)) { throw "Missing sample court calibration: $courtCalibration" }
if (-not (Test-Path -LiteralPath $playerRoi -PathType Leaf)) { throw "Missing sample player ROI: $playerRoi" }
foreach ($downloadScript in @("download_models.py", "download_pose_model.py", "download_sample.py")) {
    & $pythonExecutable (Join-Path $PSScriptRoot $downloadScript)
    if ($LASTEXITCODE -ne 0) { throw "Asset verification failed: $downloadScript" }
}
function Resolve-ObservationCache([string]$requested, [string]$fileName) {
    if (-not $requested) { $requested = Join-Path $projectRoot "results\m2-verified-sample" }
    if (Test-Path -LiteralPath $requested -PathType Container) {
        $requested = Join-Path $requested $fileName
    }
    if (-not (Test-Path -LiteralPath $requested -PathType Leaf)) { return $null }
    return (Resolve-Path -LiteralPath $requested).Path
}
$sampleOutput = Join-Path $projectRoot ("results\m3_sample_" + (Get-Date -Format "yyyyMMdd_HHmmss_fff"))
$runArguments = @(
    (Join-Path $projectRoot "run.py"), "--milestone", "m3",
    "--video", (Join-Path $projectRoot "data\sample.mp4"),
    "--court-calibration", $courtCalibration,
    "--player-roi", $playerRoi, "--players-mode", "singles",
    "--device", "cpu", "--threads", "2", "--out", $sampleOutput
)
if (-not $Fresh) {
    $shuttleCache = Resolve-ObservationCache $ShuttleResults "shuttle.json"
    $playerCache = Resolve-ObservationCache $PlayerResults "players.json"
    if ($shuttleCache -and $playerCache) {
        $runArguments += @("--shuttle-results", $shuttleCache, "--player-results", $playerCache)
        Write-Host "Reusing shuttle and player observations after the CLI validates source content, timing and settings."
    } else {
        Write-Host "A complete shuttle/player cache pair is unavailable. Running fresh inference for both models; short footage can take several minutes on CPU."
    }
} else {
    Write-Host "Fresh inference requested. Short footage can take several minutes on CPU."
}
& $pythonExecutable @runArguments
if ($LASTEXITCODE -ne 0) { throw "M3 sample analysis failed." }
Write-Host "Combined video: $(Join-Path $sampleOutput 'annotated.mp4')"
Write-Host "Court map video: $(Join-Path $sampleOutput 'minimap.mp4')"
Write-Host "Court positions: $(Join-Path $sampleOutput 'court_positions.csv')"
