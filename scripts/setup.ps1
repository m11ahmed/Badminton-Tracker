param(
    [string]$PythonExecutable = "",
    [ValidateSet("cpu", "cuda")][string]$Device = "cpu"
)
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$env:TEMP = Join-Path $projectRoot "working"
$env:TMP = $env:TEMP
$env:PIP_CACHE_DIR = Join-Path $projectRoot "working\pip-cache"
$env:TORCH_HOME = Join-Path $projectRoot "models\torch-cache"
$env:YOLO_CONFIG_DIR = Join-Path $projectRoot "working\ultralytics"
$env:MPLCONFIGDIR = Join-Path $projectRoot "working\matplotlib"
$env:PYTHONDONTWRITEBYTECODE = "1"
New-Item -ItemType Directory -Path $env:TEMP, $env:PIP_CACHE_DIR -Force | Out-Null
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython)) {
    if (-not $PythonExecutable) {
        $bundledPython = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
        if (Test-Path -LiteralPath $bundledPython) { $PythonExecutable = $bundledPython }
        else {
            $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
            if ($pythonCommand -and $pythonCommand.Source -notlike "*WindowsApps*") { $PythonExecutable = $pythonCommand.Source }
            else { throw "Pass -PythonExecutable with the full path to an installed Python 3.10 to 3.13 interpreter." }
        }
    }
    & $PythonExecutable -m venv (Join-Path $projectRoot ".venv")
    if ($LASTEXITCODE -ne 0) { throw "Python environment creation failed." }
}
$torchIndex = if ($Device -eq "cuda") { "https://download.pytorch.org/whl/cu128" } else { "https://download.pytorch.org/whl/cpu" }
$torchVersion = if ($Device -eq "cuda") { "2.8.0+cu128" } else { "2.8.0+cpu" }
$torchvisionVersion = if ($Device -eq "cuda") { "0.23.0+cu128" } else { "0.23.0+cpu" }
& $venvPython -m pip install "torch==$torchVersion" "torchvision==$torchvisionVersion" --index-url $torchIndex
if ($LASTEXITCODE -ne 0) { throw "PyTorch installation failed." }
& $venvPython -m pip install -r (Join-Path $projectRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
& $venvPython -m pip check
if ($LASTEXITCODE -ne 0) { throw "Dependency validation failed." }
Write-Host "Setup complete. Python: $venvPython"
