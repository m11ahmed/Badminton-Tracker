param([int]$Port = 8501)
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonExecutable = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonExecutable -PathType Leaf)) { throw "Project environment is missing. Run scripts\setup.ps1 first." }
if ($Port -lt 1024 -or $Port -gt 65535) { throw "Choose a port from 1024 to 65535." }
$env:TEMP = Join-Path $projectRoot "working\tmp"
$env:TMP = $env:TEMP
$env:YOLO_CONFIG_DIR = Join-Path $projectRoot "working\ultralytics"
$env:TORCH_HOME = Join-Path $projectRoot "working\torch"
$env:MPLCONFIGDIR = Join-Path $projectRoot "working\matplotlib"
$env:XDG_CACHE_HOME = Join-Path $projectRoot "working\cache"
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTHONUNBUFFERED = "1"
New-Item -ItemType Directory -Path $env:TEMP -Force | Out-Null
Set-Location -LiteralPath $projectRoot
& $pythonExecutable -c "import streamlit"
if ($LASTEXITCODE -ne 0) { throw "Streamlit is unavailable. Install the project requirements using scripts\setup.ps1." }
Write-Host "Open http://127.0.0.1:$Port in your browser. Keep this terminal open; Ctrl+C stops the demo."
& $pythonExecutable -m streamlit run (Join-Path $projectRoot "app.py") --server.address 127.0.0.1 --server.port $Port --server.headless true --server.fileWatcherType none --server.maxUploadSize 1024 --browser.gatherUsageStats false
if ($LASTEXITCODE -ne 0) { throw "The demo stopped with an error. If the port is occupied, use -Port 8502." }