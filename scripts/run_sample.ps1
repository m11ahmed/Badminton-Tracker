$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$env:TEMP = Join-Path $projectRoot "working"
$env:TMP = $env:TEMP
$env:PYTHONDONTWRITEBYTECODE = "1"
$pythonExecutable = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonExecutable)) { throw "Run scripts\setup.ps1 first." }
& $pythonExecutable (Join-Path $PSScriptRoot "download_models.py")
if ($LASTEXITCODE -ne 0) { throw "Model verification failed." }
& $pythonExecutable (Join-Path $PSScriptRoot "download_sample.py")
if ($LASTEXITCODE -ne 0) { throw "Sample verification failed." }
$sampleOutput = Join-Path $projectRoot ("results\sample_" + (Get-Date -Format "yyyyMMdd_HHmmss_fff"))
& $pythonExecutable (Join-Path $projectRoot "run.py") --video (Join-Path $projectRoot "data\sample.mp4") --out $sampleOutput
if ($LASTEXITCODE -ne 0) { throw "Sample analysis failed." }
& $pythonExecutable (Join-Path $PSScriptRoot "evaluate_sample.py") --results $sampleOutput
if ($LASTEXITCODE -ne 0) { throw "Sample evaluation failed." }
Write-Host "Open: $(Join-Path $sampleOutput 'annotated.mp4')"
