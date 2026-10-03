$env:APP_ENV = "development"
Write-Host "=========================================" -ForegroundColor Green
Write-Host " Starting SONIC-REDA Backend (Port 12000)" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green

$pythonCandidates = @(
    (Join-Path $PSScriptRoot "sonic-core\.venv\Scripts\python.exe"),
    (Join-Path $PSScriptRoot ".venv\Scripts\python.exe")
)

$pythonCandidates += @(
    Get-ChildItem (Join-Path $env:LOCALAPPDATA "Python") -Directory -Filter "pythoncore-*" -ErrorAction SilentlyContinue |
        ForEach-Object { Join-Path $_.FullName "python.exe" }
)

$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
if ($pythonCommand) {
    $pythonCandidates += $pythonCommand.Source
}

$pythonPath = $null
foreach ($candidate in ($pythonCandidates | Select-Object -Unique)) {
    if (-not (Test-Path $candidate)) {
        continue
    }

    $null = & $candidate -c "import uvicorn" 2>$null
    if ($LASTEXITCODE -eq 0) {
        $pythonPath = $candidate
        break
    }
}

if (-not $pythonPath) {
    throw "No Python interpreter with uvicorn is available. Install the sonic-core dependencies and retry."
}

Set-Location (Join-Path $PSScriptRoot "sonic-core")

while ($true) {
    & $pythonPath -m uvicorn sonic.api.main:app --host 127.0.0.1 --port 12000
    $exitCode = $LASTEXITCODE
    Write-Host "[SONIC] Server stopped (exit $exitCode). Restarting in 2s... (Ctrl+C to quit)" -ForegroundColor Yellow
    Start-Sleep -Seconds 2
}
