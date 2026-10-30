$env:APP_ENV = "development"
Write-Host "=========================================" -ForegroundColor Green
Write-Host " Starting SONIC-REDA Backend (Port 12000)" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green

$pythonPath = Join-Path $PSScriptRoot "sonic-core\.venv\Scripts\python.exe"
if (-not (Test-Path $pythonPath)) {
    $pythonCommand = Get-Command python -ErrorAction Stop
    $pythonPath = $pythonCommand.Source
}

Set-Location (Join-Path $PSScriptRoot "sonic-core")

while ($true) {
    & $pythonPath -m uvicorn sonic.api.main:app --host 127.0.0.1 --port 12000
    $exitCode = $LASTEXITCODE
    Write-Host "[SONIC] Server stopped (exit $exitCode). Restarting in 2s... (Ctrl+C to quit)" -ForegroundColor Yellow
    Start-Sleep -Seconds 2
}
