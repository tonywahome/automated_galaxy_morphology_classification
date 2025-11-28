Write-Host 'Galaxy AI Deployment Setup' -ForegroundColor Cyan
Write-Host ''

# Check model in application/models
if (Test-Path 'application\models\galaxyai_model.h5') {
    Write-Host 'OK: Model found in application\models' -ForegroundColor Green
    if (-not (Test-Path 'models')) { New-Item -ItemType Directory -Path 'models' | Out-Null }
    Copy-Item 'application\models\galaxyai_model.h5' 'models\' -Force
    Write-Host 'OK: Model copied to models directory' -ForegroundColor Green
} elseif (Test-Path 'models\galaxyai_model.h5') {
    Write-Host 'OK: Model already in models directory' -ForegroundColor Green
} else {
    Write-Host 'ERROR: Model not found' -ForegroundColor Red
}

Write-Host ''
Write-Host 'Setup complete. Run: .\quick_start.ps1' -ForegroundColor Yellow
