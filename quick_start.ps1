# Quick Start - Deploy Galaxy AI Locally (Windows PowerShell)

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Galaxy AI - Quick Start Deployment" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check prerequisites
Write-Host "Checking prerequisites..." -ForegroundColor Yellow

if (!(Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "✗ Docker is not installed" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Docker installed" -ForegroundColor Green

if (!(Get-Command docker-compose -ErrorAction SilentlyContinue)) {
    # Check for docker compose (v2 syntax)
    try {
        docker compose version | Out-Null
    } catch {
        Write-Host "✗ Docker Compose is not installed" -ForegroundColor Red
        exit 1
    }
}
Write-Host "✓ Docker Compose installed" -ForegroundColor Green

# Check model file
if (!(Test-Path "models\galaxyai_model.h5")) {
    Write-Host "✗ Model file not found: models\galaxyai_model.h5" -ForegroundColor Red
    Write-Host "Please train the model first by running: python train_model.py" -ForegroundColor Yellow
    exit 1
}
Write-Host "✓ Model file exists" -ForegroundColor Green

Write-Host ""
Write-Host "Starting deployment..." -ForegroundColor Yellow
Write-Host ""

# Navigate to docker directory
Push-Location application\docker

# Stop any running containers
Write-Host "Stopping existing containers..." -ForegroundColor Yellow
docker-compose down 2>$null

# Build and start containers
Write-Host "Building and starting containers..." -ForegroundColor Yellow
docker-compose up --build -d

# Wait for health check
Write-Host ""
Write-Host "Waiting for API to be healthy..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

$MAX_RETRIES = 30
$RETRY_COUNT = 0
$healthy = $false

while ($RETRY_COUNT -lt $MAX_RETRIES) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 2 -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            Write-Host "✓ API is healthy!" -ForegroundColor Green
            $healthy = $true
            break
        }
    } catch {
        # Continue waiting
    }
    $RETRY_COUNT++
    Write-Host "Waiting... ($RETRY_COUNT/$MAX_RETRIES)" -ForegroundColor Yellow
    Start-Sleep -Seconds 2
}

if (-not $healthy) {
    Write-Host "✗ API failed to start. Check logs:" -ForegroundColor Red
    docker-compose logs api
    Pop-Location
    exit 1
}

Pop-Location

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "✓ Deployment Complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Access your application:" -ForegroundColor White
Write-Host "  API:        http://localhost:8000" -ForegroundColor Cyan
Write-Host "  API Docs:   http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "  UI:         http://localhost:8501" -ForegroundColor Cyan
Write-Host "  Health:     http://localhost:8000/health" -ForegroundColor Cyan
Write-Host ""
Write-Host "View logs:" -ForegroundColor White
Write-Host "  docker-compose logs -f api" -ForegroundColor Yellow
Write-Host "  docker-compose logs -f ui" -ForegroundColor Yellow
Write-Host ""
Write-Host "Test prediction:" -ForegroundColor White
Write-Host "  curl -X POST http://localhost:8000/predict \\" -ForegroundColor Yellow
Write-Host "    -F 'image=@path/to/galaxy.jpg'" -ForegroundColor Yellow
Write-Host ""
Write-Host "Stop deployment:" -ForegroundColor White
Write-Host "  cd application\docker; docker-compose down" -ForegroundColor Yellow
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
