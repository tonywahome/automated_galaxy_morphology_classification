# application/locust/run_multi_container_test.ps1
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "GalaxyAI Multi-Container Load Testing" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Configuration
$TEST_DURATION = "5m"
$USERS = 100
$SPAWN_RATE = 10
$RESULTS_BASE = "application\locust\results"

# Create results directory
New-Item -ItemType Directory -Force -Path $RESULTS_BASE | Out-Null

function Run-LoadTest {
    param(
        [int]$Containers,
        [string]$Host
    )
    
    $testName = "test_${Containers}_containers"
    
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "Testing with $Containers container(s)" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    
    # Start containers
    Set-Location "application\docker"
    if ($Containers -eq 1) {
        docker-compose up -d
    } else {
        $env:REPLICAS = $Containers
        docker-compose up -d --scale api=$Containers
    }
    Set-Location "..\..\"
    
    Write-Host "Waiting for services to be ready..."
    Start-Sleep -Seconds 30
    
    # Run load test
    locust -f application\locust\locustfile.py `
        --host=$Host `
        --users $USERS `
        --spawn-rate $SPAWN_RATE `
        --run-time $TEST_DURATION `
        --headless `
        --html="$RESULTS_BASE\${testName}_report.html" `
        --csv="$RESULTS_BASE\${testName}" `
        --loglevel INFO
    
    Write-Host "Test completed for $Containers container(s)" -ForegroundColor Green
    Write-Host "Results saved to $RESULTS_BASE\${testName}_*"
    Write-Host ""
    
    # Stop containers
    Set-Location "application\docker"
    docker-compose down
    Set-Location "..\..\"
    
    Write-Host "Cooling down for 15 seconds..."
    Start-Sleep -Seconds 15
}

# Test with 1 container
Run-LoadTest -Containers 1 -Host "http://localhost:8000"

# Test with 2 containers
Run-LoadTest -Containers 2 -Host "http://localhost:80"

# Test with 4 containers
Run-LoadTest -Containers 4 -Host "http://localhost:80"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "All tests completed!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Generating comparison report..."

# Generate comparison report
python application\locust\analyze_results.py $RESULTS_BASE