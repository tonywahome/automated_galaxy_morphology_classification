# deploy.ps1 - Quick deployment script for Windows

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Galaxy AI Deployment Script" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Select deployment option:" -ForegroundColor Yellow
Write-Host "1. Local Docker"
Write-Host "2. AWS ECS"
Write-Host "3. GCP Cloud Run"
Write-Host "4. Kubernetes"
Write-Host "5. Run Production Evaluation"
Write-Host ""
$option = Read-Host "Enter option (1-5)"

switch ($option) {
    "1" {
        Write-Host "Deploying locally with Docker..." -ForegroundColor Green
        
        # Check if Docker is installed
        try {
            $dockerVersion = docker --version
            Write-Host "Docker found: $dockerVersion" -ForegroundColor Green
        }
        catch {
            Write-Host "ERROR: Docker is not installed or not running!" -ForegroundColor Red
            Write-Host "Please install Docker Desktop from: https://www.docker.com/products/docker-desktop/" -ForegroundColor Yellow
            exit 1
        }
        
        # Check if Docker Desktop is running
        $dockerInfo = docker info 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Host "ERROR: Docker Desktop is not running!" -ForegroundColor Red
            Write-Host "Please start Docker Desktop and try again." -ForegroundColor Yellow
            exit 1
        }
        
        Set-Location ..\docker
        docker compose up --build -d
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Deployment complete!" -ForegroundColor Green
            Write-Host "API: http://localhost:8000" -ForegroundColor Cyan
            Write-Host "UI: http://localhost:8501" -ForegroundColor Cyan
        }
        else {
            Write-Host "Deployment failed. Check Docker logs for details." -ForegroundColor Red
        }
    }
    "2" {
        Write-Host "Deploying to AWS ECS..." -ForegroundColor Green
        aws cloudformation create-stack `
            --stack-name galaxyai-production `
            --template-body file://aws/cloudformation-template.yaml `
            --capabilities CAPABILITY_IAM
        Write-Host "Deployment initiated. Check AWS Console for status." -ForegroundColor Yellow
    }
    "3" {
        Write-Host "Deploying to GCP Cloud Run..." -ForegroundColor Green
        bash gcp/deploy_cloud_run.sh
    }
    "4" {
        Write-Host "Deploying to Kubernetes..." -ForegroundColor Green
        kubectl apply -f kubernetes/deployment.yaml
        Write-Host "Deployment complete!" -ForegroundColor Green
        kubectl get services -n galaxyai
    }
    "5" {
        $endpoint = Read-Host "Enter API endpoint"
        $testData = Read-Host "Enter test data directory"
        python production_evaluation.py --endpoint "$endpoint" --test-data "$testData"
    }
    default {
        Write-Host "Invalid option" -ForegroundColor Red
        exit 1
    }
}
