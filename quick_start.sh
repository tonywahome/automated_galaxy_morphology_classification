#!/bin/bash
# Quick Start - Deploy Galaxy AI Locally

echo "=========================================="
echo "Galaxy AI - Quick Start Deployment"
echo "=========================================="
echo ""

# Check prerequisites
echo "Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    echo "✗ Docker is not installed"
    exit 1
fi
echo "✓ Docker installed"

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "✗ Docker Compose is not installed"
    exit 1
fi
echo "✓ Docker Compose installed"

# Check model file
if [ ! -f "models/galaxyai_model.h5" ]; then
    echo "✗ Model file not found: models/galaxyai_model.h5"
    echo "Please train the model first by running: python train_model.py"
    exit 1
fi
echo "✓ Model file exists"

echo ""
echo "Starting deployment..."
echo ""

# Navigate to docker directory
cd application/docker

# Stop any running containers
echo "Stopping existing containers..."
docker-compose down 2>/dev/null || true

# Build and start containers
echo "Building and starting containers..."
docker-compose up --build -d

# Wait for health check
echo ""
echo "Waiting for API to be healthy..."
sleep 10

MAX_RETRIES=30
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "✓ API is healthy!"
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "Waiting... ($RETRY_COUNT/$MAX_RETRIES)"
    sleep 2
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "✗ API failed to start. Check logs:"
    docker-compose logs api
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ Deployment Complete!"
echo "=========================================="
echo ""
echo "Access your application:"
echo "  API:        http://localhost:8000"
echo "  API Docs:   http://localhost:8000/docs"
echo "  UI:         http://localhost:8501"
echo "  Health:     http://localhost:8000/health"
echo ""
echo "View logs:"
echo "  docker-compose logs -f api"
echo "  docker-compose logs -f ui"
echo ""
echo "Test prediction:"
echo "  curl -X POST http://localhost:8000/predict \\"
echo "    -F 'image=@path/to/galaxy.jpg'"
echo ""
echo "Stop deployment:"
echo "  docker-compose down"
echo ""
echo "=========================================="
