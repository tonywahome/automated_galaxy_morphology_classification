#!/bin/bash
# deployment/gcp/deploy_cloud_run.sh
# Deploy Galaxy AI model to Google Cloud Run

set -e

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-your-gcp-project-id}"
REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="galaxyai-api"
IMAGE_NAME="gcr.io/$PROJECT_ID/$SERVICE_NAME"

echo "=========================================="
echo "Deploying Galaxy AI to GCP Cloud Run"
echo "=========================================="
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Service: $SERVICE_NAME"
echo ""

# Set project
echo "Setting GCP project..."
gcloud config set project $PROJECT_ID

# Build and push container
echo "Building container image..."
cd ../..
gcloud builds submit \
  --tag $IMAGE_NAME:latest \
  --timeout=20m \
  --machine-type=n1-highcpu-8 \
  --dockerfile=application/docker/Dockerfile.api \
  .

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image $IMAGE_NAME:latest \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --min-instances 1 \
  --max-instances 10 \
  --timeout 300 \
  --concurrency 80 \
  --cpu-throttling \
  --set-env-vars="TF_CPP_MIN_LOG_LEVEL=2,MODEL_PATH=/app/models/galaxyai_model.h5" \
  --port 8000 \
  --execution-environment gen2

# Get service URL
echo ""
echo "Retrieving service URL..."
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME \
  --platform managed \
  --region $REGION \
  --format 'value(status.url)')

echo "=========================================="
echo "✓ Deployment complete!"
echo "=========================================="
echo "Service URL: $SERVICE_URL"
echo "API Docs: $SERVICE_URL/docs"
echo "Health Check: $SERVICE_URL/health"
echo ""

# Test deployment
echo "Testing deployment..."
sleep 5
if curl -sf $SERVICE_URL/health > /dev/null; then
  echo "✓ Health check passed"
  echo ""
  echo "You can now run production evaluation:"
  echo "  python deployment/production_evaluation.py \\"
  echo "    --endpoint $SERVICE_URL \\"
  echo "    --test-data data/test/"
else
  echo "✗ Health check failed"
  echo "Check logs: gcloud run logs read $SERVICE_NAME --region $REGION"
  exit 1
fi

echo ""
echo "To test prediction:"
echo "curl -X POST $SERVICE_URL/predict -F \"image=@test_image.jpg\""