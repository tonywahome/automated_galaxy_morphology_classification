#!/bin/bash
# deploy.sh - Quick deployment script

echo "=========================================="
echo "Galaxy AI Deployment Script"
echo "=========================================="
echo ""
echo "Select deployment option:"
echo "1. Local Docker"
echo "2. AWS ECS"
echo "3. GCP Cloud Run"
echo "4. Kubernetes"
echo "5. Run Production Evaluation"
echo ""
read -p "Enter option (1-5): " option

case $option in
  1)
    echo "Deploying locally with Docker..."
    cd application/docker
    docker-compose up --build -d
    echo "Deployment complete!"
    echo "API: http://localhost:8000"
    echo "UI: http://localhost:8501"
    ;;
  2)
    echo "Deploying to AWS ECS..."
    aws cloudformation create-stack \
      --stack-name galaxyai-production \
      --template-body file://deployment/aws/cloudformation-template.yaml \
      --capabilities CAPABILITY_IAM
    echo "Deployment initiated. Check AWS Console for status."
    ;;
  3)
    echo "Deploying to GCP Cloud Run..."
    bash deployment/gcp/deploy_cloud_run.sh
    ;;
  4)
    echo "Deploying to Kubernetes..."
    kubectl apply -f deployment/kubernetes/deployment.yaml
    echo "Deployment complete!"
    kubectl get services -n galaxyai
    ;;
  5)
    read -p "Enter API endpoint: " endpoint
    read -p "Enter test data directory: " test_data
    python deployment/production_evaluation.py \
      --endpoint "$endpoint" \
      --test-data "$test_data"
    ;;
  *)
    echo "Invalid option"
    exit 1
    ;;
esac