# Galaxy AI - Cloud Deployment Guide

Complete guide for deploying the Galaxy AI model to production environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Docker Deployment](#local-docker-deployment)
3. [AWS Deployment](#aws-deployment)
4. [GCP Deployment](#gcp-deployment)
5. [Azure Deployment](#azure-deployment)
6. [Kubernetes Deployment](#kubernetes-deployment)
7. [Production Evaluation](#production-evaluation)
8. [Monitoring and Scaling](#monitoring-and-scaling)
9. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Tools

- **Docker** (20.10+) and Docker Compose (2.0+)
- **Git** for version control
- **Python** 3.10+ with pip
- Cloud CLI tools (install based on target platform):
  - AWS: `aws-cli` (2.0+)
  - GCP: `gcloud` SDK
  - Azure: `azure-cli`
  - Kubernetes: `kubectl` (1.24+)

### Model Requirements

Ensure your trained model exists at:

```
models/galaxyai_model.h5
```

If not trained yet, run:

```bash
python train_model.py
```

### Build Docker Images

Before deployment, build and optionally push Docker images:

```bash
# Build API image
docker build -t yourusername/galaxyai-api:latest \
  -f application/docker/Dockerfile.api .

# Build UI image
docker build -t yourusername/galaxyai-ui:latest \
  -f application/docker/Dockerfile.ui .

# Push to Docker Hub (optional for local deployment)
docker push yourusername/galaxyai-api:latest
docker push yourusername/galaxyai-ui:latest
```

---

## Local Docker Deployment

### Quick Start

Use the deployment script:

```bash
./application/deployment/deploy.sh
# Select option 1: Local Docker
```

Or manually with Docker Compose:

```bash
cd application/docker
docker-compose up --build -d
```

### Access the Application

- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **UI**: http://localhost:8501
- **Health Check**: http://localhost:8000/health

### Test Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -F "image=@path/to/galaxy.jpg"
```

### View Logs

```bash
docker-compose logs -f api
docker-compose logs -f ui
```

### Stop Deployment

```bash
docker-compose down
```

---

## AWS Deployment

### Option 1: AWS ECS Fargate (Recommended)

#### Setup

1. Install AWS CLI and configure credentials:

```bash
aws configure
```

2. Update Docker image in CloudFormation template:

```bash
# Edit application/deployment/aws/cloudformation-template.yaml
# Update DockerImage parameter default value
```

3. Deploy stack:

```bash
aws cloudformation create-stack \
  --stack-name galaxyai-production \
  --template-body file://application/deployment/aws/cloudformation-template.yaml \
  --capabilities CAPABILITY_IAM \
  --parameters \
    ParameterKey=DockerImage,ParameterValue=yourusername/galaxyai-api:latest
```

4. Monitor deployment:

```bash
aws cloudformation describe-stacks \
  --stack-name galaxyai-production \
  --query 'Stacks[0].StackStatus'
```

5. Get API endpoint:

```bash
aws cloudformation describe-stacks \
  --stack-name galaxyai-production \
  --query 'Stacks[0].Outputs[?OutputKey==`APIEndpoint`].OutputValue' \
  --output text
```

#### Features

- **Auto-scaling**: 2-10 tasks based on CPU (70%)
- **Load Balancing**: Application Load Balancer with health checks
- **High Availability**: Multi-AZ deployment across 2 availability zones
- **Logging**: CloudWatch Logs with 7-day retention
- **Cost Optimization**: Mix of Fargate and Fargate Spot

#### Update Deployment

```bash
# Update task definition with new image
aws ecs update-service \
  --cluster galaxyai-cluster \
  --service galaxyai-service \
  --force-new-deployment
```

#### Delete Stack

```bash
aws cloudformation delete-stack --stack-name galaxyai-production
```

### Option 2: AWS Elastic Beanstalk

```bash
# Initialize EB application
eb init -p docker galaxyai-app --region us-east-1

# Create environment
eb create galaxyai-env --instance-type t3.large

# Deploy
eb deploy

# Get URL
eb status
```

### Option 3: AWS SageMaker

For managed ML inference with SageMaker endpoints. (See AWS documentation for detailed setup)

---

## GCP Deployment

### Option 1: Google Cloud Run (Recommended)

#### Setup

1. Install gcloud SDK and authenticate:

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

2. Update deployment script:

```bash
# Edit application/deployment/gcp/deploy_cloud_run.sh
# Set PROJECT_ID and REGION
export GCP_PROJECT_ID=your-project-id
export GCP_REGION=us-central1
```

3. Deploy:

```bash
cd application/deployment/gcp
chmod +x deploy_cloud_run.sh
./deploy_cloud_run.sh
```

#### Features

- **Serverless**: Pay only for actual usage
- **Auto-scaling**: 1-10 instances based on load
- **Custom Domain**: Easy HTTPS setup with managed certificates
- **Cold Start**: Minimal with gen2 execution environment
- **Memory**: 4GB per instance
- **Concurrency**: 80 requests per instance

#### Manual Deployment

```bash
# Build with Cloud Build
gcloud builds submit --tag gcr.io/PROJECT_ID/galaxyai-api

# Deploy to Cloud Run
gcloud run deploy galaxyai-api \
  --image gcr.io/PROJECT_ID/galaxyai-api:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --min-instances 1 \
  --max-instances 10
```

#### Update Deployment

```bash
# Deploy new version
gcloud run deploy galaxyai-api --image gcr.io/PROJECT_ID/galaxyai-api:latest
```

### Option 2: Google Kubernetes Engine (GKE)

```bash
# Create GKE cluster
gcloud container clusters create galaxyai-cluster \
  --num-nodes 3 \
  --machine-type n1-standard-4 \
  --region us-central1

# Get credentials
gcloud container clusters get-credentials galaxyai-cluster

# Deploy using kubectl
kubectl apply -f application/deployment/kubernetes/deployment.yaml
```

---

## Azure Deployment

### Option 1: Azure Container Instances (ACI)

#### Setup

1. Install Azure CLI and login:

```bash
az login
```

2. Deploy to ACI:

```bash
cd application/deployment/azure
chmod +x deploy_azure.sh
./deploy_azure.sh aci
```

#### Features

- **Fast Deployment**: < 60 seconds to running container
- **No Orchestration**: Simple single-container deployment
- **Pay Per Second**: Cost-effective for development/testing
- **4GB Memory**: Sufficient for model inference

### Option 2: Azure Kubernetes Service (AKS)

```bash
cd application/deployment/azure
./deploy_azure.sh aks
```

#### Features

- **Production-Ready**: Full Kubernetes orchestration
- **Auto-scaling**: HPA based on CPU/memory
- **Monitoring**: Integrated Azure Monitor
- **High Availability**: Multi-node cluster

### Option 3: Azure Machine Learning

For managed ML endpoints with Azure ML. (See Azure documentation)

---

## Kubernetes Deployment

### Generic Kubernetes (Works on any K8s cluster)

#### Prerequisites

- Kubernetes cluster (1.24+)
- `kubectl` configured
- Docker images pushed to registry

#### Deploy

1. Update image references in manifest:

```bash
# Edit application/deployment/kubernetes/deployment.yaml
# Replace: yourusername/galaxyai-api:latest
```

2. Apply manifests:

```bash
kubectl apply -f application/deployment/kubernetes/deployment.yaml
```

3. Check deployment status:

```bash
kubectl get pods -n galaxyai
kubectl get services -n galaxyai
kubectl get hpa -n galaxyai
```

4. Get external IP:

```bash
kubectl get service galaxyai-api-service -n galaxyai
```

#### Features Included

- **Namespace**: Isolated `galaxyai` namespace
- **ConfigMap**: Environment configuration
- **PVC**: Persistent storage for model
- **Deployment**: 2 replicas with rolling updates
- **Service**: LoadBalancer for external access
- **HPA**: 2-10 pods based on CPU (70%) and memory (80%)
- **PDB**: Ensures minimum availability during updates
- **Ingress**: Optional custom domain with TLS
- **Network Policy**: Security restrictions

#### Monitor

```bash
# Watch pods
kubectl get pods -n galaxyai -w

# View logs
kubectl logs -f deployment/galaxyai-api -n galaxyai

# Describe HPA
kubectl describe hpa galaxyai-api-hpa -n galaxyai
```

#### Scale Manually

```bash
kubectl scale deployment galaxyai-api -n galaxyai --replicas=5
```

#### Update Deployment

```bash
kubectl set image deployment/galaxyai-api \
  api=yourusername/galaxyai-api:v2 \
  -n galaxyai
```

---

## Production Evaluation

After deployment, evaluate model performance in production.

### Run Evaluation

```bash
python application/deployment/production_evaluation.py \
  --endpoint http://your-api-endpoint \
  --test-data data/test/ \
  --output-dir reports/production_evaluation/
```

### Parameters

- `--endpoint`: API endpoint URL (required)
- `--test-data`: Path to test dataset directory (required)
- `--output-dir`: Directory for reports (default: `reports/production_evaluation/`)
- `--batch-size`: Images per batch (default: 10)
- `--continuous`: Enable continuous monitoring mode
- `--interval`: Monitoring interval in seconds (default: 300)

### Test Data Structure

Organize test images by class:

```
data/test/
├── class_0/
│   ├── image1.jpg
│   └── image2.jpg
├── class_1/
│   ├── image1.jpg
│   └── image2.jpg
...
└── class_9/
    ├── image1.jpg
    └── image2.jpg
```

### Output Reports

The evaluation generates:

1. **JSON Report** (`evaluation_results_TIMESTAMP.json`):

   - Overall accuracy
   - Per-class metrics (precision, recall, F1)
   - Latency statistics (p50, p95, p99)
   - Confidence distribution
   - Confusion matrix

2. **Visualizations** (`evaluation_plots_TIMESTAMP.png`):
   - Confusion matrix heatmap
   - Per-class accuracy bar chart
   - Latency distribution histogram
   - Confidence distribution boxplot

### Continuous Monitoring

For long-running monitoring:

```bash
python application/deployment/production_evaluation.py \
  --endpoint http://your-api-endpoint \
  --test-data data/test/ \
  --continuous \
  --interval 300
```

This will:

- Run evaluation every 5 minutes (300 seconds)
- Generate timestamped reports
- Track performance trends over time
- Alert on performance degradation

### Quick Deploy + Evaluate

Use the deployment script:

```bash
./application/deployment/deploy.sh
# Select option 5: Run Production Evaluation
# Enter API endpoint
# Enter test data directory
```

---

## Monitoring and Scaling

### Health Checks

All deployments include `/health` endpoint:

```bash
curl http://your-api-endpoint/health
```

Response:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-11-28T10:30:00"
}
```

### Metrics to Monitor

1. **Latency**: API response time (target: < 500ms)
2. **Throughput**: Requests per second
3. **Error Rate**: Failed predictions (target: < 1%)
4. **CPU/Memory**: Resource utilization (target: < 80%)
5. **Model Accuracy**: Production prediction accuracy

### Cloud-Specific Monitoring

#### AWS CloudWatch

```bash
# View API logs
aws logs tail /ecs/galaxyai --follow

# Check metrics
aws cloudwatch get-metric-statistics \
  --namespace AWS/ECS \
  --metric-name CPUUtilization \
  --dimensions Name=ServiceName,Value=galaxyai-service
```

#### GCP Cloud Monitoring

```bash
# View logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=galaxyai-api"

# View metrics in console
gcloud console logs-viewer
```

#### Azure Monitor

```bash
# View container logs
az container logs --resource-group galaxyai-rg --name galaxyai-api

# Stream logs
az container attach --resource-group galaxyai-rg --name galaxyai-api
```

### Scaling Configuration

#### AWS ECS

- Min: 2 tasks, Max: 10 tasks
- Triggers: CPU > 70%, Memory > 80%
- Cooldown: 60s scale-out, 300s scale-in

#### GCP Cloud Run

- Min: 1 instance, Max: 10 instances
- Triggers: Request concurrency (80 req/instance)
- Automatic scaling based on traffic

#### Kubernetes HPA

- Min: 2 pods, Max: 10 pods
- Triggers: CPU > 70%, Memory > 80%
- Scale-up: 30s, Scale-down: 300s

---

## Troubleshooting

### Common Issues

#### 1. Model Loading Errors

**Symptom**: API returns "Model not loaded" or 500 errors

**Solutions**:

```bash
# Check model file exists
ls -lh models/galaxyai_model.h5

# Verify model in container
docker exec -it galaxyai-api ls -lh /app/models/

# Check container logs
docker logs galaxyai-api
```

#### 2. High Latency

**Symptom**: Predictions take > 1 second

**Solutions**:

- Increase CPU/memory allocation
- Enable model caching
- Use GPU instances (if available)
- Optimize image preprocessing

#### 3. Out of Memory

**Symptom**: Container crashes or OOMKilled

**Solutions**:

```bash
# Increase memory limit in docker-compose.yml
services:
  api:
    deploy:
      resources:
        limits:
          memory: 6G  # Increase from 4G

# For Kubernetes, update deployment.yaml
resources:
  limits:
    memory: "6Gi"
```

#### 4. Connection Refused

**Symptom**: Cannot connect to API endpoint

**Solutions**:

```bash
# Check container status
docker ps

# Check port mapping
docker port galaxyai-api

# Test locally
curl http://localhost:8000/health

# Check firewall rules (cloud deployments)
```

#### 5. Prediction Accuracy Issues

**Symptom**: Low accuracy in production vs. training

**Solutions**:

- Verify preprocessing matches training pipeline
- Check image quality and format
- Run production evaluation script
- Review model version deployed

### Debugging Commands

```bash
# Docker
docker logs -f galaxyai-api
docker exec -it galaxyai-api bash
docker stats galaxyai-api

# Kubernetes
kubectl describe pod <pod-name> -n galaxyai
kubectl logs -f <pod-name> -n galaxyai
kubectl exec -it <pod-name> -n galaxyai -- bash

# AWS ECS
aws ecs describe-tasks --cluster galaxyai-cluster --tasks <task-id>
aws logs tail /ecs/galaxyai --follow

# GCP Cloud Run
gcloud run services logs read galaxyai-api --limit 50
gcloud run revisions describe <revision> --region us-central1

# Azure
az container logs --resource-group galaxyai-rg --name galaxyai-api
az container show --resource-group galaxyai-rg --name galaxyai-api
```

### Performance Tuning

#### Optimize Docker Image

```dockerfile
# Use multi-stage build
FROM python:3.10-slim as builder
# ... build steps ...

FROM python:3.10-slim
# Copy only necessary files
```

#### Enable Model Caching

Add to API code:

```python
from functools import lru_cache

@lru_cache(maxsize=1)
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)
```

#### Batch Predictions

For high throughput, enable batch prediction endpoint.

---

## Support and Resources

### Documentation

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving)
- [Galaxy10 Dataset](http://www.astroml.org/datasets/)

### Cloud Platform Guides

- [AWS ECS Best Practices](https://docs.aws.amazon.com/AmazonECS/latest/bestpracticesguide/)
- [GCP Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Azure Container Instances](https://docs.microsoft.com/en-us/azure/container-instances/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)

### Model Information

- **Architecture**: EfficientNetV2-S with custom classification head
- **Input**: 256x256 RGB images
- **Output**: 10 galaxy morphology classes
- **Framework**: TensorFlow 2.x / Keras

### Contact

For issues or questions:

1. Check this documentation
2. Review container logs
3. Run production evaluation
4. Open GitHub issue with logs and error details

---

## Next Steps

1. **Deploy Locally**: Test with Docker Compose
2. **Run Evaluation**: Validate model performance
3. **Choose Cloud Platform**: Based on requirements
4. **Deploy to Cloud**: Follow platform-specific guide
5. **Monitor**: Set up continuous monitoring
6. **Scale**: Configure auto-scaling policies
7. **Iterate**: Update model and redeploy

---

**Last Updated**: November 28, 2025
**Version**: 1.0.0
