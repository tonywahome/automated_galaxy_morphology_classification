# Cloud Deployment & Production Evaluation

This directory contains all necessary files for deploying the Galaxy AI model to production environments and evaluating its performance.

## Quick Start

### Local Deployment (Fastest)

From the project root:

**Windows PowerShell:**

```powershell
.\quick_start.ps1
```

**Linux/Mac:**

```bash
chmod +x quick_start.sh
./quick_start.sh
```

Or use the deployment script:

```bash
./application/deployment/deploy.sh
# Select option 1: Local Docker
```

### Access Points

After deployment:

- **API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **UI Dashboard**: http://localhost:8501
- **Health Check**: http://localhost:8000/health

## Directory Structure

```
deployment/
├── deploy.sh                    # Main deployment script (all platforms)
├── production_evaluation.py     # Production model evaluation
├── DEPLOYMENT.md               # Comprehensive deployment guide
├── aws/
│   └── cloudformation-template.yaml  # AWS ECS Fargate deployment
├── gcp/
│   └── deploy_cloud_run.sh          # Google Cloud Run deployment
├── azure/
│   └── deploy_azure.sh              # Azure ACI/AKS deployment
└── kubernetes/
    └── deployment.yaml              # Generic Kubernetes deployment
```

## Deployment Options

### 1. Local Docker (Development/Testing)

- **Pros**: Fast, no cloud costs, full control
- **Cons**: Single machine, no auto-scaling
- **Setup Time**: ~2 minutes
- **Use Case**: Development, testing, demos

```bash
cd application/docker
docker-compose up --build -d
```

### 2. AWS ECS Fargate (Production)

- **Pros**: Auto-scaling, high availability, managed infrastructure
- **Cons**: Requires AWS account, setup complexity
- **Setup Time**: ~10 minutes
- **Use Case**: Production workloads, enterprise

```bash
aws cloudformation create-stack \
  --stack-name galaxyai-production \
  --template-body file://aws/cloudformation-template.yaml \
  --capabilities CAPABILITY_IAM
```

### 3. GCP Cloud Run (Serverless)

- **Pros**: Pay-per-use, instant scaling, managed HTTPS
- **Cons**: Cold starts, GCP-specific
- **Setup Time**: ~5 minutes
- **Use Case**: Variable traffic, cost optimization

```bash
cd gcp
./deploy_cloud_run.sh
```

### 4. Azure Container Instances (Simple)

- **Pros**: Fast deployment, pay-per-second
- **Cons**: No orchestration, single container
- **Setup Time**: ~3 minutes
- **Use Case**: Simple deployments, testing

```bash
cd azure
./deploy_azure.sh aci
```

### 5. Kubernetes (Any Platform)

- **Pros**: Portable, powerful orchestration
- **Cons**: Complexity, cluster management
- **Setup Time**: ~15 minutes
- **Use Case**: Multi-cloud, complex applications

```bash
kubectl apply -f kubernetes/deployment.yaml
```

## Production Evaluation

After deployment, evaluate model performance in production:

### Basic Evaluation

```bash
python production_evaluation.py \
  --endpoint http://your-api-endpoint \
  --test-data ../../../data/test/
```

### Continuous Monitoring

```bash
python production_evaluation.py \
  --endpoint http://your-api-endpoint \
  --test-data ../../../data/test/ \
  --continuous \
  --interval 300  # Every 5 minutes
```

### Output

Generates timestamped reports in `reports/production_evaluation/`:

1. **JSON Report** (`evaluation_results_YYYYMMDD_HHMMSS.json`):

   - Overall accuracy
   - Per-class metrics (precision, recall, F1)
   - Latency statistics (p50, p95, p99)
   - Confidence distribution
   - All individual predictions

2. **Visualizations** (`evaluation_plots_YYYYMMDD_HHMMSS.png`):
   - Per-class accuracy bar chart
   - Confidence distribution histogram
   - Latency distribution histogram
   - Confusion matrix heatmap

### Example Output

```
==========================================
EVALUATION RESULTS
==========================================
Total Predictions: 500
Correct: 435
Accuracy: 87.00%
Average Confidence: 85.30%
Average Latency: 287.5ms
P95 Latency: 420.3ms
P99 Latency: 598.7ms

Per-Class Performance:
  Disturbed: 82.00% (41/50)
  Merging: 86.00% (43/50)
  Round Smooth: 90.00% (45/50)
  ...
==========================================
```

## Prerequisites

### All Deployments

- Docker 20.10+
- Docker Compose 2.0+
- Trained model at `models/galaxyai_model.h5`

### AWS

```bash
# Install AWS CLI
pip install awscli

# Configure credentials
aws configure
```

### GCP

```bash
# Install gcloud SDK
# https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

### Azure

```bash
# Install Azure CLI
# https://docs.microsoft.com/en-us/cli/azure/install-azure-cli

# Login
az login
```

### Kubernetes

```bash
# Install kubectl
# https://kubernetes.io/docs/tasks/tools/

# Configure cluster access
kubectl config use-context YOUR_CLUSTER
```

## Configuration

### Environment Variables

- `MODEL_PATH`: Path to model file (default: `/app/models/galaxyai_model.h5`)
- `TF_CPP_MIN_LOG_LEVEL`: TensorFlow logging level (default: `2`)
- `API_URL`: API endpoint for UI (default: `http://localhost:8000`)

### Resource Limits

#### Docker Compose

Edit `../docker/docker-compose.yml`:

```yaml
services:
  api:
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: "2"
```

#### Kubernetes

Edit `kubernetes/deployment.yaml`:

```yaml
resources:
  limits:
    memory: "4Gi"
    cpu: "2000m"
```

### Scaling Configuration

#### AWS ECS

- Min: 2 tasks, Max: 10 tasks
- Trigger: CPU > 70%

#### GCP Cloud Run

- Min: 1 instance, Max: 10 instances
- Trigger: Automatic based on concurrency

#### Kubernetes HPA

- Min: 2 pods, Max: 10 pods
- Triggers: CPU > 70%, Memory > 80%

## Monitoring

### Health Checks

All deployments include health endpoints:

```bash
curl http://your-endpoint/health
```

Response:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-11-28T10:30:00"
}
```

### Logs

**Docker:**

```bash
docker-compose logs -f api
```

**AWS:**

```bash
aws logs tail /ecs/galaxyai --follow
```

**GCP:**

```bash
gcloud run logs read galaxyai-api --limit 50
```

**Azure:**

```bash
az container logs --resource-group galaxyai-rg --name galaxyai-api
```

**Kubernetes:**

```bash
kubectl logs -f deployment/galaxyai-api -n galaxyai
```

## Troubleshooting

### Common Issues

1. **Model not found**

   - Ensure `models/galaxyai_model.h5` exists
   - Check volume mounts in docker-compose.yml

2. **Out of memory**

   - Increase memory limit (4GB → 6GB)
   - Reduce batch size if applicable

3. **Health check failing**

   - Check container logs
   - Verify port mappings
   - Ensure model loads successfully

4. **Slow predictions**
   - Check CPU/memory allocation
   - Enable model caching
   - Consider GPU instances

### Debug Commands

```bash
# Check container status
docker ps

# View logs
docker logs galaxyai-api

# Enter container
docker exec -it galaxyai-api bash

# Test API locally
curl -X POST http://localhost:8000/predict \
  -F "image=@test_galaxy.jpg"
```

## Performance Benchmarks

### Expected Metrics

- **Latency**: 200-500ms per prediction
- **Throughput**: 10-20 requests/second (single container)
- **Accuracy**: 85-90% on Galaxy10 test set
- **Memory**: 2-3GB per container
- **CPU**: 1-2 cores per container

### Load Testing

Use Locust for load testing:

```bash
cd ../locust
locust -f locustfile.py --host http://your-endpoint
```

## Cost Estimates

### AWS ECS

- **Small** (2 tasks, 0.5 vCPU, 1GB): ~$30/month
- **Medium** (2 tasks, 1 vCPU, 2GB): ~$60/month
- **Large** (4 tasks, 2 vCPU, 4GB): ~$240/month

### GCP Cloud Run

- **Low Traffic** (1M requests/month): ~$10/month
- **Medium Traffic** (10M requests/month): ~$100/month
- **High Traffic** (100M requests/month): ~$1000/month

### Azure ACI

- **Small** (1 vCPU, 1.5GB): ~$35/month
- **Medium** (2 vCPU, 4GB): ~$120/month

_Estimates based on us-east-1/us-central1 regions, 24/7 operation_

## Security Considerations

1. **API Authentication**: Add API key validation
2. **HTTPS**: Use SSL certificates for production
3. **Rate Limiting**: Configure nginx rate limits
4. **Input Validation**: Validate image size/format
5. **Network Security**: Use VPC/security groups
6. **Secrets Management**: Use AWS Secrets Manager, GCP Secret Manager, or Azure Key Vault

## Next Steps

1. ✅ Deploy locally for testing
2. ✅ Run production evaluation
3. ⬜ Choose cloud platform
4. ⬜ Deploy to staging environment
5. ⬜ Configure monitoring and alerts
6. ⬜ Set up CI/CD pipeline
7. ⬜ Deploy to production
8. ⬜ Monitor and iterate

## Support

- **Documentation**: See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed guide
- **Issues**: Check container logs first
- **Performance**: Run production evaluation to diagnose
- **Questions**: Review troubleshooting section

---

**Last Updated**: November 28, 2025  
**Version**: 1.0.0  
**Maintainer**: Galaxy AI Team
