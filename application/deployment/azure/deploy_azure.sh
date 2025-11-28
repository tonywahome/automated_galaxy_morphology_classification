#!/bin/bash
# deployment/azure/deploy_azure.sh
# Deploy Galaxy AI model to Azure Container Instances or Azure Kubernetes Service

set -e

# Configuration
RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:-galaxyai-rg}"
LOCATION="${AZURE_LOCATION:-eastus}"
ACR_NAME="${AZURE_ACR_NAME:-galaxyaiacr}"
IMAGE_NAME="galaxyai-api"
DEPLOYMENT_TYPE="${1:-aci}"  # aci or aks

echo "=========================================="
echo "Deploying Galaxy AI to Azure"
echo "=========================================="
echo "Deployment Type: $DEPLOYMENT_TYPE"
echo "Resource Group: $RESOURCE_GROUP"
echo "Location: $LOCATION"
echo ""

# Create resource group
echo "Creating resource group..."
az group create \
  --name $RESOURCE_GROUP \
  --location $LOCATION

# Create Azure Container Registry
echo "Creating Azure Container Registry..."
az acr create \
  --resource-group $RESOURCE_GROUP \
  --name $ACR_NAME \
  --sku Basic \
  --admin-enabled true

# Login to ACR
echo "Logging in to ACR..."
az acr login --name $ACR_NAME

# Build and push image
echo "Building and pushing container image..."
cd ../..
az acr build \
  --registry $ACR_NAME \
  --image $IMAGE_NAME:latest \
  --file application/docker/Dockerfile.api \
  .

# Get ACR credentials
ACR_LOGIN_SERVER=$(az acr show --name $ACR_NAME --query loginServer --output tsv)
ACR_USERNAME=$(az acr credential show --name $ACR_NAME --query username --output tsv)
ACR_PASSWORD=$(az acr credential show --name $ACR_NAME --query "passwords[0].value" --output tsv)

if [ "$DEPLOYMENT_TYPE" == "aci" ]; then
  echo "Deploying to Azure Container Instances..."
  
  az container create \
    --resource-group $RESOURCE_GROUP \
    --name galaxyai-api \
    --image $ACR_LOGIN_SERVER/$IMAGE_NAME:latest \
    --registry-login-server $ACR_LOGIN_SERVER \
    --registry-username $ACR_USERNAME \
    --registry-password $ACR_PASSWORD \
    --dns-name-label galaxyai-$RANDOM \
    --ports 8000 \
    --cpu 2 \
    --memory 4 \
    --environment-variables \
      TF_CPP_MIN_LOG_LEVEL=2 \
      MODEL_PATH=/app/models/galaxyai_model.h5
  
  # Get FQDN
  FQDN=$(az container show \
    --resource-group $RESOURCE_GROUP \
    --name galaxyai-api \
    --query ipAddress.fqdn \
    --output tsv)
  
  echo "=========================================="
  echo "✓ Deployment complete!"
  echo "=========================================="
  echo "API Endpoint: http://$FQDN:8000"
  echo "API Docs: http://$FQDN:8000/docs"
  echo "Health Check: http://$FQDN:8000/health"
  
  # Test deployment
  echo ""
  echo "Testing deployment..."
  sleep 10
  if curl -sf http://$FQDN:8000/health > /dev/null; then
    echo "✓ Health check passed"
  else
    echo "✗ Health check failed"
    az container logs --resource-group $RESOURCE_GROUP --name galaxyai-api
  fi

elif [ "$DEPLOYMENT_TYPE" == "aks" ]; then
  echo "Deploying to Azure Kubernetes Service..."
  
  AKS_CLUSTER="galaxyai-aks"
  
  # Create AKS cluster
  az aks create \
    --resource-group $RESOURCE_GROUP \
    --name $AKS_CLUSTER \
    --node-count 2 \
    --node-vm-size Standard_D4s_v3 \
    --enable-addons monitoring \
    --generate-ssh-keys \
    --attach-acr $ACR_NAME
  
  # Get AKS credentials
  az aks get-credentials \
    --resource-group $RESOURCE_GROUP \
    --name $AKS_CLUSTER \
    --overwrite-existing
  
  # Update Kubernetes manifest with ACR image
  sed "s|yourusername/galaxyai-api:latest|$ACR_LOGIN_SERVER/$IMAGE_NAME:latest|g" \
    ../kubernetes/deployment.yaml > /tmp/deployment.yaml
  
  # Deploy to AKS
  kubectl apply -f /tmp/deployment.yaml
  
  echo "=========================================="
  echo "✓ Deployment initiated to AKS!"
  echo "=========================================="
  echo "Waiting for external IP..."
  
  # Wait for LoadBalancer IP
  kubectl wait --namespace galaxyai \
    --for=condition=ready pod \
    --selector=app=galaxyai \
    --timeout=300s
  
  EXTERNAL_IP=$(kubectl get service galaxyai-api-service \
    --namespace galaxyai \
    --output jsonpath='{.status.loadBalancer.ingress[0].ip}')
  
  echo "API Endpoint: http://$EXTERNAL_IP"
  echo "API Docs: http://$EXTERNAL_IP/docs"
  echo "Health Check: http://$EXTERNAL_IP/health"

else
  echo "Invalid deployment type. Use 'aci' or 'aks'"
  exit 1
fi

echo ""
echo "Deployment complete!"
