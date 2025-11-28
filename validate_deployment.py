#!/usr/bin/env python3
"""
Test script to validate deployment setup
"""

import os
import sys
from pathlib import Path
import subprocess
import json

def check_file(path, description):
    """Check if a file exists."""
    if Path(path).exists():
        print(f"✓ {description}: {path}")
        return True
    else:
        print(f"✗ {description} missing: {path}")
        return False

def check_directory(path, description):
    """Check if a directory exists."""
    if Path(path).exists() and Path(path).is_dir():
        print(f"✓ {description}: {path}")
        return True
    else:
        print(f"✗ {description} missing: {path}")
        return False

def check_command(cmd, description):
    """Check if a command is available."""
    try:
        subprocess.run([cmd, "--version"], capture_output=True, check=False)
        print(f"✓ {description} installed")
        return True
    except FileNotFoundError:
        print(f"✗ {description} not found")
        return False

def main():
    print("=" * 60)
    print("Galaxy AI Deployment Setup Validation")
    print("=" * 60)
    print()
    
    all_ok = True
    
    # Check prerequisites
    print("Checking prerequisites...")
    print("-" * 60)
    all_ok &= check_command("docker", "Docker")
    all_ok &= check_command("docker-compose", "Docker Compose")
    print()
    
    # Check model
    print("Checking model...")
    print("-" * 60)
    all_ok &= check_file("models/galaxyai_model.h5", "Trained model")
    if Path("models/galaxyai_model.h5").exists():
        size_mb = Path("models/galaxyai_model.h5").stat().st_size / (1024 * 1024)
        print(f"  Model size: {size_mb:.1f} MB")
    print()
    
    # Check deployment files
    print("Checking deployment files...")
    print("-" * 60)
    all_ok &= check_file("application/docker/Dockerfile.api", "API Dockerfile")
    all_ok &= check_file("application/docker/Dockerfile.ui", "UI Dockerfile")
    all_ok &= check_file("application/docker/docker-compose.yml", "Docker Compose")
    all_ok &= check_file("application/docker/nginx.conf", "Nginx config")
    print()
    
    # Check cloud deployment files
    print("Checking cloud deployment files...")
    print("-" * 60)
    all_ok &= check_file("application/deployment/deploy.sh", "Main deployment script")
    all_ok &= check_file("application/deployment/aws/cloudformation-template.yaml", "AWS CloudFormation")
    all_ok &= check_file("application/deployment/gcp/deploy_cloud_run.sh", "GCP Cloud Run script")
    all_ok &= check_file("application/deployment/azure/deploy_azure.sh", "Azure deployment script")
    all_ok &= check_file("application/deployment/kubernetes/deployment.yaml", "Kubernetes manifests")
    print()
    
    # Check evaluation
    print("Checking evaluation setup...")
    print("-" * 60)
    all_ok &= check_file("application/deployment/production_evaluation.py", "Production evaluation script")
    all_ok &= check_directory("data/test", "Test data directory")
    
    # Count test images
    test_dir = Path("data/test")
    if test_dir.exists():
        test_images = sum(1 for _ in test_dir.rglob("*.jpg")) + sum(1 for _ in test_dir.rglob("*.png"))
        print(f"  Test images found: {test_images}")
        if test_images == 0:
            print(f"  ⚠ Warning: No test images found in data/test/")
    print()
    
    # Check documentation
    print("Checking documentation...")
    print("-" * 60)
    all_ok &= check_file("application/deployment/DEPLOYMENT.md", "Deployment guide")
    all_ok &= check_file("application/deployment/README.md", "Deployment README")
    all_ok &= check_file("quick_start.sh", "Quick start script (Linux)")
    all_ok &= check_file("quick_start.ps1", "Quick start script (Windows)")
    print()
    
    # Summary
    print("=" * 60)
    if all_ok:
        print("✓ All checks passed! Ready to deploy.")
        print()
        print("Next steps:")
        print("  1. Run local deployment:")
        print("     Windows: .\\quick_start.ps1")
        print("     Linux:   ./quick_start.sh")
        print()
        print("  2. Or use the deployment script:")
        print("     ./application/deployment/deploy.sh")
        print()
        print("  3. After deployment, run evaluation:")
        print("     python application/deployment/production_evaluation.py \\")
        print("       --endpoint http://localhost:8000 \\")
        print("       --test-data data/test/")
        return 0
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
