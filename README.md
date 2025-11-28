# GalaxyAI - Automated Galaxy Morphology Classification System

This repository contains an end-to-end Machine Learning pipeline for classifying galaxy images into morphological categories, with cloud deployment, monitoring, and retraining capabilities. [cite: 995, 997, 998]

## 1. Project Overview

The GalaxAI project is an MLOps demonstration in the field of astrophysics, focused on automated image classification.

| Field            | Description                                                                                                                                                                                  |
| :--------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Project Name** | Galaxy AI - Automated Galaxy Morphology Classification System [cite: 997]                                                                                                                    |
| **Objective**    | Demonstrate an end-to-end Machine Learning pipeline for classifying galaxy images into morphological categories, with cloud deployment, monitoring, and retraining capabilities. [cite: 998] |
| **Domain**       | Astronomy / Astrophysics [cite: 999]                                                                                                                                                         |
| **Data Type**    | RGB Galaxy Images (Non-tabular) [cite: 1000]                                                                                                                                                 |
| **Model Type**   | Multi-class Classification (10 classes) [cite: 1001]                                                                                                                                         |

---

## 2. Problem Statement

Galaxy morphology classification is fundamental to understanding galaxy formation and evolution. Traditionally, this is a labor-intensive task performed by human volunteers (e.g., via Galaxy Zoo). This project automates galaxy classification using deep learning, enabling rapid, consistent, and scalable morphological analysis through a complete MLOps pipeline. [cite: 1003, 1004]

### Classification Categories (10 Classes)

The system classifies galaxies into one of 10 categories, each represented by a class ID:

| Class | Label                     | Description                                             |
| :---: | :------------------------ | :------------------------------------------------------ |
| **0** | **Disturbed**             | Galaxies showing gravitational disturbance [cite: 1006] |
| **1** | **Merging**               | Two or more galaxies merging [cite: 1006]               |
| **2** | **Round Smooth**          | Elliptical with round, smooth appearance [cite: 1006]   |
| **3** | **In-between Smooth**     | Intermediate roundness elliptical [cite: 1006]          |
| **4** | **Cigar-Shaped**          | Elongated elliptical galaxies [cite: 1006]              |
| **5** | **Barred Spiral**         | Spiral with central bar structure [cite: 1006]          |
| **6** | **Unbarred Tight Spiral** | Tightly wound spiral arms, no bar [cite: 1006]          |
| **7** | **Unbarred Loose Spiral** | Loosely wound spiral arms, no bar [cite: 1006]          |
| **8** | **Edge-on No Bulge**      | Disk galaxy edge-on, no bulge [cite: 1006]              |
| **9** | **Edge-on With Bulge**    | Disk galaxy edge-on, visible bulge [cite: 1006]         |

---

## 3. Data Sources

### Primary Dataset: Galaxy10 DECaLS

| Specification    | Value                                                               |
| :--------------- | :------------------------------------------------------------------ |
| **Source**       | astroNN Galaxy10 DECaLS Dataset [cite: 1009]                        |
| **URL**          | https://astronn.readthedocs.io/en/latest/galaxy10.html [cite: 1010] |
| **Image Size**   | [256×256 pixels (RGB) [cite: 1012]                                  |
| **Format**       | [HDF5 file [cite: 1013]                                             |
| **Total Images** | [17,736 [cite: 1014]                                                |
| **Classes**      | 10 morphological categories [cite: 1015]                            |

### Data Access Code

The dataset can be loaded directly using the `astroNN` library in Python: [cite: 1034]

````python
from astroNN.datasets import galaxy10
images, labels = galaxy10.load_data()
# images.shape = (17736, 256, 256, 3)
# labels.shape = (17736,)
``` [cite: 1036, 1037, 1038]
````

---

## 4. Quick Start Guide

### Prerequisites

Before setting up the project, ensure you have the following installed:

| Requirement        | Version | Installation                                                |
| :----------------- | :------ | :---------------------------------------------------------- |
| **Python**         | 3.8+    | [Download](https://www.python.org/downloads/)               |
| **Docker Desktop** | Latest  | [Download](https://www.docker.com/products/docker-desktop/) |
| **Git**            | Latest  | [Download](https://git-scm.com/downloads)                   |

### Option 1: Local Development Setup (Without Docker)

#### Step 1: Clone the Repository

```powershell
git clone https://github.com/tonywahome/automated_galaxy_morphology_classification.git
cd automated_galaxy_morphology_classification
```

#### Step 2: Create Virtual Environment

```powershell
# Create virtual environment
python -m venv galaxy_env

# Activate virtual environment (Windows PowerShell)
.\galaxy_env\Scripts\Activate.ps1

# Or for Command Prompt
.\galaxy_env\Scripts\activate.bat
```

#### Step 3: Install Dependencies

```powershell
# Install main dependencies
pip install -r requirements.txt

# Install API dependencies
pip install -r api/requirements.txt

# Install UI dependencies
pip install -r ui/requirements.txt
```

#### Step 4: Verify Model File

Ensure the trained model exists at `models/galaxyai_model.h5`. If not present, you need to train the model first using the notebook in `notebook/galaxy_AI.ipynb`.

```powershell
# Check if model exists
Test-Path models/galaxyai_model.h5
```

#### Step 5: Start the API Server

Open a **new terminal** and run:

```powershell
# Activate environment
.\galaxy_env\Scripts\Activate.ps1

# Start FastAPI server
cd api
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

You should see output like:

```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

#### Step 6: Test the API

Open another terminal and test the API health endpoint:

```powershell
# Test API health
curl http://localhost:8000/health

# Expected response:
# {"status":"healthy","uptime_formatted":"0:00:15","model_load_time":"2025-11-28T..."}
```

Or open in browser: http://localhost:8000/docs for interactive API documentation.

#### Step 7: Start the UI

Open a **new terminal** (keep the API running) and run:

```powershell
# Activate environment
.\galaxy_env\Scripts\Activate.ps1

# Start Streamlit UI
streamlit run ui/app.py
```

The UI will automatically open in your browser at http://localhost:8501

#### Step 8: Verify Everything is Running

| Service      | URL                        | Status Check                 |
| :----------- | :------------------------- | :--------------------------- |
| **API**      | http://localhost:8000      | http://localhost:8000/health |
| **API Docs** | http://localhost:8000/docs | Interactive Swagger UI       |
| **UI**       | http://localhost:8501      | Main application interface   |

### Option 2: Docker Deployment (Recommended for Production)

#### Prerequisites

- Docker Desktop installed and running
- At least 4GB RAM available
- Port 8000 and 8501 available

#### Step 1: Navigate to Docker Directory

```powershell
cd application/docker
```

#### Step 2: Start with Docker Compose

```powershell
# Build and start all services
docker compose up --build -d

# Check container status
docker compose ps

# View logs
docker compose logs -f
```

#### Step 3: Access the Services

- **API:** http://localhost:8000
- **UI:** http://localhost:8501
- **API Docs:** http://localhost:8000/docs

#### Step 4: Stop Services

```powershell
# Stop all containers
docker compose down

# Stop and remove volumes
docker compose down -v
```

### Option 3: One-Click Deployment

Use the automated deployment script:

#### Windows (PowerShell):

```powershell
.\quick_start.ps1
```

#### Linux/Mac (Bash):

```bash
chmod +x quick_start.sh
./quick_start.sh
```

---

## 5. Using the Application

### Web UI Features

1. **Upload Galaxy Image**: Click "Browse files" to upload a galaxy image (JPG, PNG, JPEG)
2. **View Prediction**: See the predicted galaxy morphology class and confidence scores
3. **Analyze Results**: View probability distribution across all 10 classes
4. **Model Status**: Check API health, uptime, and model information in the sidebar

### API Usage

#### Health Check

```bash
curl http://localhost:8000/health
```

#### Predict Galaxy Class

```python
import requests

# Upload image for prediction
url = "http://localhost:8000/predict"
files = {"file": open("galaxy_image.jpg", "rb")}
response = requests.post(url, files=files)

print(response.json())
# Output: {"predicted_class": 5, "class_name": "Barred Spiral", "confidence": 0.89, ...}
```

#### Get Model Information

```bash
curl http://localhost:8000/model/info
```

---

## 6. Troubleshooting

### API Not Starting

**Error:** `ModuleNotFoundError: No module named 'fastapi'`

**Solution:**

```powershell
pip install fastapi uvicorn tensorflow pillow numpy
```

### UI Not Connecting to API

**Error:** `API Unreachable` in sidebar

**Solutions:**

1. Verify API is running: `curl http://localhost:8000/health`
2. Check if port 8000 is in use: `netstat -ano | findstr :8000`
3. Ensure firewall allows local connections
4. Restart API server

### Model File Not Found

**Error:** `FileNotFoundError: [Errno 2] No such file or directory: 'models/galaxyai_model.h5'`

**Solution:**

1. Check if model file exists
2. Train model using `notebook/galaxy_AI.ipynb`
3. Ensure you're running from project root directory

### Docker Issues

**Error:** `docker: command not found`

**Solution:**

1. Install Docker Desktop
2. Start Docker Desktop application
3. Wait for Docker daemon to fully start
4. Verify: `docker --version`

**Error:** `Cannot connect to Docker daemon`

**Solution:**

1. Start Docker Desktop
2. Check Docker is running in system tray
3. Restart Docker Desktop if needed

### Port Already in Use

**Error:** `Address already in use`

**Solution:**

```powershell
# Find process using port 8000
netstat -ano | findstr :8000

# Kill the process (replace PID with actual process ID)
taskkill /PID <PID> /F

# Or use different ports
uvicorn main:app --port 8001
streamlit run ui/app.py --server.port 8502
```

---

## 7. Next Steps

- **Deploy to Cloud**: See `application/deployment/` for AWS, GCP, Azure, and Kubernetes deployment options
- **Load Testing**: Run performance tests using Locust in `application/locust/`
- **Model Retraining**: Use the `/retrain` API endpoint or notebook for retraining
- **Monitoring**: Set up production monitoring and logging

For detailed deployment instructions, see [application/deployment/README.md](application/deployment/README.md)
