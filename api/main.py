# api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
import os
import json
from datetime import datetime

# Import route modules
from api.routes import health, predict, retrain

app = FastAPI(
    title="GalaxAI API",
    description="Galaxy Morphology Classification API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
MODEL = None
MODEL_LOAD_TIME = None
START_TIME = datetime.now()

CLASS_NAMES = [
    "Disturbed", "Merging", "Round Smooth", "In-between Smooth",
    "Cigar-shaped", "Barred Spiral", "Unbarred Tight Spiral",
    "Unbarred Loose Spiral", "Edge-on No Bulge", "Edge-on With Bulge"
]

@app.on_event("startup")
async def load_model():
    global MODEL, MODEL_LOAD_TIME
    MODEL = tf.keras.models.load_model("models/galaxai_model.h5")
    MODEL_LOAD_TIME = datetime.now()
    
    # Share globals with route modules
    health.START_TIME = START_TIME
    health.MODEL = MODEL
    health.MODEL_LOAD_TIME = MODEL_LOAD_TIME
    
    predict.MODEL = MODEL
    predict.CLASS_NAMES = CLASS_NAMES
    
    retrain.MODEL = MODEL
    retrain.CLASS_NAMES = CLASS_NAMES
    
    print("Model loaded successfully!")

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(predict.router, tags=["Prediction"])
app.include_router(retrain.router, tags=["Retraining"])

# === MODEL INFO ENDPOINT ===

@app.get("/model/info")
async def model_info():
    """Get model metadata and version info."""
    metadata_path = "models/model_metadata.json"
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {"version": "1.0.0", "accuracy": "N/A"}
    
    return {
        "model_name": "GalaxAI Classifier",
        "architecture": "EfficientNetV2-S",
        "classes": len(CLASS_NAMES),
        "class_names": CLASS_NAMES,
        **metadata
    }

# === DATA VISUALIZATION ENDPOINTS ===

@app.get("/visualizations/class-distribution")
async def get_class_distribution():
    """Get class distribution data for visualization."""
    from astroNN.datasets import galaxy10
    _, labels = galaxy10.load_data()
    
    distribution = {}
    for i, name in enumerate(CLASS_NAMES):
        distribution[name] = int(np.sum(labels == i))
    
    return {"distribution": distribution}

@app.get("/visualizations/model-performance")
async def get_model_performance():
    """Get model performance metrics for visualization."""
    metadata_path = "models/model_metadata.json"
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return {"message": "No performance data available"}