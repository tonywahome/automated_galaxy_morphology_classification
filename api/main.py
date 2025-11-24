# api/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import tensorflow as tf
import os
import json
from datetime import datetime
import time

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
    print("Model loaded successfully!")

# === HEALTH & UPTIME ENDPOINTS ===

@app.get("/health")
async def health_check():
    """Health check and uptime information."""
    uptime = (datetime.now() - START_TIME).total_seconds()
    return {
        "status": "healthy",
        "uptime_seconds": uptime,
        "uptime_formatted": f"{uptime // 3600:.0f}h {(uptime % 3600) // 60:.0f}m",
        "model_loaded": MODEL is not None,
        "model_load_time": MODEL_LOAD_TIME.isoformat() if MODEL_LOAD_TIME else None
    }

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

# === PREDICTION ENDPOINTS ===

@app.post("/predict")
async def predict_single(image: UploadFile = File(...)):
    """Predict galaxy class from a single image."""
    start_time = time.time()
    
    if not image.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    
    contents = await image.read()
    
    # Preprocess
    from src.prediction import preprocess_image
    processed = preprocess_image(contents)
    
    # Predict
    predictions = MODEL.predict(processed, verbose=0)[0]
    class_id = int(np.argmax(predictions))
    
    latency = (time.time() - start_time) * 1000
    
    return {
        "prediction": {
            "class_id": class_id,
            "class_name": CLASS_NAMES[class_id],
            "confidence": float(predictions[class_id])
        },
        "probabilities": {
            CLASS_NAMES[i]: float(predictions[i]) 
            for i in range(len(CLASS_NAMES))
        },
        "latency_ms": latency
    }

@app.post("/predict/batch")
async def predict_batch(images: List[UploadFile] = File(...)):
    """Predict galaxy classes for multiple images."""
    results = []
    total_start = time.time()
    
    for image in images:
        contents = await image.read()
        from src.prediction import preprocess_image
        processed = preprocess_image(contents)
        predictions = MODEL.predict(processed, verbose=0)[0]
        class_id = int(np.argmax(predictions))
        
        results.append({
            "filename": image.filename,
            "class_id": class_id,
            "class_name": CLASS_NAMES[class_id],
            "confidence": float(predictions[class_id])
        })
    
    return {
        "predictions": results,
        "total_images": len(results),
        "total_latency_ms": (time.time() - total_start) * 1000
    }

# === DATA UPLOAD ENDPOINTS ===

@app.post("/upload/train-data")
async def upload_training_data(
    class_id: int,
    images: List[UploadFile] = File(...)
):
    """Upload bulk images for a specific class (for retraining)."""
    if class_id < 0 or class_id >= len(CLASS_NAMES):
        raise HTTPException(400, f"Invalid class_id. Must be 0-{len(CLASS_NAMES)-1}")
    
    upload_dir = f"data/uploads/class_{class_id}"
    os.makedirs(upload_dir, exist_ok=True)
    
    saved_files = []
    for image in images:
        if image.content_type.startswith("image/"):
            file_path = os.path.join(upload_dir, image.filename)
            with open(file_path, "wb") as f:
                f.write(await image.read())
            saved_files.append(image.filename)
    
    return {
        "status": "success",
        "class_id": class_id,
        "class_name": CLASS_NAMES[class_id],
        "files_saved": len(saved_files),
        "filenames": saved_files
    }

@app.get("/upload/status")
async def upload_status():
    """Get status of uploaded data awaiting retraining."""
    upload_dir = "data/uploads"
    status = {}
    total = 0
    
    for class_folder in os.listdir(upload_dir):
        class_path = os.path.join(upload_dir, class_folder)
        if os.path.isdir(class_path):
            count = len([f for f in os.listdir(class_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
            status[class_folder] = count
            total += count
    
    return {
        "total_uploaded": total,
        "by_class": status,
        "ready_for_retraining": total > 0
    }

# === RETRAINING ENDPOINTS ===

retraining_status = {"is_training": False, "progress": 0, "message": ""}

@app.post("/retrain/trigger")
async def trigger_retraining(background_tasks: BackgroundTasks, epochs: int = 30):
    """Trigger model retraining with uploaded data."""
    global retraining_status
    
    if retraining_status["is_training"]:
        raise HTTPException(400, "Retraining already in progress")
    
    # Check if there's data to train on
    upload_status_data = await upload_status()
    if upload_status_data["total_uploaded"] == 0:
        raise HTTPException(400, "No uploaded data found for retraining")
    
    background_tasks.add_task(run_retraining, epochs)
    
    return {
        "status": "started",
        "message": f"Retraining initiated with {epochs} epochs",
        "data_samples": upload_status_data["total_uploaded"]
    }

async def run_retraining(epochs: int):
    """Background task for retraining."""
    global retraining_status, MODEL
    
    retraining_status = {"is_training": True, "progress": 0, "message": "Starting..."}
    
    try:
        from src.retraining import RetrainingPipeline
        pipeline = RetrainingPipeline()
        
        retraining_status["message"] = "Loading data..."
        retraining_status["progress"] = 20
        
        result = pipeline.retrain(epochs=epochs)
        
        retraining_status["progress"] = 90
        retraining_status["message"] = "Reloading model..."
        
        # Reload the updated model
        MODEL = tf.keras.models.load_model("models/galaxai_model.h5")
        
        retraining_status = {
            "is_training": False,
            "progress": 100,
            "message": "Complete",
            "result": result
        }
        
    except Exception as e:
        retraining_status = {
            "is_training": False,
            "progress": 0,
            "message": f"Error: {str(e)}"
        }

@app.get("/retrain/status")
async def get_retraining_status():
    """Get current retraining status."""
    return retraining_status

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