# api/routes/retrain.py
from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks
from typing import List
import os

router = APIRouter()

# Will be set by main.py
MODEL = None
CLASS_NAMES = [
    "Disturbed", "Merging", "Round Smooth", "In-between Smooth",
    "Cigar-shaped", "Barred Spiral", "Unbarred Tight Spiral",
    "Unbarred Loose Spiral", "Edge-on No Bulge", "Edge-on With Bulge"
]

# Global retraining status
retraining_status = {"is_training": False, "progress": 0, "message": ""}
stop_training_flag = False
@router.post("/upload/train-data")
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
        # Accept if content_type is None or if it's an image type
        if not image.content_type or image.content_type.startswith("image/"):
            # Check file extension as fallback
            if image.filename and any(image.filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']):
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

@router.get("/upload/status")
async def upload_status():
    """Get status of uploaded data awaiting retraining."""
    upload_dir = "data/uploads"
    
    if not os.path.exists(upload_dir):
        return {
            "total_uploaded": 0,
            "by_class": {},
            "ready_for_retraining": False
        }
    
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

@router.post("/retrain/trigger")
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
    global retraining_status, MODEL, stop_training_flag
    import tensorflow as tf
    retraining_status = {"is_training": True, "progress": 0, "message": "Starting..."}
    
    try:
        if stop_training_flag:
            raise Exception("Retraining was interrupted.")  
        
        from src.retraining import RetrainingPipeline
        pipeline = RetrainingPipeline()
        
        retraining_status["message"] = "Loading data..."
        retraining_status["progress"] = 20
        
        if stop_training_flag:
            raise Exception("Retraining was interrupted.")
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
    finally:
        stop_training_flag = False

@router.get("/retrain/status")
async def get_retraining_status():
    """Get current retraining status."""
    return retraining_status

@router.post("/retrain/stop")
async def stop_retraining():
    """Stop the current retraining process."""
    global stop_training_flag, retraining_status

    if not retraining_status["is_training"]:
        raise HTTPException(400, "No retraining in progress")

    stop_training_flag = True

    return {
        "status": "stopping",
        "message": "Training interruption requested"
    }
