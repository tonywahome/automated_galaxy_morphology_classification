# api/routes/predict.py
from fastapi import APIRouter, File, UploadFile, HTTPException
from typing import List
import numpy as np
import time

router = APIRouter()

# Will be set by main.py
MODEL = None
CLASS_NAMES = [
    "Disturbed", "Merging", "Round Smooth", "In-between Smooth",
    "Cigar-shaped", "Barred Spiral", "Unbarred Tight Spiral",
    "Unbarred Loose Spiral", "Edge-on No Bulge", "Edge-on With Bulge"
]

@router.post("/predict")
async def predict_single(image: UploadFile = File(...)):
    """Predict galaxy class from a single image."""
    start_time = time.time()
    
    if image.content_type and not image.content_type.startswith("image/"):
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

@router.post("/predict/batch")
async def predict_batch(images: List[UploadFile] = File(...)):
    """Predict galaxy classes for multiple images."""
    results = []
    total_start = time.time()
    
    for img in images:
        if img.content_type and not img.content_type.startswith("image/"):
            continue
        contents = await img.read()
        from src.prediction import preprocess_image
        processed = preprocess_image(contents)
        predictions = MODEL.predict(processed, verbose=0)[0]
        class_id = int(np.argmax(predictions))
        
        results.append({
            "filename": img.filename,
            "class_id": class_id,
            "class_name": CLASS_NAMES[class_id],
            "confidence": float(predictions[class_id])
        })
    
    return {
        "predictions": results,
        "total_images": len(results),
        "total_latency_ms": (time.time() - total_start) * 1000
    }
