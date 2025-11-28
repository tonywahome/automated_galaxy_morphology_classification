# src/prediction.py
import numpy as np
from PIL import Image
import tensorflow as tf

CLASS_NAMES = [
    "Disturbed", "Merging", "Round Smooth", "In-between Smooth",
    "Cigar-shaped", "Barred Spiral", "Unbarred Tight Spiral",
    "Unbarred Loose Spiral", "Edge-on No Bulge", "Edge-on With Bulge"
]

def preprocess_image(image_path_or_bytes, target_size=(256, 256)):
    """Preprocess a single image for prediction.
    
    IMPORTANT: This must match the training preprocessing exactly.
    The model was trained on images normalized to [0, 1] range.
    """
    if isinstance(image_path_or_bytes, bytes):
        import io
        img = Image.open(io.BytesIO(image_path_or_bytes))
    else:
        img = Image.open(image_path_or_bytes)
    
    # Convert to RGB (handles grayscale, RGBA, etc.)
    img = img.convert('RGB')
    
    # Resize to target size using high-quality resampling
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convert to numpy array and normalize to [0, 1]
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # Add batch dimension
    return np.expand_dims(img_array, axis=0)

def predict_single(model, image):
    """Predict class for a single image."""
    processed = preprocess_image(image)
    predictions = model.predict(processed, verbose=0)[0]
    
    class_id = int(np.argmax(predictions))
    confidence = float(predictions[class_id])
    
    return {
        "class_id": class_id,
        "class_name": CLASS_NAMES[class_id],
        "confidence": confidence,
        "all_probabilities": {
            CLASS_NAMES[i]: float(predictions[i]) 
            for i in range(len(CLASS_NAMES))
        }
    }

def predict_batch(model, image_paths):
    """Predict classes for multiple images."""
    results = []
    for path in image_paths:
        result = predict_single(model, path)
        result["image_path"] = str(path)
        results.append(result)
    return results