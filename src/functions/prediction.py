import numpy as np
from PIL import Image
import tensorflow as tf

CLASS_NAMES = [
    "Disturbed", "Merging", "Round Smooth", "In-between Smooth",
    "Cigar-shaped", "Barred Spiral", "Unbarred Tight Spiral",
    "Unbarred Loose Spiral", "Edge-on No Bulge", "Edge-on With Bulge"
]

def preprocess_image(image_path_or_bytes, target_size=(256, 256)):
    """Preprocess a single image for prediction."""
    if isinstance(image_path_or_bytes, bytes):
        import io
        img = Image.open(io.BytesIO(image_path_or_bytes))
    else:
        img = Image.open(image_path_or_bytes)
    
    img = img.convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
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