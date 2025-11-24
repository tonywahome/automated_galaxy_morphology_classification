import numpy as np
from astroNN.datasets import galaxy10
from sklearn.model_selection import train_test_split
import tensorflow as tf

def load_galaxy_data():
    """Load Galaxy10 DECaLS dataset."""
    images, labels = galaxy10.load_data()
    images = images.astype('float32') / 255.0  # Normalize
    return images, labels

def split_data(images, labels, test_size=0.15, val_size=0.15):
    """Split into train/val/test sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=test_size, stratify=labels, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size/(1-test_size), 
        stratify=y_train, random_state=42
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def create_augmentation_layer():
    """Data augmentation for galaxy images."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.5),      # Galaxies have no orientation
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomBrightness(0.2),
        tf.keras.layers.RandomContrast(0.2),
    ])

def save_images_to_folders(images, labels, base_path):
    """Save images to train/test folder structure."""
    import os
    from PIL import Image
    
    class_names = [f"class_{i}" for i in range(10)]
    for class_name in class_names:
        os.makedirs(f"{base_path}/{class_name}", exist_ok=True)
    
    for idx, (img, label) in enumerate(zip(images, labels)):
        img_pil = Image.fromarray((img * 255).astype(np.uint8))
        img_pil.save(f"{base_path}/class_{label}/galaxy_{idx}.png")