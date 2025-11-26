# src/functions/retraining.py
import os
import sys
import json
import numpy as np
import tensorflow as tf
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing import load_galaxy_data, split_data
from optimized_model import (
    create_optimized_galaxy_classifier,
    compile_optimized_model,
    get_training_callbacks,
    calculate_class_weights
)
from training_utils import train_with_gradual_unfreezing

class RetrainingPipeline:
    def __init__(self, model_path='models/galaxai_model.h5', 
                 upload_dir='data/uploads/'):
        self.model_path = model_path
        self.upload_dir = upload_dir
        self.metadata_path = 'models/model_metadata.json'
        
    def load_uploaded_data(self):
        """Load user-uploaded images for retraining."""
        images = []
        labels = []
        
        for class_folder in os.listdir(self.upload_dir):
            class_path = os.path.join(self.upload_dir, class_folder)
            if os.path.isdir(class_path):
                class_id = int(class_folder.split('_')[1])
                for img_file in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_file)
                    img = tf.keras.utils.load_img(img_path, target_size=(256, 256))
                    img_array = tf.keras.utils.img_to_array(img) / 255.0
                    images.append(img_array)
                    labels.append(class_id)
        
        return np.array(images), np.array(labels)
    
    def merge_datasets(self, original_data, new_data):
        """Merge original and new data."""
        X_orig, y_orig = original_data
        X_new, y_new = new_data
        
        X_merged = np.concatenate([X_orig, X_new], axis=0)
        y_merged = np.concatenate([y_orig, y_new], axis=0)
        
        return X_merged, y_merged
    
    def retrain(self, epochs=30, fine_tune=True, initial_epochs=15, fine_tune_epochs=15):
        """Execute retraining pipeline."""
        # Load original data
        orig_images, orig_labels = load_galaxy_data()
        
        # Load new uploaded data
        new_images, new_labels = self.load_uploaded_data()
        
        if len(new_images) == 0:
            return {"status": "error", "message": "No new data found"}
        
        # Merge datasets
        X_all, y_all = self.merge_datasets(
            (orig_images, orig_labels), 
            (new_images, new_labels)
        )
        
        # Split data
        train_data, val_data, test_data = split_data(X_all, y_all)
        X_train, y_train = train_data
        X_val, y_val = val_data
        X_test, y_test = test_data
        
        # Calculate class weights
        class_weights = calculate_class_weights(y_train)
        
        # Fine-tune or create new model
        if fine_tune and os.path.exists(self.model_path):
            # Load existing model for fine-tuning
            model = tf.keras.models.load_model(self.model_path)
            
            # Get base model for gradual unfreezing
            base_model = None
            for layer in model.layers:
                if 'efficientnet' in layer.name.lower():
                    base_model = layer
                    break
            
            if base_model is None:
                print("Warning: Could not find base model, training all layers")
                base_model = model
            
            # Recompile with lower learning rate for fine-tuning
            model = compile_optimized_model(
                model,
                learning_rate=1e-5,
                label_smoothing=0.1,
                class_weights=class_weights
            )
        else:
            # Create new model
            model, base_model = create_optimized_galaxy_classifier(
                num_classes=10,
                input_shape=(256, 256, 3),
                backbone='efficientnetv2s',
                use_augmentation=True,
                dropout_rate=0.35,
                l2_reg=0.01,
                dense_units=256
            )
            
            model = compile_optimized_model(
                model,
                learning_rate=1e-4,
                label_smoothing=0.1,
                class_weights=class_weights
            )
        
        # Setup callbacks
        callbacks = get_training_callbacks(
            model_save_path=self.model_path,
            patience=10,
            min_delta=0.0005,
            reduce_lr_patience=5
        )
        
        # Train with gradual unfreezing
        history = train_with_gradual_unfreezing(
            model=model,
            base_model=base_model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            initial_epochs=initial_epochs,
            fine_tune_epochs=fine_tune_epochs,
            batch_size=64,
            initial_lr=1e-5 if fine_tune else 1e-4,
            fine_tune_lr=1e-6 if fine_tune else 1e-5,
            class_weights=class_weights,
            callbacks=callbacks
        )
        
        # Evaluate
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        
        # Save new model with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_model_path = f'models/galaxai_model_{timestamp}.h5'
        os.makedirs('models', exist_ok=True)
        model.save(new_model_path)
        model.save(self.model_path)  # Update main model
        
        # Update metadata
        metadata = {
            "version": timestamp,
            "accuracy": float(test_acc),
            "training_samples": len(X_all),
            "new_samples_added": len(new_images),
            "retrained_at": datetime.now().isoformat(),
            "initial_epochs": initial_epochs,
            "fine_tune_epochs": fine_tune_epochs,
            "total_epochs": initial_epochs + fine_tune_epochs
        }
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Clear uploads
        self._clear_uploads()
        
        return {
            "status": "success",
            "accuracy": float(test_acc),
            "model_path": new_model_path,
            "metadata": metadata,
            "history": history.history if hasattr(history, 'history') else history
        }
    
    def _clear_uploads(self):
        """Clear uploaded files after retraining."""
        import shutil
        for item in os.listdir(self.upload_dir):
            item_path = os.path.join(self.upload_dir, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)