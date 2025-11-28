# src/functions/retraining.py
import os
import sys
import json
import numpy as np
import tensorflow as tf
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from preprocessing module
import preprocessing as preproc
# Import model functions from preprocessing subdirectory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'preprocessing'))
from model import (
    create_galaxy_classifier,
    compile_model,
    train_model,
    compute_class_weights
)

# Use preprocessing functions
load_galaxy_data = preproc.load_galaxy_data
split_data = preproc.split_data

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
        
        # Fine-tune or create new model
        if fine_tune and os.path.exists(self.model_path):
            # Load existing model for fine-tuning
            model = tf.keras.models.load_model(self.model_path)
            
            # Recompile with lower learning rate for fine-tuning
            model.compile(
                optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-5),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
        else:
            # Create new model
            model, base_model = create_galaxy_classifier(
                num_classes=10,
                input_shape=(256, 256, 3)
            )
            
            model = compile_model(model, learning_rate=1e-4)
        
        # Setup callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=10,
                restore_best_weights=True,
                monitor='val_loss',
                min_delta=0.0005
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=5,
                monitor='val_loss'
            ),
            tf.keras.callbacks.ModelCheckpoint(
                self.model_path,
                save_best_only=True,
                monitor='val_loss'
            )
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            class_weight=compute_class_weights(y_train)
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