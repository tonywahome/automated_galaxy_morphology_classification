"""
Advanced Training Utilities for Galaxy Classification
- Custom learning rate schedules
- Training loops with monitoring
- Evaluation metrics and visualization
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_fscore_support, roc_auc_score
)
import json
from datetime import datetime


def create_label_smoothing_loss(label_smoothing=0.1):
    """
    Create a loss function with label smoothing for sparse labels.
    
    Args:
        label_smoothing: Smoothing factor (0.0 = no smoothing, 0.1 = 10% smoothing)
    
    Returns:
        Loss function compatible with sparse integer labels
    """
    if label_smoothing > 0:
        def smooth_sparse_categorical_crossentropy(y_true, y_pred):
            """Apply label smoothing to sparse categorical crossentropy"""
            num_classes = tf.shape(y_pred)[-1]
            y_true = tf.cast(y_true, tf.int32)
            
            # Convert to one-hot
            y_true_one_hot = tf.one_hot(y_true, depth=num_classes)
            
            # Apply label smoothing: y_smooth = y * (1 - α) + α / K
            y_smooth = y_true_one_hot * (1.0 - label_smoothing) + label_smoothing / tf.cast(num_classes, tf.float32)
            
            # Compute crossentropy
            return tf.keras.losses.categorical_crossentropy(y_smooth, y_pred)
        
        return smooth_sparse_categorical_crossentropy
    else:
        return keras.losses.SparseCategoricalCrossentropy()


class CosineDecayWithWarmup(keras.optimizers.schedules.LearningRateSchedule):
    """
    Cosine decay learning rate schedule with linear warmup.
    """
    
    def __init__(
        self,
        initial_learning_rate,
        decay_steps,
        alpha=0.0,
        warmup_steps=0,
        warmup_target=None
    ):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.alpha = alpha
        self.warmup_steps = warmup_steps
        self.warmup_target = warmup_target or initial_learning_rate
    
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        
        # Warmup phase
        if self.warmup_steps > 0:
            warmup_lr = (self.warmup_target / self.warmup_steps) * step
            
            # Cosine decay phase
            decay_step = step - self.warmup_steps
            decay_steps = self.decay_steps - self.warmup_steps
            
            cosine_decay = 0.5 * (1 + tf.cos(
                tf.constant(np.pi) * decay_step / decay_steps
            ))
            decayed = (1 - self.alpha) * cosine_decay + self.alpha
            cosine_lr = self.initial_learning_rate * decayed
            
            return tf.where(
                step < self.warmup_steps,
                warmup_lr,
                cosine_lr
            )
        else:
            cosine_decay = 0.5 * (1 + tf.cos(
                tf.constant(np.pi) * step / self.decay_steps
            ))
            decayed = (1 - self.alpha) * cosine_decay + self.alpha
            return self.initial_learning_rate * decayed
    
    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "alpha": self.alpha,
            "warmup_steps": self.warmup_steps,
            "warmup_target": self.warmup_target
        }


class MetricsLogger(keras.callbacks.Callback):
    """
    Custom callback to log detailed metrics during training.
    """
    
    def __init__(self, log_dir='logs', class_names=None):
        super().__init__()
        self.log_dir = log_dir
        self.class_names = class_names
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        # Log metrics
        self.history['train_loss'].append(logs.get('loss', 0))
        self.history['train_acc'].append(logs.get('accuracy', 0))
        self.history['val_loss'].append(logs.get('val_loss', 0))
        self.history['val_acc'].append(logs.get('val_accuracy', 0))
        
        # Log learning rate
        lr = float(keras.backend.get_value(self.model.optimizer.learning_rate))
        self.history['learning_rate'].append(lr)
        
        # Print summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Loss: {logs.get('loss', 0):.4f} | Val Loss: {logs.get('val_loss', 0):.4f}")
        print(f"  Acc: {logs.get('accuracy', 0):.4f} | Val Acc: {logs.get('val_accuracy', 0):.4f}")
        print(f"  Learning Rate: {lr:.6f}")
    
    def save_history(self, filename='training_history.json'):
        """Save training history to JSON file."""
        with open(f"{self.log_dir}/{filename}", 'w') as f:
            json.dump(self.history, f, indent=2)


def train_with_gradual_unfreezing(
    model,
    base_model,
    X_train, y_train,
    X_val, y_val,
    initial_epochs=15,
    fine_tune_epochs=10,
    batch_size=32,
    initial_lr=1e-4,
    fine_tune_lr=1e-5,
    class_weights=None,
    callbacks=None
):
    """
    Train model with two-phase gradual unfreezing strategy.
    
    Phase 1: Train with frozen base model
    Phase 2: Fine-tune with partially unfrozen base model
    
    Args:
        model: Keras model
        base_model: Base model (EfficientNet)
        X_train, y_train: Training data
        X_val, y_val: Validation data
        initial_epochs: Epochs for phase 1
        fine_tune_epochs: Epochs for phase 2
        batch_size: Batch size
        initial_lr: Learning rate for phase 1
        fine_tune_lr: Learning rate for phase 2
        class_weights: Class weights dictionary
        callbacks: List of callbacks
    
    Returns:
        Training history
    """
    
    print("="*70)
    print("PHASE 1: Training with frozen base model")
    print("="*70)
    
    # Ensure base is frozen
    base_model.trainable = False
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=initial_lr, weight_decay=1e-5),
        loss=create_label_smoothing_loss(label_smoothing=0.1),
        metrics=['accuracy', keras.metrics.SparseTopKCategoricalAccuracy(k=2, name='top2_acc')]
    )
    
    # Train
    history_phase1 = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=initial_epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n" + "="*70)
    print("PHASE 2: Fine-tuning with gradual unfreezing")
    print("="*70)
    
    # Gradual unfreezing: Unfreeze last 30% of layers
    total_layers = len(base_model.layers)
    unfreeze_from = int(total_layers * 0.7)
    
    for layer in base_model.layers[:unfreeze_from]:
        layer.trainable = False
    for layer in base_model.layers[unfreeze_from:]:
        layer.trainable = True
    
    print(f"Unfrozen {total_layers - unfreeze_from}/{total_layers} base model layers")
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=fine_tune_lr, weight_decay=1e-5),
        loss=create_label_smoothing_loss(label_smoothing=0.1),
        metrics=['accuracy', keras.metrics.SparseTopKCategoricalAccuracy(k=2, name='top2_acc')]
    )
    
    # Fine-tune
    history_phase2 = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=fine_tune_epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    # Combine histories
    combined_history = {
        'loss': history_phase1.history['loss'] + history_phase2.history['loss'],
        'accuracy': history_phase1.history['accuracy'] + history_phase2.history['accuracy'],
        'val_loss': history_phase1.history['val_loss'] + history_phase2.history['val_loss'],
        'val_accuracy': history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy']
    }
    
    return combined_history


def evaluate_model_comprehensive(
    model,
    X_test, y_test,
    class_names,
    save_dir='evaluation_results'
):
    """
    Comprehensive model evaluation with metrics and visualizations.
    
    Args:
        model: Trained model
        X_test, y_test: Test data
        class_names: List of class names
        save_dir: Directory to save results
    
    Returns:
        Dictionary of metrics
    """
    
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    print("="*70)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*70)
    
    # Predictions
    y_pred_proba = model.predict(X_test, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Overall metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average='macro')
    recall_macro = recall_score(y_test, y_pred, average='macro')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    print("\n=== OVERALL METRICS ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (macro): {precision_macro:.4f}")
    print(f"Recall (macro): {recall_macro:.4f}")
    print(f"F1-Score (macro): {f1_macro:.4f}")
    print(f"F1-Score (weighted): {f1_weighted:.4f}")
    
    # Per-class metrics
    print("\n=== PER-CLASS METRICS ===")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Save classification report
    report_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    with open(f"{save_dir}/classification_report.json", 'w') as f:
        json.dump(report_dict, f, indent=2)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Normalized Confusion Matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm_normalized, annot=True, fmt='.2f', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        vmin=0, vmax=1, cbar_kws={'label': 'Proportion'}
    )
    plt.title('Normalized Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/confusion_matrix_normalized.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # ROC-AUC
    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
        print(f"\nROC-AUC (macro): {roc_auc:.4f}")
    except:
        roc_auc = None
        print("\nROC-AUC calculation failed")
    
    # Per-class accuracy
    print("\n=== PER-CLASS ACCURACY ===")
    for i, class_name in enumerate(class_names):
        class_mask = y_test == i
        if class_mask.sum() > 0:
            class_acc = (y_pred[class_mask] == i).sum() / class_mask.sum()
            print(f"{class_name:30s}: {class_acc:.4f} ({class_mask.sum()} samples)")
    
    # Top-k accuracy
    top2_correct = sum([1 for true, pred_proba in zip(y_test, y_pred_proba) 
                        if true in np.argsort(pred_proba)[-2:]])
    top3_correct = sum([1 for true, pred_proba in zip(y_test, y_pred_proba) 
                        if true in np.argsort(pred_proba)[-3:]])
    
    top2_acc = top2_correct / len(y_test)
    top3_acc = top3_correct / len(y_test)
    
    print(f"\nTop-2 Accuracy: {top2_acc:.4f}")
    print(f"Top-3 Accuracy: {top3_acc:.4f}")
    
    # Compile results
    results = {
        'accuracy': float(accuracy),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'roc_auc': float(roc_auc) if roc_auc else None,
        'top2_accuracy': float(top2_acc),
        'top3_accuracy': float(top3_acc),
        'timestamp': datetime.now().isoformat()
    }
    
    # Save results
    with open(f"{save_dir}/evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {save_dir}/")
    
    return results


def plot_training_history(history, save_path='training_curves.png'):
    """
    Plot training history curves.
    
    Args:
        history: Training history dictionary or History object
        save_path: Path to save the plot
    """
    
    if hasattr(history, 'history'):
        history = history.history
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(history['accuracy'], label='Train', linewidth=2)
    axes[0, 0].plot(history['val_accuracy'], label='Validation', linewidth=2)
    axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    axes[0, 1].plot(history['loss'], label='Train', linewidth=2)
    axes[0, 1].plot(history['val_loss'], label='Validation', linewidth=2)
    axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning Rate (if available)
    if 'lr' in history or 'learning_rate' in history:
        lr_key = 'lr' if 'lr' in history else 'learning_rate'
        axes[1, 0].plot(history[lr_key], linewidth=2, color='green')
        axes[1, 0].set_title('Learning Rate', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Accuracy difference
    acc_diff = np.array(history['accuracy']) - np.array(history['val_accuracy'])
    axes[1, 1].plot(acc_diff, linewidth=2, color='red')
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    axes[1, 1].set_title('Overfitting Monitor (Train - Val Accuracy)', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy Difference')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to {save_path}")


if __name__ == "__main__":
    print("Training utilities module loaded successfully!")
    
    # Example: Create cosine decay schedule
    schedule = CosineDecayWithWarmup(
        initial_learning_rate=1e-4,
        decay_steps=1000,
        warmup_steps=100
    )
    
    print(f"Learning rate at step 0: {schedule(0):.6f}")
    print(f"Learning rate at step 50: {schedule(50):.6f}")
    print(f"Learning rate at step 100: {schedule(100):.6f}")
    print(f"Learning rate at step 500: {schedule(500):.6f}")
