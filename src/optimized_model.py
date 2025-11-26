"""
Optimized Galaxy Classification Model with Advanced Techniques
- Proper input normalization
- Integrated data augmentation
- Improved architecture with consistent regularization
- Label smoothing and class weights support
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetV2S, EfficientNetB0
import numpy as np


def create_augmentation_layer(rotation_factor=0.3, zoom_factor=0.15, 
                              noise_stddev=0.02, augment_prob=0.9):
    """
    Create enhanced data augmentation layer for galaxy images.
    
    Args:
        rotation_factor: Fraction of 2*pi for random rotation (0.3 = ±108°)
        zoom_factor: Zoom range as fraction
        noise_stddev: Standard deviation for Gaussian noise
        augment_prob: Probability of applying augmentation
    
    Returns:
        Sequential model with augmentation layers
    """
    return keras.Sequential([
        layers.RandomRotation(
            rotation_factor, 
            fill_mode='reflect',
            interpolation='bilinear'
        ),
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomZoom(
            (-zoom_factor, zoom_factor),
            fill_mode='reflect'
        ),
        layers.RandomBrightness(0.15),
        layers.RandomContrast(0.15),
        layers.GaussianNoise(noise_stddev),
    ], name='augmentation')


class MixupLayer(layers.Layer):
    """Implements Mixup data augmentation."""
    
    def __init__(self, alpha=0.2, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
    
    def call(self, inputs, training=None):
        if training is None or not training:
            return inputs
        
        images, labels = inputs
        batch_size = tf.shape(images)[0]
        
        # Sample lambda from Beta distribution
        lambda_val = tf.random.uniform([], 0, self.alpha)
        lambda_val = tf.maximum(lambda_val, 1 - lambda_val)
        
        # Shuffle indices
        indices = tf.random.shuffle(tf.range(batch_size))
        
        # Mix images and labels
        mixed_images = lambda_val * images + (1 - lambda_val) * tf.gather(images, indices)
        mixed_labels = lambda_val * labels + (1 - lambda_val) * tf.gather(labels, indices)
        
        return mixed_images, mixed_labels


def create_optimized_galaxy_classifier(
    num_classes=10,
    input_shape=(256, 256, 3),
    backbone='efficientnetv2s',
    use_augmentation=True,
    dropout_rate=0.3,
    l2_reg=0.01,
    dense_units=256,
    label_smoothing=0.1
):
    """
    Create optimized EfficientNet-based galaxy classifier with proper preprocessing.
    
    Args:
        num_classes: Number of galaxy classes
        input_shape: Input image shape
        backbone: 'efficientnetv2s' or 'efficientnetb0'
        use_augmentation: Whether to include augmentation in model
        dropout_rate: Dropout rate for regularization
        l2_reg: L2 regularization factor
        dense_units: Number of units in dense layer
        label_smoothing: Label smoothing factor (0 = no smoothing)
    
    Returns:
        Compiled Keras model and base model reference
    """
    
    # Select backbone
    if backbone == 'efficientnetv2s':
        base_model = EfficientNetV2S(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape,
            pooling='avg'
        )
    elif backbone == 'efficientnetb0':
        base_model = EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape,
            pooling='avg'
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    
    # Freeze base initially
    base_model.trainable = False
    
    # Build model with preprocessing
    inputs = keras.Input(shape=input_shape)
    
    # Normalization: [0, 255] -> [0, 1]
    x = layers.Rescaling(1./255)(inputs)
    
    # Data augmentation (applied during training only)
    if use_augmentation:
        x = create_augmentation_layer()(x)
    
    # EfficientNet preprocessing (model-specific normalization)
    if backbone == 'efficientnetv2s':
        x = tf.keras.applications.efficientnet_v2.preprocess_input(x * 255)
    else:
        x = tf.keras.applications.efficientnet.preprocess_input(x * 255)
    
    # Base model
    x = base_model(x, training=False)
    
    # Classification head with consistent regularization
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(
        dense_units,
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        kernel_initializer='he_normal'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate * 1.2)(x)
    x = layers.Dense(
        dense_units // 2,
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        kernel_initializer='he_normal'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate * 0.8)(x)
    
    # Output layer
    outputs = layers.Dense(
        num_classes,
        activation='softmax',
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        name='predictions'
    )(x)
    
    model = Model(inputs, outputs, name=f'GalaxyAI_{backbone}')
    
    return model, base_model


def compile_optimized_model(
    model,
    learning_rate=1e-4,
    label_smoothing=0.1,
    class_weights=None
):
    """
    Compile model with optimized configuration.
    
    Args:
        model: Keras model to compile
        learning_rate: Initial learning rate
        label_smoothing: Label smoothing factor
        class_weights: Dictionary of class weights for imbalanced data
    
    Returns:
        Compiled model
    """
    
    # Custom loss with label smoothing for sparse labels
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
        
        loss = smooth_sparse_categorical_crossentropy
    else:
        loss = keras.losses.SparseCategoricalCrossentropy()
    
    # AdamW optimizer with weight decay
    optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=1e-5,
        clipnorm=1.0  # Gradient clipping
    )
    
    # Metrics
    metrics = [
        'accuracy',
        keras.metrics.SparseTopKCategoricalAccuracy(k=2, name='top2_acc'),
        keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top3_acc')
    ]
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    return model


def get_training_callbacks(
    model_save_path='models/best_galaxy_model.h5',
    patience=8,
    min_delta=0.001,
    reduce_lr_patience=4
):
    """
    Get optimized training callbacks.
    
    Args:
        model_save_path: Path to save best model
        patience: Early stopping patience
        min_delta: Minimum improvement for early stopping
        reduce_lr_patience: Patience for learning rate reduction
    
    Returns:
        List of callbacks
    """
    
    callbacks = [
        # Save best model
        keras.callbacks.ModelCheckpoint(
            model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        
        # Early stopping with more patience
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            min_delta=min_delta,
            verbose=1
        ),
        
        # Learning rate reduction with cosine decay
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=1e-7,
            verbose=1
        ),
        
        # TensorBoard logging
        keras.callbacks.TensorBoard(
            log_dir='logs',
            histogram_freq=1,
            write_graph=True
        ),
        
        # Track learning rate
        keras.callbacks.LearningRateScheduler(
            lambda epoch, lr: lr if epoch < 5 else lr * tf.math.exp(-0.1),
            verbose=0
        )
    ]
    
    return callbacks


def gradual_unfreeze(base_model, model, unfreeze_percentage=0.3):
    """
    Gradually unfreeze layers in base model.
    
    Args:
        base_model: Base model (EfficientNet)
        model: Full model
        unfreeze_percentage: Percentage of layers to unfreeze (0.0 to 1.0)
    
    Returns:
        Number of layers unfrozen
    """
    total_layers = len(base_model.layers)
    unfreeze_from = int(total_layers * (1 - unfreeze_percentage))
    
    # Unfreeze from specific layer onwards
    for layer in base_model.layers[:unfreeze_from]:
        layer.trainable = False
    for layer in base_model.layers[unfreeze_from:]:
        layer.trainable = True
    
    # Recompile model after unfreezing
    print(f"Unfrozen {total_layers - unfreeze_from} out of {total_layers} layers")
    
    return total_layers - unfreeze_from


def calculate_class_weights(labels):
    """
    Calculate class weights for imbalanced dataset.
    
    Args:
        labels: Array of class labels
    
    Returns:
        Dictionary of class weights
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(labels)
    weights = compute_class_weight(
        'balanced',
        classes=classes,
        y=labels
    )
    
    class_weights = dict(zip(classes, weights))
    
    print("Class weights:")
    for cls, weight in class_weights.items():
        print(f"  Class {cls}: {weight:.3f}")
    
    return class_weights


def create_cosine_decay_schedule(
    initial_learning_rate=1e-4,
    decay_steps=1000,
    alpha=0.0,
    warmup_steps=100
):
    """
    Create cosine decay learning rate schedule with warmup.
    
    Args:
        initial_learning_rate: Starting learning rate
        decay_steps: Number of steps for full cosine cycle
        alpha: Minimum learning rate as fraction of initial
        warmup_steps: Number of warmup steps
    
    Returns:
        Learning rate schedule
    """
    
    def schedule(step):
        if step < warmup_steps:
            return initial_learning_rate * (step / warmup_steps)
        
        step = step - warmup_steps
        decay_steps_adjusted = decay_steps - warmup_steps
        
        cosine_decay = 0.5 * (1 + tf.math.cos(
            tf.constant(np.pi) * step / decay_steps_adjusted
        ))
        decayed = (1 - alpha) * cosine_decay + alpha
        
        return initial_learning_rate * decayed
    
    return schedule


# Focal Loss for hard examples
class FocalLoss(keras.losses.Loss):
    """
    Focal Loss for addressing class imbalance.
    Focuses training on hard examples.
    """
    
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
    
    def call(self, y_true, y_pred):
        # Convert to one-hot if needed
        y_true = tf.cast(y_true, tf.int32)
        y_true = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])
        
        # Clip predictions to prevent log(0)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate focal loss
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = self.alpha * y_true * tf.math.pow(1 - y_pred, self.gamma)
        focal_loss = weight * cross_entropy
        
        return tf.reduce_sum(focal_loss, axis=-1)


if __name__ == "__main__":
    # Example usage
    print("Creating optimized galaxy classifier...")
    
    model, base_model = create_optimized_galaxy_classifier(
        num_classes=10,
        input_shape=(256, 256, 3),
        backbone='efficientnetv2s',
        use_augmentation=True,
        dropout_rate=0.3,
        l2_reg=0.01,
        dense_units=256
    )
    
    model = compile_optimized_model(
        model,
        learning_rate=1e-4,
        label_smoothing=0.1
    )
    
    print("\nModel created successfully!")
    print(f"Total parameters: {model.count_params():,}")
    print(f"Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
