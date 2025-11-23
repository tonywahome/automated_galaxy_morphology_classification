import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetV2S

def create_galaxy_classifier(num_classes=10, input_shape=(256, 256, 3)):
    """Create EfficientNetV2-S based classifier."""
    
    # Base model with pre-trained weights
    base_model = EfficientNetV2S(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling='avg'
    )
    
    # Freeze base initially
    base_model.trainable = False
    
    # Build model
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(512, activation='relu', 
                     kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs, name='GalaxAI_Classifier')
    return model, base_model

def compile_model(model, learning_rate=1e-4):
    """Compile model with optimizer and loss."""
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', 
                 tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name='top2_acc')]
    )
    return model

def train_model(model, train_data, val_data, epochs=50, callbacks=None):
    """Train the model."""
    if callbacks is None:
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
            tf.keras.callbacks.ModelCheckpoint('models/best_model.h5', save_best_only=True)
        ]
    
    history = model.fit(
        train_data[0], train_data[1],
        validation_data=val_data,
        epochs=epochs,
        batch_size=32,
        callbacks=callbacks,
        class_weight=compute_class_weights(train_data[1])
    )
    return history

def compute_class_weights(labels):
    """Compute balanced class weights."""
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np
    
    weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    return dict(enumerate(weights))

def save_model(model, path='models/galaxai_model.h5'):
    """Save trained model."""
    model.save(path)
    
def load_model(path='models/galaxai_model.h5'):
    """Load trained model."""
    return tf.keras.models.load_model(path)