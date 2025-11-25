import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_lenet5(input_shape=(28, 28, 1), num_classes=10):
    """
    Build LeNet-5 architecture for image classification.
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of output classes
    
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        # C1: Convolutional Layer
        layers.Conv2D(6, kernel_size=5, activation='tanh', padding='same'),
        layers.AveragePooling2D(pool_size=2, strides=2),
        
        # C2: Convolutional Layer
        layers.Conv2D(16, kernel_size=5, activation='tanh'),
        layers.AveragePooling2D(pool_size=2, strides=2),
        
        # Flatten
        layers.Flatten(),
        
        # F3: Fully Connected Layer
        layers.Dense(120, activation='tanh'),
        
        # F4: Fully Connected Layer
        layers.Dense(84, activation='tanh'),
        
        # Output Layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model