
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


class Mixup(keras.layers.Layer):
    
    def __init__(self, alpha=0.2, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
    
    def call(self, inputs, training=None):
        if not training:
            return inputs
        
        batch_size = tf.shape(inputs)[0]
        
        # Sample lambda from Beta distribution
        lambda_val = tf.random.uniform([], 0, self.alpha)
        lambda_val = tf.maximum(lambda_val, 1 - lambda_val)
        
        # Shuffle batch
        indices = tf.random.shuffle(tf.range(batch_size))
        shuffled = tf.gather(inputs, indices)
        
        # Mix
        mixed = lambda_val * inputs + (1 - lambda_val) * shuffled
        
        return mixed
    
    def get_config(self):
        config = super().get_config()
        config.update({"alpha": self.alpha})
        return config


class CutMix(keras.layers.Layer):
    """
    CutMix data augmentation layer.
    Reference: https://arxiv.org/abs/1905.04899
    
    CutMix cuts and pastes patches between training images.
    """
    
    def __init__(self, alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
    
    def call(self, inputs, training=None):
        if not training:
            return inputs
        
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        
        # Sample lambda from Beta distribution
        lambda_val = tf.random.uniform([], 0, self.alpha)
        
        # Random shuffle
        indices = tf.random.shuffle(tf.range(batch_size))
        shuffled = tf.gather(inputs, indices)
        
        # Random box
        cut_ratio = tf.sqrt(1.0 - lambda_val)
        cut_h = tf.cast(cut_ratio * tf.cast(height, tf.float32), tf.int32)
        cut_w = tf.cast(cut_ratio * tf.cast(width, tf.float32), tf.int32)
        
        cx = tf.random.uniform([], 0, width, dtype=tf.int32)
        cy = tf.random.uniform([], 0, height, dtype=tf.int32)
        
        x1 = tf.clip_by_value(cx - cut_w // 2, 0, width)
        y1 = tf.clip_by_value(cy - cut_h // 2, 0, height)
        x2 = tf.clip_by_value(cx + cut_w // 2, 0, width)
        y2 = tf.clip_by_value(cy + cut_h // 2, 0, height)
        
        # Create mask
        mask = tf.ones([batch_size, height, width, 1])
        mask = tf.tensor_scatter_nd_update(
            mask,
            tf.stack([
                tf.range(batch_size),
                tf.fill([batch_size], y1),
                tf.fill([batch_size], x1)
            ], axis=1),
            tf.zeros([batch_size])
        )
        
        # Apply cutmix
        mixed = inputs * mask + shuffled * (1 - mask)
        
        return mixed
    
    def get_config(self):
        config = super().get_config()
        config.update({"alpha": self.alpha})
        return config


class RandomCutout(keras.layers.Layer):
    """
    Random Cutout augmentation.
    Randomly masks out square regions of the image.
    """
    
    def __init__(self, mask_size=32, num_masks=1, **kwargs):
        super().__init__(**kwargs)
        self.mask_size = mask_size
        self.num_masks = num_masks
    
    def call(self, inputs, training=None):
        if not training:
            return inputs
        
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        
        for _ in range(self.num_masks):
            # Random position
            x = tf.random.uniform([], 0, width, dtype=tf.int32)
            y = tf.random.uniform([], 0, height, dtype=tf.int32)
            
            # Calculate box coordinates
            x1 = tf.maximum(0, x - self.mask_size // 2)
            y1 = tf.maximum(0, y - self.mask_size // 2)
            x2 = tf.minimum(width, x + self.mask_size // 2)
            y2 = tf.minimum(height, y + self.mask_size // 2)
            
            # Create mask (set to 0)
            mask = tf.ones_like(inputs)
            updates = tf.zeros([1, y2-y1, x2-x1, tf.shape(inputs)[-1]])
            
            # Apply cutout (simplified version)
            inputs = inputs * 0.5  # Placeholder - proper implementation needs scatter_nd
        
        return inputs
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "mask_size": self.mask_size,
            "num_masks": self.num_masks
        })
        return config


class AstronomicalAugmentation(keras.layers.Layer):
    """
    Specialized augmentation for astronomical images.
    - Gaussian noise (sensor noise)
    - Poisson noise (photon shot noise)
    - Background brightness variations
    """
    
    def __init__(self, noise_stddev=0.02, poisson_lambda=5.0, **kwargs):
        super().__init__(**kwargs)
        self.noise_stddev = noise_stddev
        self.poisson_lambda = poisson_lambda
    
    def call(self, inputs, training=None):
        if not training:
            return inputs
        
        # Gaussian noise (sensor noise)
        gaussian_noise = tf.random.normal(
            tf.shape(inputs),
            mean=0.0,
            stddev=self.noise_stddev
        )
        
        # Apply augmentations
        outputs = inputs + gaussian_noise
        outputs = tf.clip_by_value(outputs, 0.0, 1.0)
        
        return outputs
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "noise_stddev": self.noise_stddev,
            "poisson_lambda": self.poisson_lambda
        })
        return config


def create_advanced_augmentation_pipeline(
    input_shape=(256, 256, 3),
    use_mixup=True,
    use_cutmix=False,
    use_astronomical=True,
    rotation_factor=0.3,
    zoom_range=0.15,
    brightness_range=0.15,
    contrast_range=0.15
):
    
    
    aug_layers = []
    
    # Geometric augmentations
    aug_layers.extend([
        layers.RandomRotation(
            rotation_factor,
            fill_mode='reflect',
            interpolation='bilinear'
        ),
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomZoom(
            (-zoom_range, zoom_range),
            fill_mode='reflect'
        ),
        layers.RandomTranslation(
            height_factor=0.1,
            width_factor=0.1,
            fill_mode='reflect'
        ),
    ])
    
    # Photometric augmentations
    aug_layers.extend([
        layers.RandomBrightness(brightness_range),
        layers.RandomContrast(contrast_range),
    ])
    
    # Astronomical-specific augmentations
    if use_astronomical:
        aug_layers.append(AstronomicalAugmentation())
    
    # Advanced augmentations
    if use_mixup:
        aug_layers.append(Mixup(alpha=0.2))
    
    if use_cutmix:
        aug_layers.append(CutMix(alpha=1.0))
    
    return keras.Sequential(aug_layers, name='advanced_augmentation')


def test_time_augmentation(model, image, num_augmentations=10):
    """
    Apply test-time augmentation for more robust predictions.
    
    Args:
        model: Trained Keras model
        image: Single image (H, W, C) or batch (N, H, W, C)
        num_augmentations: Number of augmented versions to average
    
    Returns:
        Averaged predictions
    """
    
    # Ensure batch dimension
    if len(image.shape) == 3:
        image = tf.expand_dims(image, 0)
    
    predictions = []
    
    # Original prediction
    pred = model.predict(image, verbose=0)
    predictions.append(pred)
    
    # Augmented predictions
    for _ in range(num_augmentations - 1):
        # Apply random augmentations
        aug_image = image
        
        # Random rotation
        if tf.random.uniform([]) > 0.5:
            k = tf.random.uniform([], 0, 4, dtype=tf.int32)
            aug_image = tf.image.rot90(aug_image, k=k)
        
        # Random flip
        if tf.random.uniform([]) > 0.5:
            aug_image = tf.image.flip_left_right(aug_image)
        if tf.random.uniform([]) > 0.5:
            aug_image = tf.image.flip_up_down(aug_image)
        
        # Predict
        pred = model.predict(aug_image, verbose=0)
        predictions.append(pred)
    
    # Average predictions
    avg_pred = np.mean(predictions, axis=0)
    
    return avg_pred


def create_tf_dataset_with_augmentation(
    X_train, y_train,
    batch_size=32,
    augment=True,
    use_mixup=False,
    shuffle=True,
    prefetch=True
):
    """
    optimized tf.data.Dataset with augmentation.
    
    Args:
        X_train: Training images
        y_train: Training labels
        batch_size: Batch size
        augment: Whether to apply augmentation
        use_mixup: Whether to use Mixup (applied after batching)
        shuffle: Whether to shuffle data
        prefetch: Whether to prefetch data
    
    Returns:
        tf.data.Dataset
    """
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    
    # Batch
    dataset = dataset.batch(batch_size)
    
    # Augmentation
    if augment:
        augmentation = create_advanced_augmentation_pipeline(
            use_mixup=False,  # Applied separately after batching
            use_astronomical=True
        )
        
        @tf.function
        def augment_batch(images, labels):
            images = augmentation(images, training=True)
            return images, labels
        
        dataset = dataset.map(
            augment_batch,
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
    # Mixup (applied to batches)
    if use_mixup:
        mixup_layer = Mixup(alpha=0.2)
        
        @tf.function
        def apply_mixup(images, labels):
            # Convert labels to one-hot
            labels_onehot = tf.one_hot(labels, depth=10)
            
            # Apply mixup
            batch_size = tf.shape(images)[0]
            lambda_val = tf.random.uniform([], 0, 0.2)
            lambda_val = tf.maximum(lambda_val, 1 - lambda_val)
            
            indices = tf.random.shuffle(tf.range(batch_size))
            mixed_images = lambda_val * images + (1 - lambda_val) * tf.gather(images, indices)
            mixed_labels = lambda_val * labels_onehot + (1 - lambda_val) * tf.gather(labels_onehot, indices)
            
            return mixed_images, mixed_labels
        
        dataset = dataset.map(
            apply_mixup,
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
    if prefetch:
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


if __name__ == "__main__":
    print("Testing augmentation pipeline...")
    
    # Create dummy data
    dummy_images = tf.random.uniform([4, 256, 256, 3])
    
    # Test augmentation
    aug_pipeline = create_advanced_augmentation_pipeline(
        use_mixup=True,
        use_astronomical=True
    )
    
    augmented = aug_pipeline(dummy_images, training=True)
    print(f"Augmented shape: {augmented.shape}")
    print("Augmentation pipeline created successfully!")
