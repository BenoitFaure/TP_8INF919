import numpy as np

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import layers, Sequential

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# -------------- Load CIFAR10 dataset --------------

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Print dimensions of training data
print("Training data shape: ", x_train.shape)
print("Training labels shape: ", y_train.shape)

# Print dimensions of test data
print("Test data shape: ", x_test.shape)
print("Test labels shape: ", y_test.shape)

# Print number of unique classes
num_classes = len(np.unique(y_train))
print("Number of classes: ", num_classes)

# -------------- Preprocess data --------------

# Preprocess block

data_augmentation = Sequential(
    [
        layers.Normalization(),
        # layers.Resizing(image_size, image_size),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(
            height_factor=0.2, width_factor=0.2
        ),
    ],
    name="data_augmentation",
)

