# Classification on CIFAR-10 dataset using a not pretrained ResNet-18

# Dependencies
import datetime
import numpy as np

import tensorflow as tf

# ------------------- tf imports -------------------

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Dropout, \
    AveragePooling2D, BatchNormalization, Activation, Add, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers.experimental.preprocessing import RandomRotation, \
      RandomZoom, RandomFlip, RandomTranslation, Resizing
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint

# ------------------- Settings ------------------- #

# Data settings
resize = (32, 32)

# Model settings
initializer = tf.initializers.he_normal(seed=42)
classification_initializer = tf.initializers.glorot_normal(seed=42)

activation_function = 'relu' # 'selu'
use_bias = False # Batch Norm takes care of it
epsilon_bn = 1.001e-5
kernel_reg = None # tf.keras.regularizers.l2(0.0001)

# Training settings
epochs = 1000 # 90
batch_size = 64 # 256

# Define optimizer
learning_rate = 0.005
# optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
optimizer = tf.keras.optimizers.Adam(
    learning_rate=learning_rate, clipvalue=0.12)
# optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01)

# Define loss
loss = CategoricalCrossentropy(label_smoothing=0.2)

# ------------------- Load data ------------------- #

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Print dimensions of training data
print("Training data shape: ", x_train.shape)
print("Training labels shape: ", y_train.shape)

# Print dimensions of test data
print("Test data shape: ", x_test.shape)
print("Test labels shape: ", y_test.shape)

# Print number of unique classes
num_classes = len(np.unique(y_train))
print("Number of classes: ", num_classes)

# ------------------- Preprocess data ------------------- #

# One hot encode labels
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Load as tf dataset
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

# Normalize images
def normalize_img(image, label):

    # Resize images
    image = tf.image.resize(image, resize)
    image = tf.cast(image, tf.float32) / 255.

    return image, label

# Normalise images
ds_train = train_dataset.map(normalize_img)
ds_test = test_dataset.map(normalize_img)

# ------------------- Define model ------------------- #

# Identity block
def identity_block(filter, kernel_size=3):
    def _identity_block(x):
        input_x = x

        # Layer 1
        x = Conv2D(filter, kernel_size=kernel_size, 
                   padding = 'same', kernel_initializer=initializer,
                   use_bias=use_bias, kernel_regularizer=kernel_reg)(x)
        x = BatchNormalization(axis=3, epsilon=epsilon_bn)(x)
        x = Activation(activation_function)(x)

        # Layer 2
        x = Conv2D(filter, kernel_size=kernel_size, 
                   padding = 'same', kernel_initializer=initializer,
                   use_bias=use_bias, kernel_regularizer=kernel_reg)(x)
        x = BatchNormalization(axis=3, epsilon=epsilon_bn)(x)

        # Add Residue
        x = Add()([x, input_x])
        
        # Activation
        x = Activation(activation_function)(x)

        return x
    
    return _identity_block

# Convolutional Res block
def convolutional_res_block(filter, kernel_size=3):
    def _convolutional_res_block(x):
        input_x = x

        # Layer 1
        x = Conv2D(filter, kernel_size=kernel_size, 
                   padding = 'same', kernel_initializer=initializer,
                   use_bias=use_bias, kernel_regularizer=kernel_reg)(x)
        x = BatchNormalization(axis=3, epsilon=epsilon_bn)(x)
        x = Activation(activation_function)(x)

        # Layer 2
        x = Conv2D(filter, kernel_size=kernel_size, 
                   padding = 'same', kernel_initializer=initializer,
                   use_bias=use_bias, kernel_regularizer=kernel_reg)(x)
        x = BatchNormalization(axis=3, epsilon=epsilon_bn)(x)

        # Layer 3
        input_x = Conv2D(filter, kernel_size=kernel_size, 
                   padding = 'same', kernel_initializer=initializer,
                   use_bias=use_bias, kernel_regularizer=kernel_reg)(input_x)
        input_x = BatchNormalization(axis=3, epsilon=epsilon_bn)(input_x)

        # Add Residue
        x = Add()([x, input_x])
        
        # Activation
        x = Activation(activation_function)(x)

        return x
    
    return _convolutional_res_block

# Convolutional block
def convolutional_block(filter, kernel_size=3, strides=2):
    def _convolutional_block(x):
        
        x = Conv2D(filter, kernel_size=kernel_size, 
                   padding = 'same', kernel_initializer=initializer,
               
                   use_bias=use_bias, strides=strides, kernel_regularizer=kernel_reg)(x)
        x = BatchNormalization(axis=3, epsilon=epsilon_bn)(x)
        x = Activation(activation_function)(x)

        return x
    
    return _convolutional_block

# Define model

input_shape = (resize[0], resize[1], 3)
input_layer = Input(shape=input_shape)

data_aug = [
    RandomRotation(0.1, seed=42),
    RandomTranslation(0.1, 0.1, seed=42),
    RandomZoom(0.1, seed=42),
    RandomFlip(seed=42),
]

model_name = 'ResNet-9'
conv_net = [

    convolutional_block(64, strides=1),
    convolutional_block(128, strides=1),
    MaxPooling2D(pool_size=2, strides=2),

    identity_block(128, kernel_size=3),

    convolutional_block(256, strides=1),
    MaxPooling2D(pool_size=2, strides=2),
    
    convolutional_block(512, strides=1),
    MaxPooling2D(pool_size=2, strides=2),

    identity_block(512, kernel_size=3),

    MaxPooling2D(pool_size=4, strides=1)
]

# classification_net = [
#     Flatten(),
#     Dense(512, kernel_initializer=initializer, activation=activation_function),
#     Dropout(0.5),
#     Dense(num_classes, kernel_initializer=classification_initializer, activation='softmax')
# ]

classification_net = [
    Flatten(),
    Dropout(0.3), # 0.5
    Dense(num_classes, kernel_initializer=classification_initializer, activation='softmax')
]

# Compile function
def create_classifier():

    def compile_layers(input, layers):
        for layer in layers:
            input = layer(input)
        return input

    # Build Data augmentation
    data_augmentation_layers = compile_layers(input_layer, data_aug)

    # Build Feature Extractor
    conv_net_layers = compile_layers(data_augmentation_layers, conv_net)

    # Build Classifier
    classification_layers = compile_layers(conv_net_layers, classification_net)

    output_layer = classification_layers

    # Build model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# ------------------- Prepare Data ------------------- #

# Configure Dataset for Performance
AUTOTUNE = tf.data.experimental.AUTOTUNE

# ds_train = ds_train.cache()
ds_train = ds_train.shuffle(1000).batch(batch_size).prefetch(buffer_size=AUTOTUNE)

ds_test = ds_test.batch(batch_size).prefetch(buffer_size=AUTOTUNE)

del x_train, y_train, x_test, y_test

# ------------------- Prepare model ------------------- #\

# Create model
model = create_classifier()

# # Summary
# model.summary()
# exit()

# Compile model
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# Summary
model.summary()

# ------------------- Train model ------------------- #

# Define train function
def train_test_model(model):

    # Callbacks
    name_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '_' + model_name + '_' + str(0)

    #   Save model callback
    model_loc = 'models/' + name_str + '.h5'
    checkpoint = ModelCheckpoint(model_loc, monitor='val_accuracy', 
                                save_best_only=True, mode='max', verbose=1)
    
    print('Model saved at: ' + model_loc)

    #   Tensorboard callback
    log_dir = "logs/fit/" + name_str
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs/fit/" + name_str, histogram_freq=1)

    print('tensorboard --logdir ' + log_dir)

    # Train model
    run_hist = model.fit(ds_train, validation_data=ds_test,
                         epochs=epochs, batch_size=batch_size,
                         callbacks=[checkpoint, tensorboard_callback])
    
    return run_hist

# Run train and validation
run_hist = train_test_model(model)

# ------------------- Evaluate model ------------------- #

# Evaluate model
test_loss, test_acc = model.evaluate(ds_test, verbose=2)
print(f'Test accuracy: {test_acc}')