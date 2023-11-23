import numpy as np

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Dropout, AveragePooling2D, BatchNormalization, Activation, Add, MaxPooling2D
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt

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

# Settings
resize = (256, 256)

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

    return tf.cast(image, tf.float32) / 255., label

# Normalise images
ds_train = train_dataset.map(normalize_img)
ds_test = test_dataset.map(normalize_img)

# # Data augmentation
# def data_augment(x):
#     x = RandomRotation(0.2)(x)
#     x = RandomTranslation(0.2, 0.2)(x)
#     return x

# x_train = data_augment(x_train)

# -------------- Declare Layers --------------

initializer = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=42)

def identity_block(filter, kernel_size=3):
    def _identity_block(x):
        input_x = x
        # Layer 1
        x = Conv2D(filter, kernel_size=kernel_size, padding = 'same', kernel_initializer=initializer)(x)
        x = BatchNormalization(axis=3)(x)
        x = Activation('selu')(x)
        # Layer 2
        x = Conv2D(filter, kernel_size=kernel_size, padding = 'same', kernel_initializer=initializer)(x)
        x = BatchNormalization(axis=3)(x)
        # Add Residue
        x = Add()([x, input_x])
        # Activation
        x = Activation('selu')(x)
        return x
    
    return _identity_block

def conv_res_block(filter, kernel_size=3):
    def _conv_res_block(x):
        input_x = x

        # Layer 1
        x = Conv2D(filter, kernel_size=kernel_size, padding = 'same', kernel_initializer=initializer)(x)
        x = BatchNormalization(axis=3)(x)
        x = Activation('selu')(x)
        # Layer 2
        x = Conv2D(filter, kernel_size=kernel_size, padding = 'same', kernel_initializer=initializer)(x)
        x = BatchNormalization(axis=3)(x)

        # Residue Layer
        input_x = Conv2D(filter, kernel_size=1, padding = 'same', kernel_initializer=initializer)(input_x)

        # Add Residue
        x = Add()([x, input_x])
        
        # Activation
        x = Activation('selu')(x)

        return x
    
    return _conv_res_block

# -------------- Declare the model --------------

input_shape = (resize[0], resize[1], 3)
input_layer = Input(shape=input_shape)

# Feature Extractor
def conv_block(filter, kernel_size = 3, stride = 2):
    def _conv_block(x):
        x = Conv2D(filter, kernel_size=kernel_size, strides=stride, padding='same', kernel_initializer=initializer)(x)
        x = BatchNormalization()(x)
        x = Activation('selu')(x)
        return x
    return _conv_block

def conv_pool_block(filter, kernel_size = 3, stride = 2, pool_size = 2, pool_stride = 1):
    def _conv_pool_block(x):
        x = Conv2D(filter, kernel_size=kernel_size, strides=stride, padding='same', kernel_initializer=initializer)(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=pool_size, strides=pool_stride, padding = 'same')(x)
        x = Activation('selu')(x)
        return x
    return _conv_pool_block

# conv_net = [
#     conv_block(64, kernel_size = 7, stride = 2),
#     conv_res_block(64),
#     conv_res_block(64),
#     conv_pool_block(128, kernel_size = 5, stride = 2),
#     conv_res_block(128),
#     conv_res_block(128),
#     conv_block(256, kernel_size = 3, stride = 2),
#     conv_res_block(256),
#     conv_res_block(256),
#     conv_pool_block(256, kernel_size = 3, stride = 2),
#     AveragePooling2D(pool_size=(4, 4), padding = 'same'),
# ] # 0.71

conv_net = [
    conv_pool_block(64, kernel_size = 7, stride = 2, pool_size=3, pool_stride=2),
    identity_block(64),
    identity_block(64),

    AveragePooling2D(pool_size=(2, 2), padding = 'same'),
    
    conv_res_block(128),
    identity_block(128),

    AveragePooling2D(pool_size=(2, 2), padding = 'same'),

    conv_res_block(256),
    identity_block(256),

    AveragePooling2D(pool_size=(4, 4), padding = 'same'),

    conv_res_block(512),
    identity_block(512),
    
    conv_pool_block(512, kernel_size = 3, stride = 2, pool_size=2, pool_stride=2)
] # res_net_3 - 0.8447

# res_net_2 - 0.01


# Classifier
classification_net = [
    Flatten(),
    Dense(512, activation='selu', kernel_initializer=initializer),
    Dropout(0.2),
    Dense(num_classes, activation='softmax', kernel_initializer=initializer)
]

# -------------- Prepare for Train and Test --------------

# Train settings
epochs = 50
batch_size = 8

# Configure Dataset for Performance
AUTOTUNE = tf.data.experimental.AUTOTUNE

# ds_train = ds_train.cache()
ds_train = ds_train.batch(batch_size)
ds_train = ds_train.prefetch(buffer_size=AUTOTUNE)

ds_test = ds_test.batch(batch_size)
ds_test = ds_test.prefetch(buffer_size=AUTOTUNE)

del x_train, y_train, x_test, y_test

def create_classifier():

    def compile_layers(input, layers):
        for layer in layers:
            input = layer(input)
        return input

    # Build Feature Extractor
    conv_net_layers = compile_layers(input_layer, conv_net)

    # Build Classifier
    classification_layers = compile_layers(conv_net_layers, classification_net)

    # Build model
    model = Model(inputs=input_layer, outputs=classification_layers)

    return model

def train_test_model(model, model_name='models/model.h5'):

    # Define optimizer
    optimizer = 'adam'

    # Compile model
    model.compile(optimizer=optimizer, loss=CategoricalCrossentropy(label_smoothing=0.2), metrics=['accuracy'])

    # Define a ModelCheckpoint callback
    checkpoint = ModelCheckpoint(model_name, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

    # Train model
    run_hist = model.fit(ds_train, epochs=epochs, batch_size=batch_size, validation_data=ds_test, callbacks=[checkpoint])
    
    return run_hist

# -------------- Train and Test the model --------------

# Create model
model = create_classifier()

model.summary()

# Run train and validation
run_hist = train_test_model(model)

# Run test
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
print(f'Test accuracy: {test_acc}')

# Plot accuracy and loss
plt.plot(run_hist.history['accuracy'], label='train')
plt.plot(run_hist.history['val_accuracy'], label='test')
plt.plot(run_hist.history['loss'], label='loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()