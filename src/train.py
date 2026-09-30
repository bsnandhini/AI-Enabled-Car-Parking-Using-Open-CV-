import numpy as np
import random as rn
import tensorflow as tf
import os
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau

from data_loader import X,Y
from cnn_model import build_model

# Project base directory 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#Fix random seed
np.random.seed(42)
rn.seed(42)
tf.random.set_seed(42)

# Separate data
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.25,
    random_state=42
)

# Build CNN model
model = build_model()

# Training configuration
batch_size = 128
epochs = 10

# Use callback only ReduceLROnPlateau
red_lr = ReduceLROnPlateau(
    monitor='val_accuracy',
    patience=3,
    verbose=1,
    factor=0.1
)

# Data augmentation to prevent overfitting
datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=False
)

datagen.fit(x_train)


# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Display model summary
model.summary()

# Train model
History = model.fit(
    datagen.flow(
        x_train,
        y_train,
        batch_size = batch_size
    ),
    epochs= epochs,
    validation_data=(x_test,y_test),
    verbose=1,
    steps_per_epoch=x_train.shape[0] // batch_size,
    callbacks=[red_lr]
)



# Save model
model_dir = os.path.join(BASE_DIR, "models")
os.makedirs(model_dir, exist_ok=True)
model.save(os.path.join(model_dir, "first_cnn.h5"))

# Save dataset sample image
output_dir = os.path.join(BASE_DIR, "output_images")
os.makedirs(output_dir, exist_ok=True)

plt.savefig(os.path.join(output_dir, "dataset_sample.png"))