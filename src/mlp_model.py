import numpy as np 
import pandas as pd 
import os 
import cv2

from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras

# Dataset path 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset = os.path.join(BASE_DIR, "dataset")

# Get data types 
data_types = os.listdir(dataset) 
print(data_types) 
print("Types of data found:", len(data_types))

# Create dataframe
rs =[]

for item in data_types:
    data_path = os.path.join(dataset, item)

    for image_name in os.listdir(data_path):
        image_path = os.path.join(data_path, image_name)
        rs.append((item , image_path))

    rs_df = pd.DataFrame(rs, columns=["rm_type", "image"])

print(rs_df.head()) 
print(rs_df.tail())

print("Total number of images in the dataset:", len(rs_df))

# Count images in each category 
rm_count = rs_df["rm_type"].value_counts() 
print("Images in each category:") 
print(rm_count) 

# Load images 
im_size = 300 

images = [] 
labels = []

for item in data_types:
    data_path = os.path.join(dataset, item)

    for filename in os.listdir(data_path):
        image_path = os.path.join(data_path, filename)

        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)

        if img is None:
            continue

        img = cv2.resize(img, (im_size, im_size))

        images.append(img)
        labels.append(item)

# Convert images to numpy array 
images = np.array(images)

print("Images shape:", images.shape)

# Normalize pixel values 
images = images.astype("float32") / 255.0

# Encode labels 
y_labelencoder = LabelEncoder()
y = y_labelencoder.fit_transform(labels)

print("Encoded labels:", y)


# Shuffle data 
images, y = shuffle(images, y, random_state=1)

# Train-test split 
train_x, test_x, train_y, test_y = train_test_split( images, y, test_size=0.05, random_state=1, stratify=y )

# Check shapes 
print("Training images:", train_x.shape)
print("Training labels:", train_y.shape) 
print("Testing images:", test_x.shape) 
print("Testing labels:", test_y.shape)

# Build MLP model 
model = keras.Sequential([ 
    keras.layers.Input(shape=(300, 300, 3)), 
    keras.layers.Flatten(), 
    keras.layers.Dense(256, activation="tanh"), 
    keras.layers.Dense(2, activation="softmax") 
])

# Compile model 
model.compile( optimizer=tf.keras.optimizers.Adam(), loss="sparse_categorical_crossentropy", metrics=["accuracy"] )

# Train model 
history = model.fit( train_x, train_y, epochs=20, validation_data=(test_x, test_y) )

# Evaluate model 
test_loss, test_accuracy = model.evaluate(test_x, test_y)

print("Test Loss:", test_loss) 
print("Test Accuracy:", test_accuracy)

# Predictions 
y_pred = np.argmax(model.predict(test_x), axis=1)

print("Predicted labels:", y_pred)

# Save model
model.save("mlp_model.keras") 
print("MLP model saved successfully.")