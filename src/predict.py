import os
import cv2
import numpy as np
import tensorflow as tf

# Project base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Loading the saved CNN model 
MODEL_PATH = os.path.join(BASE_DIR, "models", "first_cnn.h5")

# Image size used in the CNN model
IMG_SIZE = 150

model = tf.keras.models.load_model(MODEL_PATH)

def predict_image(image_path):
    #Read image
    img = cv2.imread(image_path , cv2.IMREAD_COLOR)

    if img is None:
        raise FileNotFoundError("Image not found: " + image_path)

    # Resize image 
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Convert image to numpy array 
    img = np.array(img)

    # Normalize image i
    img = img / 255.0

    # Add batch dimension 
    img = np.expand_dims(img, axis=0)

    # Getting prediction 
    pred = model.predict(img, verbose=0)

    # Get predicted class 
    pred_digit = np.argmax(pred, axis=1)[0]

    # Class labels from the notebook 
    if pred_digit == 0: 
        return "Free" 
    else: 
        return "Full"