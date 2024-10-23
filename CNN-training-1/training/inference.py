import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Constants
IMAGE_SIZE = 256
THRESHOLD = 0.01  # Tuning parameter for autoencoder reconstruction error

# Load models
AUTOENCODER = load_model("../saved-models/autoencoder.keras")
CNN_MODEL = load_model("../saved-models/model.keras")


def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# Check if image is potato or non-potato
def is_potato_leaf(img_path):
    img_array = preprocess_image(img_path)
    reconstructed = AUTOENCODER.predict(img_array)
    reconstruction_error = np.mean(np.abs(img_array - reconstructed))
    return reconstruction_error < THRESHOLD


# Classify potato leaf
def classify_potato_leaf(img_path):
    if is_potato_leaf(img_path):
        img_array = preprocess_image(img_path)
        predictions = CNN_MODEL.predict(img_array)
        class_idx = np.argmax(predictions)
        classes = ["Early Blight", "Healthy", "Late Blight"]
        return classes[class_idx]
    else:
        return "Not a potato leaf"


if __name__ == "__main__":
    img_path = "dataset"  
    result = classify_potato_leaf(img_path)
    print(f"Classification Result: {result}")
