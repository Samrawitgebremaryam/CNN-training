import tensorflow as tf
from tensorflow.keras import layers, models
import os

# Constants
IMAGE_SIZE = 256
BATCH_SIZE = 32
NUM_CLASSES = 3  # Early Blight, Late Blight, Healthy

# Load dataset
def load_data(data_dir):
    return tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="categorical"  
    )

# Define CNN Model
def build_cnn():
    model = models.Sequential([
        layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dense(NUM_CLASSES, activation="softmax")  # 3 classes
    ])
    return model

# Compile and Train CNN
def train_cnn(data_dir):
    dataset = load_data(data_dir)
    cnn_model = build_cnn()
    cnn_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    cnn_model.fit(dataset, epochs=20, verbose=1)
    cnn_model.save("../saved-models/model.keras")

if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    data_dir = os.path.join(current_dir, "../dataset")
    train_cnn(data_dir)
