import tensorflow as tf
from tensorflow.keras import layers, models
import os

IMAGE_SIZE = 256
BATCH_SIZE = 32


# Load dataset
def load_data(data_dir):
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        label_mode=None,  
    )
    dataset = dataset.map(lambda x: x / 255.0)  
    return dataset


def build_autoencoder():
    model = models.Sequential(
        [
            layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2), padding="same"),
            layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2), padding="same"),
            layers.Conv2D(8, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2), padding="same"),
            layers.Conv2D(8, (3, 3), activation="relu", padding="same"),
            layers.UpSampling2D((2, 2)),
            layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
            layers.UpSampling2D((2, 2)),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.UpSampling2D((2, 2)),
            layers.Conv2D(3, (3, 3), activation="sigmoid", padding="same"),
        ]
    )
    return model


# Compile and Train Autoencoder
def train_autoencoder(data_dir):
    dataset = load_data(data_dir)
    autoencoder = build_autoencoder()
    autoencoder.compile(optimizer="adam", loss="mse")
    autoencoder.fit(dataset, epochs=20, verbose=1)
    autoencoder.save("../saved-models/autoencoder.keras")


if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    data_dir = os.path.join(current_dir, "../plantvillage")
    train_autoencoder(data_dir)
