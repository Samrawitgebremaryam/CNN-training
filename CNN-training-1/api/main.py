from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

# Initialize FastAPI app
app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the autoencoder and CNN models
autoencoder = load_model("../saved-models/autoencoder.keras")
cnn_model = load_model("../saved-models/model.keras")

# Define the class names for CNN
class_names = ["early_blight", "healthy", "late_blight"]


# Function to preprocess images for the autoencoder
def preprocess_image_for_autoencoder(img: Image.Image):
    img = img.resize((128, 128))
    img = img.convert("RGB")
    img_array = np.array(img) / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# Function to preprocess images for the CNN
def preprocess_image_for_cnn(img: Image.Image):
    img = img.resize((224, 224))
    img = img.convert("RGB")
    img_array = np.array(img) / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    return img_array


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    autoencoder_input = preprocess_image_for_autoencoder(img)

    reconstruction = autoencoder.predict(autoencoder_input)
    reconstruction_error = np.mean(np.square(autoencoder_input - reconstruction))

    threshold = 0.01

    if reconstruction_error > threshold:
        return JSONResponse(content={"message": "Not a potato leaf."})

    cnn_input = preprocess_image_for_cnn(img)

    cnn_predictions = cnn_model.predict(cnn_input)
    predicted_class_index = np.argmax(cnn_predictions, axis=1)
    predicted_class = class_names[predicted_class_index[0]]

    return JSONResponse(
        content={
            "prediction": predicted_class,
            "confidence": cnn_predictions[0][predicted_class_index[0]],
        }
    )


# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
