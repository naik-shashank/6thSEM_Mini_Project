from fastapi import FastAPI, File, UploadFile
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
from io import BytesIO

# Initialize FastAPI app
app = FastAPI()

# Load the trained deepfake detector model
model = load_model("deepfake_detector.h5")

def preprocess_image(file):
    """Convert uploaded image into a format suitable for the model"""
    img = Image.open(BytesIO(file))  # Open image from binary data
    img = img.resize((128, 128))  # Resize to match model input size
    img_array = image.img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    return img_array

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """API endpoint to predict if an image is deepfake or real"""
    try:
        # Read and preprocess the image
        contents = await file.read()
        img_array = preprocess_image(contents)

        # Make prediction using the model
        prediction = model.predict(img_array)

        # Interpret results
        result = "Fake" if prediction[0][0] < 0.5 else "Real"

        return {
            "filename": file.filename,
            "prediction": result,
            "confidence": float(prediction[0][0])  # Convert numpy float to Python float
        }

    except Exception as e:
        return {"error": str(e)}
