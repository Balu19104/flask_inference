from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
import cv2
from PIL import Image, UnidentifiedImageError
import io
from flask_cors import CORS
import os
import requests

# Constants
MODEL_PATH = 'plant_diseases.h5'
GDRIVE_MODEL_URL = "https://drive.google.com/uc?export=download&id=1zNhFMKgJcLWeVQr-eRy8E23QFDWNWDiT"

# Ensure model exists
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("🔽 Downloading model from Google Drive...")
        response = requests.get(GDRIVE_MODEL_URL)
        if response.status_code == 200:
            with open(MODEL_PATH, 'wb') as f:
                f.write(response.content)
            print("✅ Model downloaded.")
        else:
            raise Exception("❌ Failed to download model from Google Drive.")

download_model()

# Initialize Flask app
app = Flask(__name__)
# CORS(app)
from datetime import timedelta

CORS(app, supports_credentials=True, resources={
    r"/predict": {
        "origins": "https://frontend-for-crop-ai-react-app.vercel.app",
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})


# Load model
print("🔍 Loading model...")
model = load_model(MODEL_PATH)
print("✅ Model loaded.")

# Define class labels (15 classes)
class_labels = [
    'Piment: Bacterial_spot', 
    'Piment: healthy', 
    'Pomme de terre: Early_blight', 
    'Pomme de terre: Late_blight', 
    'Pomme de terre: Healthy', 
    'Tomate: Bacterial Spot', 
    'Tomate: Early Blight', 
    'Tomate: Late Blight', 
    'Tomate: Leaf mold', 
    'Tomate: Septoria leaf spot', 
    'Tomate: Siper mites', 
    'Tomate: Spot', 
    "Tomate: Yellow Leaf Curl", 
    'Tomate: Virus Mosaïque', 
    'Tomate: Healthy'
]

# Image preprocessing
def preprocess_image(image_bytes, target_size=(224, 224)):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('L')  # grayscale
    except UnidentifiedImageError:
        raise ValueError("Invalid image format.")
    image = cv2.resize(np.array(image), target_size)
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        image_bytes = image_file.read()
        processed_image = preprocess_image(image_bytes)
        preds = model.predict(processed_image)
        confidences = preds[0]
        max_confidence = float(np.max(confidences))
        predicted_index = int(np.argmax(confidences))
        predicted_label = class_labels[predicted_index]

        return jsonify({
            'predicted_class': predicted_label,
            'confidence': round(max_confidence, 2)
        })

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': 'Inference failed: ' + str(e)}), 500
# ... (existing imports, model loading, and functions)

@app.route('/')
def index():
    return '''
    <h2>🌾 CropAI Backend is Live!</h2>
    <p>Use the <code>/predict</code> endpoint with a POST request to classify plant leaf diseases.</p>
    '''

# Start the server
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5050))
    app.run(host='0.0.0.0', port=port)
