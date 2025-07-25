# from flask import Flask, request, jsonify
# import numpy as np
# from PIL import Image, UnidentifiedImageError
# import tensorflow as tf
# import io
# from flask_cors import CORS
# app = Flask(__name__)
# CORS(app)
# # Load the TFLite model
# interpreter = tf.lite.Interpreter(model_path="model.tflite")
# interpreter.allocate_tensors()

# # Get input and output tensors
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# # Log the expected input shape
# expected_shape = input_details[0]['shape']  # e.g., [1, 200, 200, 3]
# print("Expected input shape from model:", expected_shape)
# expected_height, expected_width = expected_shape[1], expected_shape[2]

# def preprocess_image(image_bytes):
#     try:
#         image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
#     except UnidentifiedImageError:
#         raise ValueError("Uploaded file is not a valid image.")

#     image = image.resize((expected_width, expected_height))  # model expects 200x200
#     image_array = np.array(image, dtype=np.float32) / 255.0
#     image_array = np.expand_dims(image_array, axis=0)
#     return image_array

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image file provided'}), 400

#     image_file = request.files['image']
#     if image_file.filename == '':
#         return jsonify({'error': 'No file selected'}), 400

#     if not image_file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
#         return jsonify({'error': 'Only JPG/PNG images allowed'}), 400

#     try:
#         image_bytes = image_file.read()

#         # Debug: Save raw image file
#         with open("debug_uploaded_image.jpg", "wb") as f:
#             f.write(image_bytes)

#         input_tensor = preprocess_image(image_bytes)

#         print("Input tensor shape:", input_tensor.shape)
#         print("Input tensor min/max:", np.min(input_tensor), np.max(input_tensor))

#         interpreter.set_tensor(input_details[0]['index'], input_tensor)
#         interpreter.invoke()

#         output = interpreter.get_tensor(output_details[0]['index'])[0]

#         print("Raw model output:", output)

#         predicted_class = int(np.argmax(output))
#         confidence = float(np.max(output))

#         print("Predicted class:", predicted_class, "Confidence:", confidence)

#         return jsonify({
#             'predicted_class': predicted_class,
#             'confidence': round(confidence, 4)
#         })

#     except ValueError as e:
#         return jsonify({'error': str(e)}), 400
#     except Exception as e:
#         return jsonify({'error': 'Internal server error: ' + str(e)}), 500

# if __name__ == '__main__':
#     app.run(port=5050, debug=True)
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
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
        with open(MODEL_PATH, 'wb') as f:
            f.write(response.content)
        print("✅ Model downloaded.")

download_model()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load model
print("🔍 Loading model...")
model = load_model(MODEL_PATH)
print("✅ Model loaded.")

# Define class labels
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

# Preprocessing
def preprocess_image(image_bytes, target_size=(224, 224)):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('L')  # grayscale
    except UnidentifiedImageError:
        raise ValueError("Invalid image format.")
    image = cv2.resize(np.array(image), target_size)
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Inference route
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

# Start the server
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5050))
    app.run(host='0.0.0.0', port=port)
