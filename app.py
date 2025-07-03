import os
import numpy as np
from PIL import Image
import tensorflow as tf
from flask import Flask, request, render_template, jsonify

# --- Configuration and Initialization ---

# Initialize the Flask application
app = Flask(__name__)

# Define the path for uploaded images
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define the path to the trained model
MODEL_PATH = 'retinal_model_multilabel.h5'

# Define the image dimensions expected by the model
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Define the class labels in the exact order the model was trained on
# This is crucial for correctly interpreting the model's output.
CLASS_LABELS = [
    'Normal',
    'Diabetes',
    'Glaucoma',
    'Cataract',
    'Age related Macular Degeneration',
    'Hypertension',
    'Pathological Myopia',
    'Other diseases/abnormalities'
]

# --- Model Loading ---

# Load the trained deep learning model
# Use a try-except block to handle potential errors during model loading.
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# --- Helper Functions ---

def preprocess_image(image_path):
    """
    Loads and preprocesses an image file to be model-ready.
    Args:
        image_path (str): The path to the image file.
    Returns:
        np.ndarray: A preprocessed image as a NumPy array.
    """
    try:
        # Open the image file using Pillow
        img = Image.open(image_path).convert('RGB')
        # Resize the image to the target dimensions
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))
        # Convert the image to a NumPy array
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        # Normalize the pixel values to the [0, 1] range
        img_array /= 255.0
        # Expand the dimensions to create a batch of size 1
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None

# --- Flask Routes ---

@app.route('/', methods=['GET'])
def index():
    """
    Renders the main page with the file upload form.
    """
    # The 'index.html' template should be in a 'templates' folder.
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles the image upload and prediction process.
    Accepts multiple files and returns predictions in JSON format or renders them on the page.
    """
    if model is None:
        return "Model not loaded. Please check server logs.", 500

    # Get the list of files from the POST request
    uploaded_files = request.files.getlist('files')
    if not uploaded_files or uploaded_files[0].filename == '':
        return "No files selected.", 400

    results = []
    for file in uploaded_files:
        try:
            # Create a secure path for the uploaded file
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Preprocess the image for the model
            preprocessed_image = preprocess_image(filepath)
            
            if preprocessed_image is not None:
                # Get model predictions (probabilities)
                predictions = model.predict(preprocessed_image)[0]

                # Map probabilities to class labels
                prediction_dict = {CLASS_LABELS[i]: float(predictions[i]) for i in range(len(CLASS_LABELS))}
                
                results.append({
                    'filename': file.filename,
                    'predictions': prediction_dict
                })
        except Exception as e:
            print(f"Prediction failed for file {file.filename}: {e}")
            results.append({
                'filename': file.filename,
                'error': 'Failed to process this image.'
            })

    # For API usage, you might want to return JSON directly:
    # return jsonify(results)

    # For web interface usage, render the results in the template:
    return render_template('index.html', results=results)

# --- Main Application Runner ---

if __name__ == '__main__':
    # Run the Flask app
    # debug=True allows for auto-reloading on code changes.
    # Set debug=False in a production environment.
    app.run(debug=True)