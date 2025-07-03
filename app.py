# app.py

import os
import numpy as np
from PIL import Image
import tensorflow as tf
from flask import Flask, request, render_template
import fitz  # PyMuPDF
import google.generativeai as genai
import shutil

# --- Configuration and Initialization ---

# Initialize the Flask application
app = Flask(__name__)

# Define paths
UPLOAD_FOLDER = 'uploads'
TEMP_IMG_FOLDER = 'temp_images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEMP_IMG_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define the path to the trained model and image dimensions
MODEL_PATH = 'retinal_model_multilabel.h5'
IMG_HEIGHT, IMG_WIDTH = 224, 224

# Define class labels
CLASS_LABELS = [
    'Normal', 'Diabetes', 'Glaucoma', 'Cataract',
    'Age related Macular Degeneration', 'Hypertension',
    'Pathological Myopia', 'Other diseases/abnormalities'
]

# --- Model and API Loading ---

# Load the trained deep learning model
try:
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully.")
    else:
        print(f"Model file not found at {MODEL_PATH}")
        model = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Configure Google Gemini API
try:
    gemini_api_key = os.environ.get('GEMINI_API_KEY')
    if gemini_api_key:
        genai.configure(api_key=gemini_api_key)
        gemini_model = genai.GenerativeModel('gemini-pro')
        print("Gemini API configured successfully.")
    else:
        gemini_model = None
        print("GEMINI_API_KEY environment variable not found.")
except Exception as e:
    gemini_model = None
    print(f"Error configuring Gemini API: {e}")


# --- Helper Functions ---

def preprocess_image(image_path):
    """Loads and preprocesses an image file to be model-ready."""
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array /= 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None

def extract_from_pdf(pdf_path):
    """Extracts text and images from a PDF file."""
    text = ""
    image_paths = []
    try:
        doc = fitz.open(pdf_path)
        # Extract text
        for page in doc:
            text += page.get_text("text") + "\n"
        
        # Extract images
        for i in range(len(doc)):
            for img in doc.get_page_images(i):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_path = os.path.join(TEMP_IMG_FOLDER, f"img_{xref}.{image_ext}")
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                image_paths.append(image_path)
        doc.close()
    except Exception as e:
        print(f"Error extracting from PDF {pdf_path}: {e}")
    return text, image_paths

def get_gemini_overview(report_text, image_analysis):
    """Generates a patient overview using the Gemini API."""
    if not gemini_model:
        return "Gemini API not configured. Cannot generate overview."

    # Create a detailed prompt for the LLM
    prompt = (
        "You are a helpful medical assistant. Based on the following patient report and retinal scan analysis, "
        "provide a concise and easy-to-understand summary for a healthcare professional. \n\n"
        "### Patient Report Text:\n"
        f"{report_text}\n\n"
        "### Retinal Scan Analysis Results:\n"
    )
    for result in image_analysis:
        prompt += f"- Image: {result['filename']}\n  Probabilities:\n"
        for disease, prob in result['predictions'].items():
            prompt += f"    - {disease}: {prob*100:.2f}%\n"

    prompt += "\n### Summary Request:\n"
    prompt += "Please provide a final overview of the patient's potential condition based on all the provided data."

    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return "Failed to generate AI overview due to an API error."

# --- Flask Routes ---

@app.route('/', methods=['GET'])
def index():
    """Renders the main page."""
    # Clean up temp folder from previous runs
    if os.path.exists(TEMP_IMG_FOLDER):
        shutil.rmtree(TEMP_IMG_FOLDER)
        os.makedirs(TEMP_IMG_FOLDER)
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles file uploads and the full analysis pipeline."""
    if model is None:
        return render_template('index.html', error="Retinal analysis model not loaded. Please check server logs.")

    uploaded_files = request.files.getlist('files')
    if not uploaded_files or uploaded_files[0].filename == '':
        return render_template('index.html', error="No files selected.")

    results = []
    for file in uploaded_files:
        try:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # --- Initialize result structure ---
            result_data = {
                'filename': filename,
                'extracted_text': "No text extracted.",
                'image_analyses': [],
                'ai_overview': "Not generated.",
                'error': None
            }

            if filename.lower().endswith('.pdf'):
                # --- PDF Processing ---
                extracted_text, image_paths = extract_from_pdf(filepath)
                result_data['extracted_text'] = extracted_text if extracted_text.strip() else "No text found in PDF."

                if not image_paths:
                    result_data['error'] = "No images found in the uploaded PDF."
                else:
                    for img_path in image_paths:
                        preprocessed_image = preprocess_image(img_path)
                        if preprocessed_image is not None:
                            predictions = model.predict(preprocessed_image)[0]
                            prediction_dict = {CLASS_LABELS[i]: float(predictions[i]) for i in range(len(CLASS_LABELS))}
                            result_data['image_analyses'].append({
                                'filename': os.path.basename(img_path),
                                'predictions': prediction_dict
                            })
                
                # Generate Gemini overview if there's data
                if result_data['extracted_text'] or result_data['image_analyses']:
                    result_data['ai_overview'] = get_gemini_overview(result_data['extracted_text'], result_data['image_analyses'])

            else:
                # --- Direct Image Processing ---
                preprocessed_image = preprocess_image(filepath)
                if preprocessed_image is not None:
                    predictions = model.predict(preprocessed_image)[0]
                    prediction_dict = {CLASS_LABELS[i]: float(predictions[i]) for i in range(len(CLASS_LABELS))}
                    result_data['image_analyses'].append({
                        'filename': filename,
                        'predictions': prediction_dict
                    })
                    result_data['ai_overview'] = get_gemini_overview("No report text provided (image upload).", result_data['image_analyses'])
                else:
                    result_data['error'] = 'Could not preprocess the uploaded image.'
            
            results.append(result_data)

        except Exception as e:
            print(f"Processing failed for file {file.filename}: {e}")
            results.append({
                'filename': file.filename,
                'error': 'A critical error occurred during processing.'
            })

    return render_template('index.html', results=results)

# --- Main Application Runner ---

if __name__ == '__main__':
    app.run(debug=False)