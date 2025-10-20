import os
import requests
import numpy as np
from flask import Flask, request, render_template, send_from_directory
import tflite_runtime.interpreter as tflite
from PIL import Image

# --- TFLITE MODEL DOWNLOAD LOGIC ---
MODEL_PATH = "plant_disease_model.tflite" # Using .tflite model now
UPLOADS_DIR = "static/uploads"
# Your new URL is now included:
MODEL_URL = "https://huggingface.co/rishabh110304/Plant_Disease_Detection/resolve/main/plant_disease_model_v2.tflite?download=true" 

if not os.path.exists(MODEL_PATH):
    print(f"Model not found at {MODEL_PATH}. Downloading from {MODEL_URL}...")
    try:
        response = requests.get(MODEL_URL)
        response.raise_for_status()  # Raise an exception for bad status codes
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        print("Model downloaded successfully.")
    except Exception as e:
        print(f"Failed to download model: {e}")
else:
    print(f"Model file '{MODEL_PATH}' already exists.")

if not os.path.exists(UPLOADS_DIR):
    os.makedirs(UPLOADS_DIR)
# --- END OF DOWNLOAD LOGIC ---


# --- 1. Initialize Flask App & Load TFLite Model ---
app = Flask(__name__)

print("Loading TFLite model...")
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Model loaded successfully!")

# --- 2. Define Constants ---
IMG_HEIGHT = input_details[0]['shape'][1] # Get size from model
IMG_WIDTH = input_details[0]['shape'][2]
class_names = [ # (Your class names remain the same)
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# --- 3. Define the Prediction Function (Updated for TFLite) ---
def predict_image(img_path):
    img = Image.open(img_path).resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    
    interpreter.set_tensor(input_details[0]['index'], img_batch)
    interpreter.invoke()
    
    prediction = interpreter.get_tensor(output_details[0]['index'])
    predicted_class_index = np.argmax(prediction[0])
    predicted_class_name = class_names[predicted_class_index]
    confidence = np.max(prediction[0])
    
    return predicted_class_name, f"{confidence*100:.2f}"

# --- 4. Define Routes ---
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', prediction="No image file selected.")
        
    file = request.files['image']
    
    if file.filename == '':
        return render_template('index.html', prediction="No image file selected.")
        
    if file:
        filename = file.filename
        file_path = os.path.join(UPLOADS_DIR, filename)
        file.save(file_path)
        
        predicted_class, confidence = predict_image(file_path)
        
        return render_template('index.html', 
                               prediction=predicted_class, 
                               confidence=confidence,
                               image_path=os.path.join(UPLOADS_DIR, filename))

@app.route('/static/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOADS_DIR, filename)

# --- 5. Run the App ---
if __name__ == '__main__':
    app.run(debug=True)

