import os
import numpy as np
from flask import Flask, request, render_template, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# --- 1. Initialize the Flask App ---
app = Flask(__name__)

# --- 2. Load the Saved Model ---
print("Loading trained model...")
model = load_model('plant_disease_model.h5')
print("Model loaded successfully!")

# --- 3. Define Constants ---
IMG_HEIGHT = 224
IMG_WIDTH = 224
# This list must be in the same order as your training data
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# --- 4. Define the Prediction Function ---
def predict_image(img_path, model):
    img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_batch)
    predicted_class_index = np.argmax(prediction[0])
    predicted_class_name = class_names[predicted_class_index]
    confidence = np.max(prediction[0])
    
    return predicted_class_name, f"{confidence*100:.2f}"

# --- 5. Define the Routes ---

# Home Page Route
@app.route('/', methods=['GET'])
def home():
    # Just render the HTML page
    return render_template('index.html')

# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', prediction="No image file selected.")
        
    file = request.files['image']
    
    if file.filename == '':
        return render_template('index.html', prediction="No image file selected.")
        
    if file:
        # Save the uploaded file to the 'static/uploads' directory
        filename = file.filename
        file_path = os.path.join('static/uploads', filename)
        file.save(file_path)
        
        # Make a prediction
        predicted_class, confidence = predict_image(file_path, model)
        
        # Pass the results and the image path back to the HTML page
        return render_template('index.html', 
                               prediction=predicted_class, 
                               confidence=confidence,
                               image_path=file_path)

# This route is to serve the uploaded images
@app.route('/static/uploads/<filename>')
def send_file(filename):
    return send_from_directory('static/uploads', filename)

# --- 6. Run the App ---
if __name__ == '__main__':
    app.run(debug=True)