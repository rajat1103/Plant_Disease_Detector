# Plant Disease Detector

This project is a web application that uses a machine learning model to detect diseases in plants. Users can upload an image of a plant leaf, and the application will identify the disease and provide a confidence score.

## How to Run the Project

Follow these steps to run the project locally:

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Plant_Disease_Detector.git
cd Plant_Disease_Detector
```

### 2. Create a Virtual Environment

It's recommended to use a virtual environment to manage dependencies.

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies

Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

### 4. Run the Application

Start the Flask development server:

```bash
python app.py
```

The application will be accessible at `http://127.0.0.1:5000` in your web browser.

### 5. Using the Application

- Open your web browser and navigate to `http://127.0.0.1:5000`.
- Click on "Choose an Image" to upload an image of a plant leaf.
- Click the "Predict Disease" button to see the analysis result.

The model will be downloaded automatically the first time you make a prediction.
