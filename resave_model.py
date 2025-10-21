import tensorflow as tf

# Path to your original, modern .h5 model
keras_model_path = "plant_disease_model.h5"

# Path to the new, universal SavedModel directory
saved_model_dir = "saved_model_directory"

print("Loading the modern Keras model...")
# We use compile=False here as good practice, since we only need the structure and weights
model = tf.keras.models.load_model(keras_model_path, compile=False)
print("Model loaded successfully.")

# --- THE FIX ---
# Save the model in the universal SavedModel format using the new export() method
print(f"Exporting model to universal format at: {saved_model_dir}")
model.export(saved_model_dir)
# --- END OF THE FIX ---

print("Model successfully exported in the universal SavedModel format.")

