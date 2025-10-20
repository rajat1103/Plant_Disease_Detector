import tensorflow as tf

# Define the path to your original Keras model
keras_model_path = "plant_disease_model.h5"

# Define the path for the new TensorFlow Lite model
tflite_model_path = "plant_disease_model.tflite"

print(f"Loading Keras model from: {keras_model_path}")
model = tf.keras.models.load_model(keras_model_path)
print("Model loaded successfully.")

# Create a TFLite converter object
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# --- THIS IS THE NEW, IMPORTANT PART ---
# Ensure the model is compatible with the tflite-runtime version (e.g., 2.14) on the server
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, # Enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS # Enable TensorFlow ops.
]
# --- END OF NEW PART ---

# Apply standard optimizations
converter.optimizations = [tf.lite.Optimize.DEFAULT]

print("Converting model to a compatible TensorFlow Lite format...")
tflite_model = converter.convert()
print("Conversion complete.")

# Save the new .tflite model to a file
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f"TensorFlow Lite model saved to: {tflite_model_path}")