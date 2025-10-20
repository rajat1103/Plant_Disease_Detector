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
# Apply standard optimizations
converter.optimizations = [tf.lite.Optimize.DEFAULT]

print("Converting model to TensorFlow Lite format...")
tflite_model = converter.convert()
print("Conversion complete.")
 # Save the new .tflite model to a file
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)
print(f"TensorFlow Lite model saved to: {tflite_model_path}")
