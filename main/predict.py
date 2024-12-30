from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the trained model
model = load_model("brain_tumor_classifier.h5")

# Ensure the class names are consistent
class_names = ["no_tumor", "pituitary_tumor", "glioma_tumor", "meningioma_tumor"]

def predict_tumor(image_path):
    # Load and preprocess the image
    img = load_img(image_path, target_size=(150, 150))  # Resize to match the model's input size
    img_array = img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_index]
    confidence = predictions[0][predicted_class_index]

    return predicted_class, confidence

image_path = "path_to_user_uploaded_image.jpg"  # Replace with the actual file path
predicted_class, confidence = predict_tumor(image_path)

print(f"Predicted Class: {predicted_class}")
print(f"Confidence: {confidence * 100:.2f}%")

