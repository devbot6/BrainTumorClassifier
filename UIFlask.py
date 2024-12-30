
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from flask import Flask, request, jsonify, render_template, render_template_string

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



app = Flask(__name__)

# @app.route('/')
# def upload_page():
#     return render_template("upload.html")  # HTML file for user upload

@app.route('/')
def upload_page():
    return render_template_string("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Brain Tumor Classification</title>
    </head>
    <body>
        <h1>Brain Tumor Classification</h1>
        <form action="/predict" method="post" enctype="multipart/form-data">
            <label for="file">Upload MRI Scan:</label>
            <input type="file" name="file" id="file" accept="image/*" required>
            <br><br>
            <button type="submit">Classify</button>
        </form>
    </body>
    </html>
    """)

@app.route('/predict', methods=['POST'])
def upload_and_predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file:
        file_path = f"./uploads/{file.filename}"  # Save file to a local folder
        file.save(file_path)

        # Make prediction
        predicted_class, confidence = predict_tumor(file_path)
        return jsonify({
            "class": predicted_class,
            "confidence": f"{confidence * 100:.2f}%"
        })

if __name__ == '__main__':
    app.run(debug=True)
