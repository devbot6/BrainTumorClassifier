from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from flask import Flask, request, jsonify, render_template_string
import os

# Load the trained model
model = load_model("brain_tumor_classifier.h5")

# Ensure the class names are consistent
class_names = ["no_tumor", "pituitary_tumor", "glioma_tumor", "meningioma_tumor"]

def predict_tumor(image_path):
    # Load and preprocess the image
    img = load_img(image_path, target_size=(150, 150))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_index]
    confidence = predictions[0][predicted_class_index]

    return predicted_class, confidence

app = Flask(__name__)

# Ensure upload directory exists
os.makedirs('./uploads', exist_ok=True)

@app.route('/')
def upload_page():
    # Copy the entire content of the HTML artifact here
    return render_template_string("""
    <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NeuroScan AI | Brain Tumor Classification</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/tailwindcss/2.2.19/tailwind.min.css" rel="stylesheet">
    <style>
        .upload-area {
            border: 2px dashed #CBD5E0;
            transition: all 0.3s ease;
        }
        .upload-area:hover {
            border-color: #4299E1;
            background-color: #EBF8FF;
        }
        .result-card {
            opacity: 0;
            transform: translateY(20px);
            transition: all 0.5s ease;
        }
        .result-card.show {
            opacity: 1;
            transform: translateY(0);
        }
    </style>
</head>
<body class="bg-gray-50 min-h-screen">
    <div class="container mx-auto px-4 py-8">
        <!-- Header -->
        <header class="text-center mb-12">
            <h1 class="text-4xl font-bold text-gray-800 mb-2">NeuroScan AI</h1>
            <p class="text-gray-600">Advanced Brain Tumor Classification System</p>
        </header>

        <!-- Main Content -->
        <div class="max-w-2xl mx-auto bg-white rounded-lg shadow-lg p-8">
            <form action="/predict" method="post" enctype="multipart/form-data" id="upload-form">
                <!-- Upload Area -->
                <div class="upload-area rounded-lg p-8 text-center mb-8">
                    <div class="mb-4">
                        <svg class="mx-auto h-12 w-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"/>
                        </svg>
                        <h3 class="mt-2 text-sm font-medium text-gray-900">Upload MRI Scan</h3>
                        <p class="mt-1 text-sm text-gray-500">Supported formats: PNG, JPEG, DICOM</p>
                    </div>
                    
                    <input type="file" name="file" id="file" accept="image/*" required
                           class="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4
                                  file:rounded-full file:border-0 file:text-sm file:font-semibold
                                  file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"/>
                </div>

                <!-- Submit Button -->
                <button type="submit" 
                        class="w-full bg-blue-600 text-white py-3 px-6 rounded-lg
                               hover:bg-blue-700 transition duration-200 flex items-center justify-center">
                    <span class="mr-2">Analyze Scan</span>
                    <svg class="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"/>
                    </svg>
                </button>
            </form>

            <!-- Results Section (Hidden by default) -->
            <div id="results" class="result-card mt-8 p-6 bg-gray-50 rounded-lg hidden">
                <h3 class="text-lg font-semibold text-gray-800 mb-4">Analysis Results</h3>
                <div class="flex items-center justify-between">
                    <div>
                        <p class="text-sm text-gray-600">Classification:</p>
                        <p id="classification" class="text-lg font-bold text-blue-600"></p>
                    </div>
                    <div>
                        <p class="text-sm text-gray-600">Confidence:</p>
                        <p id="confidence" class="text-lg font-bold text-green-600"></p>
                    </div>
                </div>
            </div>
        </div>

        <!-- Footer -->
        <footer class="text-center mt-12 text-gray-500 text-sm">
            <p>© 2024 NeuroScan AI. For research and educational purposes only.</p>
        </footer>
    </div>

    <script>
        document.getElementById('upload-form').onsubmit = async (e) => {
            e.preventDefault();
            
            const formData = new FormData(e.target);
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                // Update and show results
                document.getElementById('classification').textContent = data.class.replace(/_/g, ' ').toUpperCase();
                document.getElementById('confidence').textContent = data.confidence;
                
                const results = document.getElementById('results');
                results.classList.remove('hidden');
                setTimeout(() => results.classList.add('show'), 100);
                
            } catch (error) {
                console.error('Error:', error);
                alert('An error occurred during analysis. Please try again.');
            }
        };
    </script>
</body>
</html>
    """)

@app.route('/predict', methods=['POST'])
def upload_and_predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file and file.filename:
        file_path = os.path.join('./uploads', file.filename)
        file.save(file_path)

        try:
            predicted_class, confidence = predict_tumor(file_path)
            
            # Clean up the uploaded file
            os.remove(file_path)
            
            return jsonify({
                "class": predicted_class,
                "confidence": f"{confidence * 100:.2f}%"
            })
        except Exception as e:
            # Clean up the uploaded file in case of error
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Invalid file"}), 400

if __name__ == '__main__':
    app.run(debug=True)