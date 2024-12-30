# Brain Tumor Classification App

This is a Flask-based web application that uses a deep learning model to classify brain MRI scans into one of the following categories:
- No Tumor
- Pituitary Tumor
- Glioma Tumor
- Meningioma Tumor

Users can upload their MRI scans, and the application will predict the presence and type of tumor with confidence scores.

---

## Features

- **Deep Learning Model**: Built using TensorFlow/Keras with Convolutional Neural Networks (CNNs).
- **Web Interface**: Simple and user-friendly upload page built with Flask.
- **Image Classification**: Classifies MRI scans into one of four categories.
- **Confidence Score**: Provides a confidence percentage for the prediction.

---

## Directory Structure


project/
|-- UIFlask.py                 # Flask application
|-- brain_tumor_classifier.h5  # Trained deep learning model
|-- uploads/               # Folder to store uploaded MRI scans
|-- README.md              # Project documentation

## Setup Instructions

### Prerequisites

Ensure you have the following installed on your system:

- Python 3.8 or higher
- TensorFlow
- Flask
- Other dependencies listed in `requirements.txt`

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/brain-tumor-classification.git
cd brain-tumor-classification

### Step 2: Install Dependencies
```bash
pip install Flask TensorFlow numpy pillow werkzeug

### Step 3: Download Model
Ensure the trained model file (brain_tumor_classifier.h5) is in the project root directory. You can replace this with your trained model if needed.

### Step 4: Run the Application
```bash
python UIFlask.py

### Step 5: Visit the Application and Run!
Visit the app in your browser at: http://127.0.0.1:5000/





