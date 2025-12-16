import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- CONFIGURATION ---
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes so React can connect

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 8
MODEL_PATH = "skin_disease_resnet18_best.pth"

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc", "unknown_class"]
DISEASE_DESCRIPTIONS = {
    "akiec": "Actinic keratoses - Pre-cancerous scaly spots.",
    "bcc": "Basal cell carcinoma - A type of skin cancer.",
    "bkl": "Benign keratosis - Non-cancerous skin growth.",
    "df": "Dermatofibroma - Benign skin nodule.",
    "mel": "Melanoma - Serious form of skin cancer.",
    "nv": "Melanocytic nevi - Common moles.",
    "vasc": "Vascular lesions - Blood vessel abnormalities.",
    "unknown_class": "Unidentified skin condition.",
}

# --- LOAD MODEL ---
def load_model():
    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model = model.to(device)
        model.eval()
        return model
    except FileNotFoundError:
        print(f"ERROR: Model file '{MODEL_PATH}' not found. Please ensure it is in the same directory.")
        return None
    except Exception as e:
        print(f"ERROR: {e}")
        return None

model = load_model()

# --- PREPROCESSING ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # 1. Read image bytes
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # 2. Transform image
        input_tensor = transform(image).unsqueeze(0).to(device) # Add batch dimension
        
        # 3. Prediction
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            class_idx = predicted.item()
            class_name = CLASS_NAMES[class_idx]
            description = DISEASE_DESCRIPTIONS.get(class_name, "No description available.")
            confidence_score = confidence.item() * 100

        return jsonify({
            'class': class_name,
            'description': description,
            'confidence': f"{confidence_score:.2f}%"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask API Server...")
    app.run(debug=True, port=5000)