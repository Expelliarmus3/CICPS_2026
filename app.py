import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import os
import gc
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
# Allow CORS for your frontend
CORS(app, resources={r"/*": {"origins": "*"}})

# --- CONFIGURATION ---
# Force CPU usage to stay within Render's free tier limits
device = torch.device("cpu")
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

# Global variable to hold the model
model = None

def get_model():
    """
    Loads the model only when needed (Lazy Loading).
    This prevents the server from crashing due to timeouts/memory spikes at startup.
    """
    global model
    if model is not None:
        return model

    print(f"Loading model from {MODEL_PATH}...")
    try:
        # 1. Define Architecture
        loaded_model = models.resnet18(weights=None)
        loaded_model.fc = nn.Linear(loaded_model.fc.in_features, num_classes)

        # 2. Load Weights (Force CPU mapping)
        if os.path.exists(MODEL_PATH):
            state_dict = torch.load(MODEL_PATH, map_location=device)
            loaded_model.load_state_dict(state_dict)
            loaded_model = loaded_model.to(device)
            loaded_model.eval()
            model = loaded_model
            print("Model loaded successfully!")
            return model
        else:
            print(f"ERROR: {MODEL_PATH} not found on server.")
            return None
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return None

# --- PREPROCESSING ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# --- ROUTES ---

@app.route('/', methods=['GET'])
def health_check():
    """Simple route to stop 404 errors in logs and verify app is running."""
    return jsonify({"status": "online", "message": "Skin Disease API is running"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    # Load model just in time
    current_model = get_model()
    
    if current_model is None:
        return jsonify({'error': 'Model unavailable. Check server logs.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Read and transform image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            output = current_model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            class_idx = predicted.item()
            class_name = CLASS_NAMES[class_idx]
            description = DISEASE_DESCRIPTIONS.get(class_name, "No description available.")
            confidence_score = confidence.item() * 100

        # Optional: Clear memory after heavy prediction if needed
        # gc.collect() 

        return jsonify({
            'class': class_name,
            'description': description,
            'confidence': f"{confidence_score:.2f}%"
        })

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Use PORT environment variable if available (for Render)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
