import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from PIL import Image
import numpy as np

# --- 1. CONFIGURATION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 7
MODEL_PATH = "skin_disease_resnet18_best.pth"
NEW_IMAGE_PATH = "ISIC_0000012.jpg"  # <--- CHANGE THIS PATH!

# The names of your diagnosis classes (MUST match your LabelEncoder's order)
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# Human-readable disease descriptions to show after prediction
DISEASE_DESCRIPTIONS = {
    "akiec": "Actinic keratoses and intraepithelial carcinoma (Bowen's disease)",
    "bcc": "Basal cell carcinoma, a type of skin cancer.",
    "bkl": "Benign keratosis-like lesions (solar lentigines, seborrheic keratoses).",
    "df": "Dermatofibroma.",
    "mel": "Melanoma, a serious form of skin cancer.",
    "nv": "Melanocytic nevi (a medical term for a mole).",
    "vasc": "Vascular lesions (angiomas, pyogenic granuloma).",
}


# 2. MODEL ARCHITECTURE DEFINITION

# This MUST EXACTLY match the structure trained
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)


# ------------------------------------------------------------------------
# 3. IMAGE PREPROCESSING (Inference Transform)
# ------------------------------------------------------------------------
# This MUST match the val_transform used during training
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


# ------------------------------------------------------------------------
# 4. LOAD MODEL WEIGHTS AND SET EVAL MODE
# ------------------------------------------------------------------------
print(f"Loading model from: {MODEL_PATH}...")
try:
    # Load weights onto the determined device (likely CPU)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
except FileNotFoundError:
    print(
        f"FATAL ERROR: Model file {MODEL_PATH} not found. Ensure it's in the correct location."
    )
    exit()

# Set model to evaluation mode (CRITICAL)
model.eval()
print("Model loaded successfully and set to evaluation mode.")



# 5. PREDICT ON NEW IMAGE

print(f"\nAttempting to classify image: {NEW_IMAGE_PATH}")

try:
    # Load and preprocess the image
    image = Image.open(NEW_IMAGE_PATH).convert("RGB")
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0).to(device)  # Add batch dimension

    # Run inference without tracking gradients
    with torch.no_grad():
        output = model(input_batch)

    # Interpret results: Convert raw outputs (logits) to probabilities
    probabilities = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]
    predicted_index = np.argmax(probabilities)
    confidence = probabilities[predicted_index] * 100
    predicted_class = CLASS_NAMES[predicted_index]
    description = DISEASE_DESCRIPTIONS.get(predicted_class, "Description not available.")

    print("\n--- CLASSIFICATION RESULTS ---")
    print(f"Predicted Diagnosis: {predicted_class}")
    print(f"Disease Name: {description}")
    print(f"Confidence: {confidence:.2f}%")
    print("\nFull Probabilities:")
    for name, prob in zip(CLASS_NAMES, probabilities):
        print(f"  {name}: {prob:.4f}")
    print("\n")
except FileNotFoundError:
    print(
        f"FATAL ERROR: New image file not found at {NEW_IMAGE_PATH}. Please check the path."
    )
except Exception as e:
    print(f"An unexpected error occurred during prediction: {e}")
