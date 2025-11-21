import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from PIL import Image
import numpy as np
import random
import sys


from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


from torchvision.datasets import ImageFolder


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 8  # Correctly set to 8
MODEL_PATH = "skin_disease_resnet18_best.pth"

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc", "unknown_class"]
DISEASE_DESCRIPTIONS = {
    "akiec": "Actinic keratoses...",
    "bcc": "Basal cell carcinoma...",
    "bkl": "Benign keratosis-like lesions...",
    "df": "Dermatofibroma.",
    "mel": "Melanoma...",
    "nv": "Melanocytic nevi...",
    "vasc": "Vascular lesions...",
    "unknown_class": "Description for the 8th class",
}


# --- MODEL ARCHITECTURE DEFINITION ---
model = models.resnet18(weights=None) 
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)


# --- IMAGE PREPROCESSING (Must match training) ---
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


TEST_DATA_PATH = r"C:\Users\rocky\Desktop\CICPS_2026\test_folder" 

try:
    test_dataset = ImageFolder(root=TEST_DATA_PATH, transform=transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=32,  # You can make this smaller (e.g., 16) if you run out of memory
        shuffle=False   # DO NOT shuffle for evaluation/testing
    )
    print(f"Successfully loaded test dataset from: {TEST_DATA_PATH}")
    
    
    if test_dataset.classes != CLASS_NAMES:
        print("\n!!!!!!!!!! CLASS NAME MISMATCH WARNING !!!!!!!!!!")
        print(f"Your CLASS_NAMES list: {CLASS_NAMES}")
        print(f"Folders found in path: {test_dataset.classes}")
        print("This can cause your metrics and predictions to be wrong.")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
    else:
        print(f"Classes found match script: {test_dataset.classes}")

except FileNotFoundError:
    print(f"\nFATAL ERROR: Test data folder not found at '{TEST_DATA_PATH}'")
    print("Please update the 'TEST_DATA_PATH' variable to proceed.")
    sys.exit() # Exit the script
except Exception as e:
    print(f"\nFATAL ERROR loading test data: {e}")
    sys.exit()



print(f"\nLoading model from: {MODEL_PATH}...")
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
except FileNotFoundError:
    print(f"FATAL ERROR: Model file {MODEL_PATH} not found.")
    sys.exit()
except Exception as e:
    print(f"FATAL ERROR loading model state_dict: {e}")
    sys.exit()

model.eval() # Set model to evaluation mode
print("Model loaded successfully and set to evaluation mode.")



def evaluate_model(model, data_loader, device):
    """
    Runs the model over a REAL test DataLoader to get
    true labels, predictions, and probabilities.
    """
    print(f"\n--- RUNNING REAL EVALUATION ON TEST DATASET ---")
    
    all_labels = []
    all_predictions = []
    all_probabilities = []
    
    # No gradients needed for evaluation (saves memory and computation)
    with torch.no_grad():
        # Loop through all batches in the test_loader
        for i, (inputs, labels) in enumerate(data_loader):
            # Move data to the correct device (CPU or GPU)
            inputs = inputs.to(device)
            
            # 1. Get model output (logits)
            output = model(inputs)
            
            # 2. Get predicted class (the index with the highest score)
            predicted_indices = torch.argmax(output, dim=1)
            
            # 3. Get probabilities (for AUC calculation)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            
            # 4. Store results
            # .cpu().numpy() moves data back from GPU to CPU and converts to NumPy
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted_indices.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            
            if (i + 1) % 10 == 0:
                print(f"  Processed batch {i+1} of {len(data_loader)}")

    # Convert lists to NumPy arrays for scikit-learn
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    
    print("...Evaluation complete.")
    return all_labels, all_predictions, all_probabilities


def calculate_and_print_metrics(y_true, y_pred, y_prob):
    """Calculates and prints all requested metrics."""
    print("\n--- MODEL EVALUATION METRICS ---")
    
    #  Accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")
    
    #  Precision, Recall, F1 Score (Weighted)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    print(f"Precision (Weighted): {precision:.4f}")
    print(f"Recall (Weighted): {recall:.4f}")
    print(f"F1 Score (Weighted): {f1:.4f}")
    
    #  AUC (Area Under the Curve)
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
        print(f"AUC (One-vs-Rest): {auc:.4f}")
    except ValueError as e:
        print(f"Could not calculate AUC: {e}")
    
    print("\n--- METRICS PER CLASS ---")
    precision_c, recall_c, f1_c, support_c = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0, labels=range(num_classes)
    )
    
    print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-" * 55)
    for i in range(num_classes):
        print(f"{CLASS_NAMES[i]:<15} {precision_c[i]:<10.4f} {recall_c[i]:<10.4f} {f1_c[i]:<10.4f} {support_c[i]:<10}")
        
    
    print("\n--- CONFUSION MATRIX ---")
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    print(cm)
    
    
    try:
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.savefig("real_confusion_matrix.png") # <-- Saved to a new file
        print("\nConfusion matrix plot saved to 'real_conversion_matrix.png'") # Note: I corrected a typo here, "conversion" -> "confusion"
    except Exception as e:
        print(f"Could not plot confusion matrix (matplotlib/seaborn error): {e}")



# ------------------------------------------------------------------------


true_labels, predicted_labels, predicted_probs = evaluate_model(
    model=model, 
    data_loader=test_loader, 
    device=device
)


calculate_and_print_metrics(true_labels, predicted_labels, predicted_probs)