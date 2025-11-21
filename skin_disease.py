# import torch
# import torch.nn as nn
# from torchvision import models
# from torch.utils.data import Dataset
# from torchvision import transforms
# from torch.utils.data import DataLoader
# from PIL import Image
# import os
# import pandas as pd

# from sklearn.preprocessing import LabelEncoder
# import torch.optim as optim
# from sklearn.model_selection import train_test_split

# # Load the metadata CSV
# metadata_path = r"C:\Users\Ahana\Desktop\ML\images\HAM10000_metadata.csv"
# df = pd.read_csv(metadata_path)

# # Map image filenames to diagnosis labels
# # Ensure 'image' and 'dx' columns exist
# if not {"image", "dx"}.issubset(df.columns):
#     raise ValueError("CSV file must contain 'image' and 'dx' columns.")

# # Create a new column with the image filename
# df["filename"] = df["image"].astype(str) + ".jpg"

# # Encode labels as integers
# le = LabelEncoder()
# df["label"] = le.fit_transform(df["dx"])
# filename_to_label = dict(zip(df["filename"], df["label"]))  # Use integer labels
# num_classes = len(le.classes_)

# # Example usage: get the diagnosis for a specific image filename
# # diagnosis = filename_to_label['ISIC_0024306.jpg']

# # Print a sample mapping
# for k, v in list(filename_to_label.items())[:5]:
#     print(f"{k}: {v}")


# class HAM10000Dataset(Dataset):
#     def __init__(self, image_dir, label_dict, transform=None):
#         self.image_dir = image_dir
#         self.label_dict = label_dict
#         self.transform = transform
#         self.image_filenames = list(label_dict.keys())

#     def __len__(self):
#         return len(self.image_filenames)

#     def __getitem__(self, idx):
#         filename = self.image_filenames[idx]
#         img_path = os.path.join(self.image_dir, filename)
#         image = Image.open(img_path).convert("RGB")
#         label = torch.tensor(self.label_dict[filename], dtype=torch.long)

#         if self.transform:
#             image = self.transform(image)

#         return image, label


# transform = transforms.Compose(
#     [
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#     ]
# )

# image_dir = r"C:\Users\Ahana\Desktop\ML\images\images"
# dataset = HAM10000Dataset(
#     image_dir=image_dir, label_dict=filename_to_label, transform=transform
# )
# loader = DataLoader(dataset, batch_size=32, shuffle=True)
# images, labels = next(iter(loader))
# print(images.shape)  # Should be [32, 3, 224, 224]
# print(labels[:5])  # Sample labels
# # missing = [f for f in df["filename"] if not os.path.exists(os.path.join(image_dir, f))]
# # print(f"Missing files: {len(missing)}")
# # print(missing[:5])


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = models.resnet18(pretrained=True)
# model.fc = nn.Linear(model.fc.in_features, num_classes)
# model = model.to(device)


# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# num_epochs = 10

# # Split the DataFrame into train and validation sets (stratified by label)
# train_df, val_df = train_test_split(
#     df, test_size=0.2, stratify=df["label"], random_state=42
# )

# # Create label dictionaries for each split
# train_label_dict = dict(zip(train_df["filename"], train_df["label"]))
# val_label_dict = dict(zip(val_df["filename"], val_df["label"]))

# # Create datasets and loaders for each split
# train_dataset = HAM10000Dataset(image_dir, train_label_dict, transform)
# val_dataset = HAM10000Dataset(image_dir, val_label_dict, transform)
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     correct = 0

#     for inputs, labels in train_loader:
#         inputs = inputs.to(device)
#         labels = labels.to(device)

#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
#         _, preds = torch.max(outputs, 1)
#         correct += torch.sum(preds == labels.data)

#     epoch_loss = running_loss / len(train_loader)
#     epoch_acc = correct.double() / len(train_dataset)
#     print(
#         f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}"
#     )

#     # Validation after each epoch
#     model.eval()
#     val_correct = 0
#     val_loss = 0.0
#     with torch.no_grad():
#         for val_inputs, val_labels in val_loader:
#             val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
#             val_outputs = model(val_inputs)
#             v_loss = criterion(val_outputs, val_labels)
#             val_loss += v_loss.item()
#             _, val_preds = torch.max(val_outputs, 1)
#             val_correct += torch.sum(val_preds == val_labels.data)
#     val_epoch_loss = val_loss / len(val_loader)
#     val_epoch_acc = val_correct.double() / len(val_dataset)
#     print(f"  [Validation] Loss: {val_epoch_loss:.4f}, Accuracy: {val_epoch_acc:.4f}")

# # Save the trained model
# torch.save(model.state_dict(), "skin_disease_resnet18.pth")
# print("Model saved as skin_disease_resnet18.pth")


import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch.optim as optim
from torch.optim import lr_scheduler  # NEW: Import scheduler
from sklearn.model_selection import train_test_split

# Load the metadata CSV
metadata_path = r"c:\Users\rocky\Desktop\CICPS_2026\images\ISIC_2019_Training_GroundTruth.csv"
df = pd.read_csv(metadata_path)

# Map image filenames to diagnosis labels
# Ensure 'image' and 'dx' columns exist
if not {"image", "dx"}.issubset(df.columns):
    raise ValueError("CSV file must contain 'image' and 'dx' columns.")

# Create a new column with the image filename
df["filename"] = df["image"].astype(str) + ".jpg"

# Encode labels as integers
le = LabelEncoder()
df["label"] = le.fit_transform(df["dx"])
filename_to_label = dict(zip(df["filename"], df["label"]))  # Use integer labels
num_classes = len(le.classes_)

# Example usage: get the diagnosis for a specific image filename
# diagnosis = filename_to_label['ISIC_0024306.jpg']

# Print a sample mapping
for k, v in list(filename_to_label.items())[:5]:
    print(f"{k}: {v}")


class HAM10000Dataset(Dataset):
    def __init__(self, image_dir, label_dict, transform=None):
        self.image_dir = image_dir
        self.label_dict = label_dict
        self.transform = transform
        self.image_filenames = list(label_dict.keys())

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        filename = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, filename)
        image = Image.open(img_path).convert("RGB")
        label = torch.tensor(self.label_dict[filename], dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label


# ----------------------------------------------------
# 1. NEW: DEFINE SEPARATE TRAIN AND VALIDATION TRANSFORMS
# ----------------------------------------------------

# Stronger Augmentation for TRAINING to prevent overfitting
train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomRotation(30),  # NEW: Rotation
        transforms.RandomHorizontalFlip(),  # NEW: Horizontal Flip
        transforms.RandomVerticalFlip(),  # NEW: Vertical Flip
        transforms.ColorJitter(brightness=0.1, contrast=0.1),  # NEW: Color Jitter
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# Basic Transform for VALIDATION (should not use augmentation)
val_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

image_dir = r"c:\Users\rocky\Desktop\CICPS_2026\images\images"
# NOTE: The initial 'dataset' and 'loader' setup is now technically redundant
# since you create train/val splits later, but kept for context:
# dataset = HAM10000Dataset(image_dir=image_dir, label_dict=filename_to_label, transform=train_transform)
# loader = DataLoader(dataset, batch_size=32, shuffle=True)
# ... (Print images.shape, labels[:5] if needed) ...


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------
# 2. MODEL DEFINITION & WEIGHT LOADING
# ----------------------------------------------------

# Set pretrained=False because you will load your fine-tuned weights
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Load the previously saved weights to continue/resume training
try:
    model.load_state_dict(
        torch.load("skin_disease_resnet18_best.pth", map_location=device)
    )
    print("RESUMED: Loaded previous model weights to continue training.")
except FileNotFoundError:
    print(
        "WARNING: Previous model not found. Starting with ImageNet weights (if available)."
    )
    # If using pretrained=False above, this will start from random init if the file is missing!


criterion = nn.CrossEntropyLoss()
# 3. NEW: Add L2 Regularization (weight_decay) to Adam
# Also, typically use a smaller LR when fine-tuning/resuming
optimizer = optim.Adam(
    model.parameters(), lr=0.0001, weight_decay=1e-4
)  # CHANGED LR and ADDED weight_decay

# 4. NEW: Learning Rate Scheduler (helps stabilize training)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

num_epochs = 50  # Set a higher number, relying on early stopping

# Split the DataFrame into train and validation sets (stratified by label)
train_df, val_df = train_test_split(
    df, test_size=0.2, stratify=df["label"], random_state=42
)

# Create label dictionaries for each split
train_label_dict = dict(zip(train_df["filename"], train_df["label"]))
val_label_dict = dict(zip(val_df["filename"], val_df["label"]))

# 5. UPDATED: Use the new transforms for respective datasets
train_dataset = HAM10000Dataset(
    image_dir, train_label_dict, train_transform
)  # Use train_transform
val_dataset = HAM10000Dataset(
    image_dir, val_label_dict, val_transform
)  # Use val_transform
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ----------------------------------------------------
# 6. NEW: EARLY STOPPING & CHECKPOINT VARIABLES
# ----------------------------------------------------
BEST_MODEL_PATH = "skin_disease_resnet18_best.pth"  # NEW FILE NAME!
best_val_accuracy = 0.0
patience = 5  # Stop if no improvement after 5 epochs
patience_counter = 0


for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct.double() / len(train_dataset)
    print(
        f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}"
    )

    # Validation after each epoch
    model.eval()
    val_correct = 0
    val_loss = 0.0
    with torch.no_grad():
        for val_inputs, val_labels in val_loader:
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
            val_outputs = model(val_inputs)
            v_loss = criterion(val_outputs, val_labels)
            val_loss += v_loss.item()
            _, val_preds = torch.max(val_outputs, 1)
            val_correct += torch.sum(val_preds == val_labels.data)
    val_epoch_loss = val_loss / len(val_loader)
    val_epoch_acc = val_correct.double() / len(val_dataset)
    print(f"  [Validation] Loss: {val_epoch_loss:.4f}, Accuracy: {val_epoch_acc:.4f}")

    # Early stopping logic
    if val_epoch_acc > best_val_accuracy:
        best_val_accuracy = val_epoch_acc
        patience_counter = 0
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(
            f"  >>> CHECKPOINT: New best model saved to {BEST_MODEL_PATH} with Acc: {best_val_accuracy:.4f} <<<"
        )
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(
                f"EARLY STOPPING: Validation accuracy has not improved for {patience} epochs."
            )
            break

    scheduler.step()  # <-- Move this here, after each epoch

# The final save command is now redundant and can be removed/commented out
# torch.save(model.state_dict(), "skin_disease_resnet18.pth")
# print("Model saved as skin_disease_resnet18.pth")
