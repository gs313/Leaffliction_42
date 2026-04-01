import torch
from PIL import Image
import cv2
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from torchvision.models import ResNet18_Weights
from plantcv import plantcv as pcv


# ======================
# CONFIG
# ======================
DATA_DIR = "balanced"   # change if needed
BATCH_SIZE = 32
EPOCHS = 10
LR = 0.001
MODEL_PATH = "model.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# PART 3 FUNCTIONS
# ======================


def apply_mask_pcv(img):
    # Convert PIL → numpy
    img_np = np.array(img)

    # Step 1: Convert to HSV
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

    # Step 2: Create mask (green detection)
    lower = np.array([25, 40, 40])
    upper = np.array([95, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # Step 3: Define ROI (center region)
    h, w = mask.shape
    roi = pcv.roi.rectangle(
        img=mask,
        x=int(w * 0.1),
        y=int(h * 0.1),
        h=int(h * 0.8),
        w=int(w * 0.8)
    )

    # Step 4: Apply ROI filter (YOUR LINE)
    filtered_mask = pcv.roi.filter(mask=mask, roi=roi, roi_type='partial')

    # Step 5: Apply mask to image
    result = cv2.bitwise_and(img_np, img_np, mask=filtered_mask)

    return Image.fromarray(result)


def enhance_contrast(img):
    img_np = np.array(img)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    merged = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

    return Image.fromarray(enhanced)


# ======================
# TRANSFORM (IDENTICAL TO TRAIN)
# ======================

transform = transforms.Compose([
    transforms.Lambda(lambda img: apply_mask_pcv(img)),
    transforms.Lambda(lambda img: enhance_contrast(img)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ======================
# LOAD DATASET
# ======================
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
class_names = dataset.classes
num_classes = len(class_names)

print("Classes:", class_names)

# ======================
# SPLIT DATA
# ======================
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ======================
# MODEL (TRANSFER LEARNING)
# ======================
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

# Freeze layers (optional but helps small dataset)
for param in model.parameters():
    param.requires_grad = False

# Replace classifier
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(DEVICE)

# ======================
# TRAIN SETUP
# ======================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LR)

# ======================
# TRAIN LOOP
# ======================
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")

    # ======================
    # VALIDATION
    # ======================
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total
    print(f"Validation Accuracy: {acc:.4f}")

# ======================
# SAVE MODEL
# ======================
torch.save({
    "model_state": model.state_dict(),
    "classes": class_names
}, MODEL_PATH)

print("Model saved to", MODEL_PATH)
