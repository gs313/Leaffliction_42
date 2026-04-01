import os
import torch
from PIL import Image
import cv2
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import argparse

# ======================
# CONFIG
# ======================
parser = argparse.ArgumentParser(description="Train leaf disease classifier")

parser.add_argument("data_dir", type=str, help="Path to dataset directory")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--output", type=str, default="results")

args = parser.parse_args()

DATA_DIR = args.data_dir
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
LR = args.lr
OUTPUT_DIR = args.output
MODEL_PATH = os.path.join(OUTPUT_DIR, "model.pth")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# PART 3 FUNCTIONS (PlantCV)
# ======================
from plantcv import plantcv as pcv

def apply_mask_pcv(img):
    img_np = np.array(img)

    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

    lower = np.array([25, 40, 40])
    upper = np.array([95, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    h, w = mask.shape
    roi = pcv.roi.rectangle(
        img=mask,
        x=int(w * 0.1),
        y=int(h * 0.1),
        h=int(h * 0.8),
        w=int(w * 0.8)
    )

    filtered_mask = pcv.roi.filter(mask=mask, roi=roi, roi_type='partial')

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
# TRANSFORM
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
# MODEL
# ======================
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(DEVICE)

# ======================
# TRAIN SETUP
# ======================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LR)

# ======================
# TRACKING
# ======================
train_losses = []
val_losses = []
val_accuracies = []

best_val_loss = float("inf")
patience = 3
patience_counter = 0

# ======================
# TRAIN LOOP
# ======================
for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # ======================
    # VALIDATION
    # ======================
    model.eval()
    total_val_loss = 0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    acc = correct / total
    val_accuracies.append(acc)

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"Val Loss: {avg_val_loss:.4f}")
    print(f"Val Accuracy: {acc:.4f}")
    print("-" * 30)

    # ======================
    # EARLY STOPPING
    # ======================
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0

        torch.save({
            "model_state": model.state_dict(),
            "classes": class_names
        }, MODEL_PATH)

    else:
        patience_counter += 1

    if patience_counter >= patience:
        print("Early stopping triggered!")
        break

# ======================
# FINAL METRICS
# ======================
report = classification_report(all_labels, all_preds, target_names=class_names)

print("\nClassification Report:")
print(report)

with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
    f.write(report)
# ======================
# CONFUSION MATRIX
# ======================
cm = confusion_matrix(all_labels, all_preds)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(xticks_rotation=45)
plt.title("Confusion Matrix")
# plt.show()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.close()
# ======================
# LOSS GRAPH
# ======================
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Curve")
plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"))
# plt.show()
plt.close()

# ======================
# ACCURACY GRAPH
# ======================
plt.figure()
plt.plot(val_accuracies, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy Curve")
plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_curve.png"))
# plt.show()
plt.close()

print("Model saved to", MODEL_PATH)
