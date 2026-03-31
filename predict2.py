import sys
import cv2
import numpy as np
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt

MODEL_PATH = "model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# PART 3 FUNCTIONS (SAME AS TRAIN)
# ======================

from plantcv import plantcv as pcv

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
# LOAD MODEL
# ======================

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
class_names = checkpoint["classes"]
num_classes = len(class_names)

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)

model.load_state_dict(checkpoint["model_state"])
model = model.to(DEVICE)
model.eval()

# ======================
# INPUT CHECK
# ======================

if len(sys.argv) < 2:
    print("Usage: python predict.py <image_path>")
    sys.exit(1)

img_path = sys.argv[1]

# ======================
# LOAD IMAGE
# ======================

original_image = Image.open(img_path).convert("RGB")

# Apply transform
processed_tensor = transform(original_image).unsqueeze(0).to(DEVICE)

# ======================
# PREDICTION
# ======================

with torch.no_grad():
    output = model(processed_tensor)
    _, pred = torch.max(output, 1)

pred_class = class_names[pred.item()]
print("Predicted class:", pred_class)

# ======================
# VISUALIZATION (IMPORTANT FOR EVAL)
# ======================

# Show original
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(original_image)
plt.title("Original")
plt.axis("off")

# Show processed (Part 3 result)
processed_img = apply_mask_pcv(original_image)
processed_img = enhance_contrast(processed_img)

plt.subplot(1, 2, 2)
plt.imshow(processed_img)
plt.title("Transformed (Part 3)")
plt.axis("off")

plt.suptitle(f"Prediction: {pred_class}")
plt.show()
