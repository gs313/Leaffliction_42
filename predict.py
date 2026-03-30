import sys
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt

MODEL_PATH = "model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
# TRANSFORM
# ======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ======================
# LOAD IMAGE
# ======================
if len(sys.argv) < 2:
    print("Usage: python predict.py <image_path>")
    sys.exit(1)

img_path = sys.argv[1]
image = Image.open(img_path).convert("RGB")

input_tensor = transform(image).unsqueeze(0).to(DEVICE)

# ======================
# PREDICT
# ======================
with torch.no_grad():
    output = model(input_tensor)
    _, pred = torch.max(output, 1)

pred_class = class_names[pred.item()]

print("Predicted class:", pred_class)

# ======================
# DISPLAY
# ======================
plt.imshow(image)
plt.title(f"Prediction: {pred_class}")
plt.axis("off")
plt.show()
