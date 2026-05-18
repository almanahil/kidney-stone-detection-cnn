import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from pathlib import Path
import json

from src.hho_train import KidneyCNN

# ---------------- CONFIG ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_TYPE = "hho"  # "baseline" or "hho"

IMAGE_PATH = "/Users/fatmaalhinai/Downloads/kidney_cnn_hho/data/splits/test/image from google/Screenshot 2025-12-31 at 12.06.39 PM.png"  # <-- CHANGE THIS

# ----------------------------------------

if MODEL_TYPE == "baseline":
    WEIGHTS_PATH = Path("runs/baseline/best_model.pt")
    IMG_SIZE = 128
    CLASS_NAMES = ["Non-Stone", "Stone"]

elif MODEL_TYPE == "hho":
    WEIGHTS_PATH = Path("runs/hho/best_model.pt")

    # load exact HHO hyperparameters
    with open("runs/hho/results.json", "r") as f:
        results = json.load(f)

    IMG_SIZE = int(results["best_hp"]["img_size"])
    CLASS_NAMES = results["classes"]

else:
    raise ValueError("Invalid MODEL_TYPE")

print("Using:", DEVICE)
print("Loading model from:", WEIGHTS_PATH)

# -------- Image preprocessing (same as training) --------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3), 
])


# -------- Load image --------
img = Image.open(IMAGE_PATH).convert("RGB")
x = transform(img).unsqueeze(0).to(DEVICE)

# -------- Load model --------
model = KidneyCNN(num_classes=len(CLASS_NAMES)).to(DEVICE)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
model.eval()

# -------- Predict --------
with torch.no_grad():
    logits = model(x)
    probs = F.softmax(logits, dim=1).cpu().numpy()[0]

pred_idx = probs.argmax()
pred_class = CLASS_NAMES[pred_idx]
confidence = probs[pred_idx]

print("\nImage Classification Demo")
print("----------------------------------")
print("Image:", IMAGE_PATH)
print("Predicted class:", pred_class)
print("Confidence:", f"{confidence*100:.2f}%")
