import os
import random
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
from tqdm import tqdm

# -----------------------
# Config
# -----------------------
DATA_ROOT = "data/splits"   # expects train/ val/ test folders inside
RUN_DIR = Path("runs/baseline")
RUN_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
EPOCHS = 12
IMG_SIZE = 128
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.4

# -----------------------
# Reproducibility
# -----------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# -----------------------
# Data
# -----------------------
def get_dataloaders(data_root, batch_size=32, img_size=128, num_workers=0):
    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    train_ds = datasets.ImageFolder(os.path.join(data_root, "train"), transform=train_tfms)
    val_ds   = datasets.ImageFolder(os.path.join(data_root, "val"),   transform=eval_tfms)
    test_ds  = datasets.ImageFolder(os.path.join(data_root, "test"),  transform=eval_tfms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=False)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=False)

    print("Classes:", train_ds.classes)
    return train_loader, val_loader, test_loader, len(train_ds.classes), train_ds.classes

train_loader, val_loader, test_loader, num_classes, class_names = get_dataloaders(
    DATA_ROOT, batch_size=BATCH_SIZE, img_size=IMG_SIZE
)

# -----------------------
# Model
# -----------------------
class KidneyCNN(nn.Module):
    def __init__(self, num_classes, dropout=0.4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)

# -----------------------
# Train/Eval helpers
# -----------------------
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    losses = []
    all_y, all_p = [], []

    for x, y in tqdm(loader, desc="Train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        preds = logits.argmax(1)
        all_y.extend(y.detach().cpu().numpy().tolist())
        all_p.extend(preds.detach().cpu().numpy().tolist())

    acc = accuracy_score(all_y, all_p)
    prec, rec, f1, _ = precision_recall_fscore_support(all_y, all_p, average="weighted", zero_division=0)
    return float(np.mean(losses)), acc, prec, rec, f1


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, tag="Val"):
    model.eval()
    losses = []
    all_y, all_p = [], []

    for x, y in tqdm(loader, desc=tag, leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        losses.append(loss.item())
        preds = logits.argmax(1)
        all_y.extend(y.detach().cpu().numpy().tolist())
        all_p.extend(preds.detach().cpu().numpy().tolist())

    acc = accuracy_score(all_y, all_p)
    prec, rec, f1, _ = precision_recall_fscore_support(all_y, all_p, average="weighted", zero_division=0)
    return float(np.mean(losses)), acc, prec, rec, f1

# -----------------------
# Training
# -----------------------
model = KidneyCNN(num_classes=num_classes, dropout=DROPOUT).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

history = {
    "train_loss": [], "val_loss": [],
    "train_acc": [], "val_acc": [],
    "train_f1": [], "val_f1": [],
}

best_val_f1 = -1.0
best_path = RUN_DIR / "best_model.pt"

print("Training baseline...")

for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc, tr_prec, tr_rec, tr_f1 = train_one_epoch(model, train_loader, criterion, optimizer)
    va_loss, va_acc, va_prec, va_rec, va_f1 = eval_one_epoch(model, val_loader, criterion, tag="Val")

    history["train_loss"].append(tr_loss)
    history["val_loss"].append(va_loss)
    history["train_acc"].append(tr_acc)
    history["val_acc"].append(va_acc)
    history["train_f1"].append(tr_f1)
    history["val_f1"].append(va_f1)

    if va_f1 > best_val_f1:
        best_val_f1 = va_f1
        torch.save(model.state_dict(), best_path)

    print(
        f"Epoch {epoch}/{EPOCHS} | "
        f"Train Acc={tr_acc:.3f} F1={tr_f1:.3f} | "
        f"Val Acc={va_acc:.3f} F1={va_f1:.3f}"
    )

print("\nBaseline training complete!")
print("Best val F1:", best_val_f1)

# Load best & test
model.load_state_dict(torch.load(best_path, map_location=device))
te_loss, te_acc, te_prec, te_rec, te_f1 = eval_one_epoch(model, test_loader, criterion, tag="Test")

results = {
    "classes": class_names,
    "hyperparams": {
        "epochs": EPOCHS, "img_size": IMG_SIZE, "batch_size": BATCH_SIZE,
        "lr": LR, "weight_decay": WEIGHT_DECAY, "dropout": DROPOUT
    },
    "best_val_f1": best_val_f1,
    "test_metrics": {
        "acc": te_acc, "precision": te_prec, "recall": te_rec, "f1": te_f1, "loss": te_loss
    },
    "history": history,
}

with open(RUN_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2)

# Curves for report (required)
plt.figure()
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Curve")
plt.savefig(RUN_DIR / "loss_curve.png", bbox_inches="tight")
plt.close()

plt.figure()
plt.plot(history["train_acc"], label="Train Acc")
plt.plot(history["val_acc"], label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy Curve")
plt.savefig(RUN_DIR / "acc_curve.png", bbox_inches="tight")
plt.close()

plt.figure()
plt.plot(history["train_f1"], label="Train F1")
plt.plot(history["val_f1"], label="Val F1")
plt.xlabel("Epoch")
plt.ylabel("F1")
plt.legend()
plt.title("F1 Curve")
plt.savefig(RUN_DIR / "f1_curve.png", bbox_inches="tight")
plt.close()

print("\nTEST metrics:", results["test_metrics"])
print(f"Saved everything to: {RUN_DIR}")
