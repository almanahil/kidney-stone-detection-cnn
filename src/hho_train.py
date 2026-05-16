import os
import random
import json
import math
from dataclasses import dataclass
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
DATA_ROOT = "data/splits"  # expects train/ val/ test
OUT_DIR = Path("runs/hho")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)
print("Using:", DEVICE)


# -----------------------
# Model (same as baseline but paramized)
# -----------------------
class KidneyCNN(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.4):
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
# Data
# -----------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def make_loaders(data_root: str, img_size: int, batch_size: int):
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

    # Mac-safe settings (no multiprocessing headaches)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    return train_loader, val_loader, test_loader, len(train_ds.classes), train_ds.classes


# -----------------------
# Metrics
# -----------------------
def metrics_from_preds(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return float(acc), float(prec), float(rec), float(f1)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    losses = []
    y_all, p_all = [], []

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        preds = logits.argmax(1)
        y_all.extend(y.detach().cpu().numpy().tolist())
        p_all.extend(preds.detach().cpu().numpy().tolist())

    acc, prec, rec, f1 = metrics_from_preds(y_all, p_all)
    return float(np.mean(losses)), acc, prec, rec, f1


@torch.no_grad()
def eval_one_epoch(model, loader, criterion):
    model.eval()
    losses = []
    y_all, p_all = [], []

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss = criterion(logits, y)

        losses.append(loss.item())
        preds = logits.argmax(1)
        y_all.extend(y.detach().cpu().numpy().tolist())
        p_all.extend(preds.detach().cpu().numpy().tolist())

    acc, prec, rec, f1 = metrics_from_preds(y_all, p_all)
    return float(np.mean(losses)), acc, prec, rec, f1


# -----------------------
# Hyperparameter space (HHO optimizes this)
# -----------------------
@dataclass
class HP:
    lr: float
    weight_decay: float
    dropout: float
    batch_size: int
    img_size: int


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def decode_position(pos: np.ndarray) -> HP:
    """
    pos is in [0,1]^5
    Convert to practical hyperparameters.
    """
    # log-scale learning rate: 1e-5 .. 3e-3
    lr_log = -5 + pos[0] * (math.log10(3e-3) - (-5))
    lr = 10 ** lr_log

    # weight decay log-scale: 1e-6 .. 1e-3
    wd_log = -6 + pos[1] * (math.log10(1e-3) - (-6))
    weight_decay = 10 ** wd_log

    # dropout: 0.2 .. 0.6
    dropout = 0.2 + pos[2] * (0.6 - 0.2)

    # batch size choices (Mac CPU friendly)
    bs_choices = [8, 16, 32]
    bs_idx = int(round(pos[3] * (len(bs_choices) - 1)))
    batch_size = bs_choices[clamp(bs_idx, 0, len(bs_choices) - 1)]

    # img size choices
    img_choices = [96, 128, 160]
    img_idx = int(round(pos[4] * (len(img_choices) - 1)))
    img_size = img_choices[clamp(img_idx, 0, len(img_choices) - 1)]

    return HP(lr=lr, weight_decay=weight_decay, dropout=dropout, batch_size=batch_size, img_size=img_size)


# -----------------------
# Objective: maximize VAL F1 (train only a few epochs per candidate)
# -----------------------
def evaluate_candidate(hp: HP, eval_epochs: int = 3) -> float:
    train_loader, val_loader, _, num_classes, _ = make_loaders(DATA_ROOT, hp.img_size, hp.batch_size)

    model = KidneyCNN(num_classes=num_classes, dropout=hp.dropout).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)

    best_val_f1 = -1.0
    for _ in range(eval_epochs):
        train_one_epoch(model, train_loader, criterion, optimizer)
        _, _, _, _, val_f1 = eval_one_epoch(model, val_loader, criterion)
        best_val_f1 = max(best_val_f1, val_f1)

    return best_val_f1


# -----------------------
# HHO (Harris Hawks Optimization) - simplified, stable version
# -----------------------
def hho_optimize(num_hawks=8, num_iters=8, eval_epochs=3):
    dim = 5
    X = np.random.rand(num_hawks, dim)  # positions in [0,1]

    # NEW: cache fitness so we don't re-train old hawks
    fitness = np.full((num_hawks,), -1.0, dtype=float)

    rabbit_pos = None
    rabbit_fit = -1.0
    convergence = []

    # initial evaluation (compute each hawk once)
    for i in range(num_hawks):
        hp = decode_position(X[i])
        fit = evaluate_candidate(hp, eval_epochs=eval_epochs)
        fitness[i] = fit
        if fit > rabbit_fit:
            rabbit_fit = fit
            rabbit_pos = X[i].copy()

    for t in range(num_iters):
        E1 = 2 * (1 - (t / max(1, num_iters - 1)))  # decreases from 2 to 0

        for i in range(num_hawks):
            E0 = 2 * random.random() - 1
            E = E1 * E0

            q = random.random()
            r = random.random()
            J = 2 * (1 - random.random())  # jump strength

            Xi = X[i].copy()
            Xrabbit = rabbit_pos.copy()

            if abs(E) >= 1:
                # Exploration
                rand_hawk = X[random.randint(0, num_hawks - 1)]
                Xnew = rand_hawk - r * abs(rand_hawk - 2 * r * Xi)
            else:
                # Exploitation
                if q >= 0.5 and abs(E) < 0.5:
                    # Hard besiege
                    Xnew = Xrabbit - E * abs(Xrabbit - Xi)
                elif q >= 0.5 and abs(E) >= 0.5:
                    # Soft besiege
                    Xnew = (Xrabbit - Xi) - E * abs(J * Xrabbit - Xi)
                elif q < 0.5 and abs(E) >= 0.5:
                    # Soft besiege with rapid dives (simplified)
                    Y = Xrabbit - E * abs(J * Xrabbit - Xi)
                    Z = Y + np.random.normal(0, 0.05, size=dim)
                    Xnew = Z
                else:
                    # Hard besiege with rapid dives (simplified)
                    Y = Xrabbit - E * abs(Xrabbit - Xi)
                    Z = Y + np.random.normal(0, 0.03, size=dim)
                    Xnew = Z

            Xnew = np.clip(Xnew, 0.0, 1.0)

            # Evaluate new hawk once
            hp_new = decode_position(Xnew)
            fit_new = evaluate_candidate(hp_new, eval_epochs=eval_epochs)

            # Greedy update WITHOUT recomputing fit_old
            if fit_new >= fitness[i]:
                X[i] = Xnew
                fitness[i] = fit_new

            if fit_new > rabbit_fit:
                rabbit_fit = fit_new
                rabbit_pos = Xnew.copy()

        convergence.append(rabbit_fit)
        print(f"[HHO] iter {t+1}/{num_iters} best_val_f1={rabbit_fit:.4f}")

    best_hp = decode_position(rabbit_pos)
    return best_hp, rabbit_fit, convergence



# -----------------------
# Final training using best HP
# -----------------------
def train_final(best_hp: HP, final_epochs: int = 12):
    train_loader, val_loader, test_loader, num_classes, class_names = make_loaders(
        DATA_ROOT, best_hp.img_size, best_hp.batch_size
    )

    model = KidneyCNN(num_classes=num_classes, dropout=best_hp.dropout).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=best_hp.lr, weight_decay=best_hp.weight_decay)

    history = {"train_f1": [], "val_f1": [], "train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
    best_val_f1 = -1.0
    best_path = OUT_DIR / "best_model.pt"

    for epoch in range(1, final_epochs + 1):
        tr_loss, tr_acc, _, _, tr_f1 = train_one_epoch(model, train_loader, criterion, optimizer)
        va_loss, va_acc, _, _, va_f1 = eval_one_epoch(model, val_loader, criterion)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)
        history["train_f1"].append(tr_f1)
        history["val_f1"].append(va_f1)

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            torch.save(model.state_dict(), best_path)

        print(f"[FINAL] epoch {epoch}/{final_epochs} train_f1={tr_f1:.3f} val_f1={va_f1:.3f} val_acc={va_acc:.3f}")

    # test with best model
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    te_loss, te_acc, te_prec, te_rec, te_f1 = eval_one_epoch(model, test_loader, criterion)

    return class_names, best_val_f1, {"loss": te_loss, "acc": te_acc, "precision": te_prec, "recall": te_rec, "f1": te_f1}, history


def save_plot(y, title, ylabel, out_path: Path):
    plt.figure()
    plt.plot(y)
    plt.xlabel("Iteration / Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def main():
    # You can tweak these if runtime is long on CPU:
    HAWKS = 4
    ITERS = 4
    EVAL_EPOCHS = 2
    FINAL_EPOCHS = 12

    print("\nRunning HHO search...")
    best_hp, best_fit, convergence = hho_optimize(num_hawks=HAWKS, num_iters=ITERS, eval_epochs=EVAL_EPOCHS)

    print("\nBest hyperparameters found by HHO:")
    print(best_hp)
    print("Best (HHO) val F1 during search:", best_fit)

    # Save HHO convergence curve
    save_plot(convergence, "HHO Convergence Curve (Best Val F1)", "Best Val F1", OUT_DIR / "hho_convergence.png")

    # Train final model using best HP
    print("\nTraining final model with best hyperparameters...")
    classes, best_val_f1, test_metrics, history = train_final(best_hp, final_epochs=FINAL_EPOCHS)

    # Save final curves
    save_plot(history["train_loss"], "Train Loss", "Loss", OUT_DIR / "train_loss.png")
    save_plot(history["val_loss"], "Val Loss", "Loss", OUT_DIR / "val_loss.png")
    save_plot(history["train_acc"], "Train Accuracy", "Accuracy", OUT_DIR / "train_acc.png")
    save_plot(history["val_acc"], "Val Accuracy", "Accuracy", OUT_DIR / "val_acc.png")
    save_plot(history["train_f1"], "Train F1", "F1", OUT_DIR / "train_f1.png")
    save_plot(history["val_f1"], "Val F1", "F1", OUT_DIR / "val_f1.png")

    results = {
        "best_hp": best_hp.__dict__,
        "hho_best_val_f1_search": best_fit,
        "final_best_val_f1": best_val_f1,
        "test_metrics": test_metrics,
        "classes": classes,
        "convergence": convergence,
        "notes": {
            "objective": "maximize validation F1",
            "eval_epochs_per_candidate": EVAL_EPOCHS,
            "final_epochs": FINAL_EPOCHS,
            "hawks": HAWKS,
            "iters": ITERS
        }
    }

    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n[HHO DONE] Test metrics:", test_metrics)
    print("Saved everything to:", OUT_DIR)


if __name__ == "__main__":
    main()
