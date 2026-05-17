"""
Bird Biodiversity Classification using CUB-200-2011 Dataset
Model: EfficientNetV2-B3 (Transfer Learning)
Pipeline: Preprocessing → Feature Extraction → Feature Selection → Training → Classification
"""

# ============================================================
# IMPORTS
# ============================================================
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as transforms
import torchvision.models as models
import timm  # For EfficientNetV2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                              top_k_accuracy_score, accuracy_score)
from sklearn.feature_selection import VarianceThreshold

# ============================================================
# CONFIGURATION
# ============================================================
class Config:
    # Dataset path — update this to your CUB_200_2011 root folder
    DATA_ROOT = "./CUB_200_2011"

    # Training
    BATCH_SIZE = 32
    NUM_EPOCHS = 30
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 4
    NUM_CLASSES = 200

    # Image
    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]   # ImageNet stats
    STD  = [0.229, 0.224, 0.225]

    # Feature Selection
    PCA_COMPONENTS = 512      # PCA dims for feature selection analysis
    VARIANCE_THRESHOLD = 0.01 # Remove near-zero variance features

    # Paths
    CHECKPOINT_DIR = "./checkpoints"
    RESULTS_DIR    = "./results"
    MODEL_SAVE_PATH = "./checkpoints/best_model.pth"

    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = Config()
os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

print(f"Using device: {cfg.DEVICE}")

# ============================================================
# STEP 1: DATA PREPROCESSING
# ============================================================

class CUBDataset(Dataset):
    """
    CUB-200-2011 Dataset Loader
    Expected folder structure:
        CUB_200_2011/
            images/
                001.Black_footed_Albatross/
                    Black_Footed_Albatross_0001_796111.jpg
                    ...
                002.Laysan_Albatross/
                ...
            images.txt       (image_id filename)
            train_test_split.txt  (image_id is_training_image)
            classes.txt      (class_id class_name)
            image_class_labels.txt (image_id class_id)
    """
    def __init__(self, root, split="train", transform=None):
        self.root = Path(root)
        self.transform = transform
        self.split = split

        # Load metadata files
        self.images   = self._load_txt("images.txt",      ["id", "filename"])
        self.splits   = self._load_txt("train_test_split.txt", ["id", "is_train"])
        self.labels   = self._load_txt("image_class_labels.txt", ["id", "label"])
        self.classes  = self._load_txt("classes.txt",     ["class_id", "class_name"])

        # Merge
        df = self.images.merge(self.splits, on="id").merge(self.labels, on="id")

        # Filter by split (1=train, 0=test)
        is_train = 1 if split == "train" else 0
        self.data = df[df["is_train"] == is_train].reset_index(drop=True)

        # Class names map (0-indexed)
        self.class_names = self.classes["class_name"].tolist()

        print(f"[{split.upper()}] Loaded {len(self.data)} images across {cfg.NUM_CLASSES} classes")

    def _load_txt(self, filename, columns):
        path = self.root / filename
        return pd.read_csv(path, sep=" ", header=None, names=columns)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = self.root / "images" / row["filename"]
        label = int(row["label"]) - 1  # Convert to 0-indexed

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(split="train"):
    """Returns augmented transforms for training, clean for val/test."""
    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(cfg.IMG_SIZE, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.2, hue=0.1),
            transforms.RandomRotation(20),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(cfg.MEAN, cfg.STD),
            transforms.RandomErasing(p=0.1),  # Cutout augmentation
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(cfg.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(cfg.MEAN, cfg.STD),
        ])


def get_dataloaders():
    train_dataset = CUBDataset(cfg.DATA_ROOT, split="train",
                               transform=get_transforms("train"))
    test_dataset  = CUBDataset(cfg.DATA_ROOT, split="test",
                               transform=get_transforms("test"))

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.BATCH_SIZE,
        shuffle=True, num_workers=cfg.NUM_WORKERS,
        pin_memory=True, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.BATCH_SIZE,
        shuffle=False, num_workers=cfg.NUM_WORKERS,
        pin_memory=True
    )
    return train_loader, test_loader, train_dataset.class_names


# ============================================================
# STEP 2: FEATURE EXTRACTION
# ============================================================

class BirdClassifier(nn.Module):
    """
    EfficientNetV2-B3 backbone with custom classification head.
    Uses timm library for the backbone.
    """
    def __init__(self, num_classes=200, dropout=0.4, pretrained=True):
        super().__init__()

        # Load EfficientNetV2-B3 backbone (pretrained on ImageNet)
        self.backbone = timm.create_model(
            "tf_efficientnetv2_b3",
            pretrained=pretrained,
            num_classes=0,       # Remove default classifier
            global_pool="avg"    # Global average pooling
        )
        feature_dim = self.backbone.num_features  # 1536 for EfficientNetV2-B3

        # Custom multi-layer classification head
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)   # Feature extraction
        logits = self.classifier(features)
        return logits

    def extract_features(self, x):
        """Returns raw feature vectors (for feature selection analysis)."""
        with torch.no_grad():
            return self.backbone(x)


# ============================================================
# STEP 3: FEATURE SELECTION
# ============================================================

def analyze_and_select_features(model, loader, device, n_samples=2000):
    """
    Extract features from pretrained backbone, apply:
    1. Variance Threshold — remove near-constant features
    2. PCA — reduce dimensionality and analyze variance explained
    """
    print("\n[FEATURE SELECTION] Extracting features from backbone...")
    model.eval()
    all_features, all_labels = [], []

    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(loader)):
            if len(all_features) * cfg.BATCH_SIZE >= n_samples:
                break
            images = images.to(device)
            feats = model.extract_features(images).cpu().numpy()
            all_features.append(feats)
            all_labels.extend(labels.numpy())

    X = np.vstack(all_features)
    y = np.array(all_labels)
    print(f"  Raw feature shape: {X.shape}")

    # --- Variance Threshold ---
    vt = VarianceThreshold(threshold=cfg.VARIANCE_THRESHOLD)
    X_vt = vt.fit_transform(X)
    n_removed = X.shape[1] - X_vt.shape[1]
    print(f"  After Variance Threshold: {X_vt.shape[1]} features "
          f"({n_removed} removed, threshold={cfg.VARIANCE_THRESHOLD})")

    # --- StandardScaler ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_vt)

    # --- PCA ---
    n_components = min(cfg.PCA_COMPONENTS, X_scaled.shape[1], X_scaled.shape[0])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    explained = np.cumsum(pca.explained_variance_ratio_)
    n95 = np.searchsorted(explained, 0.95) + 1
    print(f"  PCA: {n_components} components, "
          f"{n95} components explain 95% variance")

    # Plot PCA explained variance
    plt.figure(figsize=(9, 4))
    plt.plot(range(1, len(explained)+1), explained * 100, color="#4CAF50", lw=2)
    plt.axhline(95, color="red", linestyle="--", label="95% threshold")
    plt.axvline(n95, color="orange", linestyle="--", label=f"n={n95}")
    plt.xlabel("Number of PCA Components")
    plt.ylabel("Cumulative Explained Variance (%)")
    plt.title("PCA – Cumulative Explained Variance (CUB-200-2011 Features)")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{cfg.RESULTS_DIR}/pca_explained_variance.png", dpi=150)
    plt.close()
    print(f"  Saved PCA plot → {cfg.RESULTS_DIR}/pca_explained_variance.png")

    return {"vt": vt, "scaler": scaler, "pca": pca, "n95": n95}


# ============================================================
# STEP 4: TRAINING
# ============================================================

class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing loss — improves generalization on fine-grained datasets."""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_classes = pred.size(1)
        log_probs = torch.log_softmax(pred, dim=1)
        smooth_loss = -log_probs.mean(dim=1)
        nll_loss = -log_probs.gather(dim=1, index=target.unsqueeze(1)).squeeze(1)
        loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss, correct, total = 0, 0, 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [TRAIN]")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}",
                         acc=f"{100*correct/total:.2f}%")

    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, correct_top5, total = 0, 0, 0, 0
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in tqdm(loader, desc="[EVAL]"):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        correct += (preds == labels).sum().item()
        top5 = torch.topk(logits, 5, dim=1).indices
        correct_top5 += sum(labels[i] in top5[i] for i in range(len(labels)))
        total += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    return {
        "loss":    total_loss / len(loader),
        "top1":    correct / total,
        "top5":    correct_top5 / total,
        "preds":   np.array(all_preds),
        "labels":  np.array(all_labels),
        "probs":   np.array(all_probs),
    }


def train_model(model, train_loader, test_loader, class_names):
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(),
                            lr=cfg.LEARNING_RATE,
                            weight_decay=cfg.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.NUM_EPOCHS, eta_min=1e-6)

    best_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"\n{'='*60}")
    print(f" Training EfficientNetV2-B3 | {cfg.NUM_EPOCHS} epochs | {cfg.DEVICE}")
    print(f"{'='*60}\n")

    for epoch in range(cfg.NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, cfg.DEVICE, epoch)
        val_metrics = evaluate(model, test_loader, criterion, cfg.DEVICE)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["top1"])

        print(f"\nEpoch {epoch+1:3d}/{cfg.NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {100*train_acc:.2f}% | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val Top-1: {100*val_metrics['top1']:.2f}% | "
              f"Val Top-5: {100*val_metrics['top5']:.2f}%")

        # Save best model
        if val_metrics["top1"] > best_acc:
            best_acc = val_metrics["top1"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_acc": best_acc,
                "class_names": class_names,
            }, cfg.MODEL_SAVE_PATH)
            print(f"  ✅ Saved best model (Val Top-1: {100*best_acc:.2f}%)")

    # Save training history
    with open(f"{cfg.RESULTS_DIR}/training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    plot_training_history(history)
    return history, val_metrics


def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train", color="#2196F3")
    axes[0].plot(epochs, history["val_loss"],   label="Val",   color="#F44336")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, [a*100 for a in history["train_acc"]],
                 label="Train", color="#2196F3")
    axes[1].plot(epochs, [a*100 for a in history["val_acc"]],
                 label="Val", color="#F44336")
    axes[1].set_title("Accuracy (%)"); axes[1].set_xlabel("Epoch")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{cfg.RESULTS_DIR}/training_curves.png", dpi=150)
    plt.close()
    print(f"Saved training curves → {cfg.RESULTS_DIR}/training_curves.png")


# ============================================================
# STEP 5: EVALUATION & REPORTING
# ============================================================

def full_evaluation(model, test_loader, class_names):
    criterion = LabelSmoothingCrossEntropy()
    print("\n[EVALUATION] Running full evaluation on test set...")
    metrics = evaluate(model, test_loader, criterion, cfg.DEVICE)

    print(f"\n{'='*50}")
    print(f"  Final Test Top-1 Accuracy : {100*metrics['top1']:.2f}%")
    print(f"  Final Test Top-5 Accuracy : {100*metrics['top5']:.2f}%")
    print(f"{'='*50}\n")

    # Classification report (top 20 classes for brevity)
    report = classification_report(
        metrics["labels"], metrics["preds"],
        target_names=class_names,
        output_dict=True
    )
    report_df = pd.DataFrame(report).T
    report_df.to_csv(f"{cfg.RESULTS_DIR}/classification_report.csv")
    print(f"Saved classification report → {cfg.RESULTS_DIR}/classification_report.csv")

    # Confusion matrix (first 20 classes)
    n = 20
    mask = metrics["labels"] < n
    cm = confusion_matrix(metrics["labels"][mask], metrics["preds"][mask])
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, xticklabels=class_names[:n], yticklabels=class_names[:n],
                annot=True, fmt="d", cmap="Blues", linewidths=0.3)
    plt.title("Confusion Matrix (First 20 Bird Species)")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{cfg.RESULTS_DIR}/confusion_matrix.png", dpi=150)
    plt.close()
    print(f"Saved confusion matrix → {cfg.RESULTS_DIR}/confusion_matrix.png")

    return metrics


# ============================================================
# STEP 6: SINGLE IMAGE INFERENCE
# ============================================================

def load_model_for_inference(checkpoint_path, device):
    """Load trained model and class names from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint["class_names"]

    model = BirdClassifier(num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    print(f"✅ Loaded model (Best Val Acc: {100*checkpoint['best_acc']:.2f}%)")
    return model, class_names


def predict_single_image(image_path, model, class_names, device):
    """
    Classify a single bird image.
    Returns: dict with TOP-1 prediction and biodiversity info only.
    """
    transform = get_transforms("test")
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    # Get only top-1
    top_prob, top_idx = probs.topk(1)
    top_prob = float(top_prob.cpu().numpy()[0])
    top_idx  = int(top_idx.cpu().numpy()[0])       # convert numpy int64 -> Python int

    raw_name     = class_names[top_idx]
    species_name = raw_name.split(".", 1)[-1].replace("_", " ")
    class_id     = int(top_idx + 1)                # plain Python int, JSON-safe

    result = {
        "image_path":   str(image_path),
        "species":      species_name,
        "confidence":   round(top_prob * 100, 2),  # plain Python float
        "class_id":     class_id,
        "biodiversity": get_biodiversity_info(species_name, class_id)
    }
    return result


def get_biodiversity_info(species_name, class_id):
    """
    Returns biodiversity classification metadata.
    In production, enrich this from GBIF / eBird APIs.
    """
    # Simplified order mapping based on CUB families
    orders = {
        range(1, 10):   "Procellariiformes (Tubenoses)",
        range(10, 20):  "Pelecaniformes (Pelicans & allies)",
        range(20, 40):  "Charadriiformes (Shorebirds)",
        range(40, 80):  "Passeriformes (Perching birds)",
        range(80, 100): "Piciformes (Woodpeckers)",
        range(100, 140):"Passeriformes (Sparrows & Warblers)",
        range(140, 160):"Columbiformes (Doves & Pigeons)",
        range(160, 180):"Accipitriformes (Hawks & Eagles)",
        range(180, 201):"Anseriformes (Waterfowl)",
    }
    bird_order = "Aves (unclassified)"
    for r, o in orders.items():
        if class_id in r:
            bird_order = o
            break

    return {
        "kingdom":  "Animalia",
        "phylum":   "Chordata",
        "class":    "Aves",
        "order":    bird_order,
        "family":   "See eBird/GBIF for full taxonomy",
        "genus":    species_name.split()[0] if " " in species_name else "Unknown",
        "species":  species_name,
        "dataset":  "CUB-200-2011",
        "note":     "CUB-200-2011 contains 200 North American bird species."
    }


def print_prediction(result):
    bd = result["biodiversity"]
    print(f"\n{'='*60}")
    print(f" 🐦 BIRD BIODIVERSITY CLASSIFICATION RESULT")
    print(f"{'='*60}")
    print(f"  Image      : {result['image_path']}")
    print(f"  Species    : {result['species']}")
    print(f"  Confidence : {result['confidence']:.2f}%")
    print(f"\n  Biodiversity Classification:")
    print(f"    Kingdom  : {bd['kingdom']}")
    print(f"    Phylum   : {bd['phylum']}")
    print(f"    Class    : {bd['class']}")
    print(f"    Order    : {bd['order']}")
    print(f"    Family   : {bd['family']}")
    print(f"    Genus    : {bd['genus']}")
    print(f"    Species  : {bd['species']}")
    print(f"{'='*60}")


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main(force_retrain=False):
    """
    Full pipeline:
    1. Load data
    2. Build model
    3. Analyze features (feature selection)
    4. Train  ← SKIPPED automatically if checkpoint already exists
    5. Evaluate
    6. Interactive inference loop — asks user to upload images
    """
    print("\n🐦 CUB-200-2011 Bird Biodiversity Classifier")
    print("   Model: EfficientNetV2-B3 + Custom Head\n")

    # ── Check if already trained ───────────────────────────
    already_trained = os.path.exists(cfg.MODEL_SAVE_PATH) and not force_retrain

    if already_trained:
        print("✅ Checkpoint found — skipping training.")
        print(f"   Model loaded from: {cfg.MODEL_SAVE_PATH}\n")
        model, class_names = load_model_for_inference(cfg.MODEL_SAVE_PATH, cfg.DEVICE)
        interactive_inference(model, class_names)
        return model, class_names

    # ── Data ──────────────────────────────────────────────
    train_loader, test_loader, class_names = get_dataloaders()

    # ── Model ─────────────────────────────────────────────
    model = BirdClassifier(num_classes=cfg.NUM_CLASSES,
                           dropout=0.4, pretrained=True)
    model.to(cfg.DEVICE)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {total_params:.2f}M")

    # ── Feature Selection Analysis ─────────────────────────
    feature_info = analyze_and_select_features(model, train_loader, cfg.DEVICE)

    # ── Training ───────────────────────────────────────────
    history, _ = train_model(model, train_loader, test_loader, class_names)

    # ── Load best checkpoint & evaluate ───────────────────
    model, class_names = load_model_for_inference(cfg.MODEL_SAVE_PATH, cfg.DEVICE)
    metrics = full_evaluation(model, test_loader, class_names)

    print("\n✅ Training complete! Files saved to ./results/")

    # ── Interactive Inference Loop ─────────────────────────
    interactive_inference(model, class_names)

    return model, class_names


# ============================================================
# INTERACTIVE INFERENCE LOOP
# ============================================================

def interactive_inference(model=None, class_names=None):
    """
    After training, keeps asking the user to enter bird image
    paths and classifies them one by one until they type 'exit'.
    """
    # If called standalone, load model from checkpoint
    if model is None or class_names is None:
        if not os.path.exists(cfg.MODEL_SAVE_PATH):
            print("\n❌ No trained model found at:", cfg.MODEL_SAVE_PATH)
            print("   Please run the full training pipeline first.")
            return
        model, class_names = load_model_for_inference(cfg.MODEL_SAVE_PATH, cfg.DEVICE)

    print("\n" + "="*60)
    print("  🐦 BIRD BIODIVERSITY CLASSIFIER — READY FOR INFERENCE")
    print("="*60)
    print("  Enter the path to a bird image to classify it.")
    print("  Type 'exit' or press Ctrl+C to quit.\n")

    while True:
        try:
            # ── Prompt user for image path ─────────────────
            image_path = input("  📂 Enter bird image path: ").strip()

            # ── Exit condition ─────────────────────────────
            if image_path.lower() in ("exit", "quit", "q", ""):
                print("\n  👋 Exiting classifier. Goodbye!\n")
                break

            # ── Validate file exists ───────────────────────
            if not os.path.exists(image_path):
                print(f"  ❌ File not found: '{image_path}'")
                print("     Please check the path and try again.\n")
                continue

            # ── Validate it's an image ─────────────────────
            valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
            ext = os.path.splitext(image_path)[1].lower()
            if ext not in valid_exts:
                print(f"  ❌ Unsupported file type: '{ext}'")
                print(f"     Supported types: {', '.join(valid_exts)}\n")
                continue

            # ── Run classification ─────────────────────────
            print(f"\n  ⏳ Classifying '{os.path.basename(image_path)}'...")
            result = predict_single_image(
                image_path, model, class_names, cfg.DEVICE)
            print_prediction(result)

            # ── Save result to JSON ────────────────────────
            out_json = os.path.join(
                cfg.RESULTS_DIR,
                f"result_{os.path.splitext(os.path.basename(image_path))[0]}.json"
            )
            class _Encoder(json.JSONEncoder):
                def default(self, o):
                    if isinstance(o, (int, float)): return o
                    if hasattr(o, "item"): return o.item()  # numpy scalar
                    return super().default(o)
            with open(out_json, "w") as f:
                json.dump(result, f, indent=2, cls=_Encoder)
            print(f"  💾 Result saved → {out_json}")

            # ── Show result image with predictions ─────────
            try:
                show_result_plot(image_path, result)
            except Exception:
                pass  # Skip plot if display not available

            print("\n  " + "-"*56)
            print("  Enter another image path, or type 'exit' to quit.")
            print("  " + "-"*56 + "\n")

        except KeyboardInterrupt:
            print("\n\n  👋 Interrupted. Goodbye!\n")
            break
        except Exception as e:
            print(f"\n  ❌ Error during classification: {e}")
            print("     Please try a different image.\n")
            continue


def show_result_plot(image_path, result):
    """
    Displays the bird image with species and biodiversity info.
    Saved as PNG in results directory.
    """
    import matplotlib
    matplotlib.use("Agg")  # Safe for all environments

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    img = Image.open(image_path).convert("RGB")
    ax.imshow(img)
    ax.axis("off")

    bd = result["biodiversity"]
    title = (f"Species: {result['species']}  ({result['confidence']:.1f}%)\n"
             f"Order: {bd['order']}  |  Family: {bd['family']}")
    ax.set_title(title, fontsize=10, pad=10)

    plt.tight_layout()

    out_plot = os.path.join(
        cfg.RESULTS_DIR,
        f"plot_{os.path.splitext(os.path.basename(image_path))[0]}.png"
    )
    plt.savefig(out_plot, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Result plot saved → {out_plot}")


def inference_only(image_path=None):
    """
    Standalone inference — loads trained model and classifies.
    If no image_path given, starts the interactive loop.
    """
    model, class_names = load_model_for_inference(
        cfg.MODEL_SAVE_PATH, cfg.DEVICE)

    if image_path:
        # Single image mode
        result = predict_single_image(image_path, model, class_names, cfg.DEVICE)
        print_prediction(result)
        show_result_plot(image_path, result)
        return result
    else:
        # Interactive loop mode
        interactive_inference(model, class_names)


# ── Run ───────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) == 2 and sys.argv[1] not in ("--infer", "--retrain"):
        # python bird_biodiversity_classifier.py path/to/bird.jpg
        inference_only(sys.argv[1])

    elif len(sys.argv) == 2 and sys.argv[1] == "--infer":
        # python bird_biodiversity_classifier.py --infer
        # Skip training, just run interactive inference on saved model
        print("\n🐦 Loading saved model for inference...")
        interactive_inference()

    elif len(sys.argv) == 2 and sys.argv[1] == "--retrain":
        # python bird_biodiversity_classifier.py --retrain
        # Force retrain even if checkpoint exists
        print("\n🔁 Force retraining — ignoring existing checkpoint...\n")
        model, class_names = main(force_retrain=True)

    else:
        # ── SMART RUN: check if model already trained ──────
        if os.path.exists(cfg.MODEL_SAVE_PATH):
            print("\n✅ Trained model found at:", cfg.MODEL_SAVE_PATH)
            print("   Skipping training — jumping straight to inference.")
            print("   (To retrain from scratch, run with --retrain flag)\n")
            interactive_inference()
        else:
            print("\n🚀 No saved model found — starting full training pipeline...\n")
            model, class_names = main()