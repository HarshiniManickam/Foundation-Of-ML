"""
Bird Biodiversity Classification using CUB-200-2011 Dataset
Pipeline: Preprocessing → Gabor Filter Bank → Feature Extraction → PCA → SVM → Classification

Install requirements:
    pip install numpy pandas scikit-learn scikit-image opencv-python pillow matplotlib seaborn joblib tqdm
"""

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, top_k_accuracy_score)
from sklearn.pipeline import Pipeline
from skimage.filters import gabor
from skimage.color import rgb2gray
from skimage import exposure


# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
class Config:
    DATASET_ROOT      = "./CUB_200_2011"        # Path to CUB-200-2011 root folder
    IMAGE_SIZE        = (128, 128)               # Resize resolution
    N_COMPONENTS      = 150                      # PCA components kept
    TEST_SIZE         = 0.2
    RANDOM_STATE      = 42

    # ── Gabor Filter Bank (Plumage Texture) ──
    GABOR_FREQUENCIES = [0.1, 0.2, 0.3, 0.4]   # Spatial frequencies (cycles/pixel)
    GABOR_THETAS_DEG  = [0, 30, 60, 90, 120, 150]  # 6 orientations
    GABOR_SIGMA       = 2.0                      # Gaussian envelope std

    # ── SVM Hyperparameter Grid ───────────────
    SVM_PARAM_GRID = {
        'svm__C':      [0.1, 1, 10, 100],
        'svm__gamma':  ['scale', 'auto', 0.001, 0.01],
        'svm__kernel': ['rbf', 'poly']
    }

    MODEL_PATH    = "./bird_svm_model.pkl"
    SELECTOR_PFX  = "./bird_feature_selector"
    CLASSES_PATH  = "./bird_classes.pkl"
    REPORT_PATH   = "./classification_report.txt"


# ─────────────────────────────────────────────────────────────────────────────
# 2. DATASET LOADER — CUB-200-2011
# ─────────────────────────────────────────────────────────────────────────────
class CUBDatasetLoader:
    """
    Parses official CUB-200-2011 annotation files.

    Required files in DATASET_ROOT:
        images.txt              (image_id  relative_path)
        image_class_labels.txt  (image_id  class_id)
        train_test_split.txt    (image_id  is_training_image)
        classes.txt             (class_id  class_name)
        bounding_boxes.txt      (image_id  x  y  width  height)
        images/                 (actual JPEGs)
    """

    def __init__(self, root: str):
        self.root = Path(root)
        self._verify()

    def _verify(self):
        needed = ['images.txt', 'image_class_labels.txt',
                  'train_test_split.txt', 'classes.txt', 'bounding_boxes.txt']
        missing = [f for f in needed if not (self.root / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing in {self.root}: {missing}\n"
                "Download from: https://www.vision.caltech.edu/datasets/cub_200_2011/"
            )

    def load(self):
        images  = pd.read_csv(self.root/'images.txt', sep=' ', header=None,
                              names=['image_id','filepath'])
        labels  = pd.read_csv(self.root/'image_class_labels.txt', sep=' ', header=None,
                              names=['image_id','class_id'])
        splits  = pd.read_csv(self.root/'train_test_split.txt', sep=' ', header=None,
                              names=['image_id','is_train'])
        classes = pd.read_csv(self.root/'classes.txt', sep=' ', header=None,
                              names=['class_id','class_name'])
        bboxes  = pd.read_csv(self.root/'bounding_boxes.txt', sep=' ', header=None,
                              names=['image_id','x','y','width','height'])

        df = (images.merge(labels, on='image_id')
                    .merge(splits,  on='image_id')
                    .merge(classes, on='class_id')
                    .merge(bboxes,  on='image_id'))

        df['full_path'] = df['filepath'].apply(
            lambda p: str(self.root / 'images' / p))

        train_df = df[df['is_train'] == 1].reset_index(drop=True)
        test_df  = df[df['is_train'] == 0].reset_index(drop=True)
        class_names = classes['class_name'].tolist()

        print(f"  Dataset loaded  →  Train: {len(train_df)}  |  "
              f"Test: {len(test_df)}  |  Classes: {len(class_names)}")
        return train_df, test_df, class_names


# ─────────────────────────────────────────────────────────────────────────────
# 3. PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
class Preprocessor:
    """
    Steps applied to every image:
      1. Bounding-box crop   → isolate bird body, remove background
      2. Resize (128×128)    → uniform spatial resolution
      3. CLAHE               → local contrast enhancement (plumage detail)
      4. Gaussian blur       → mild denoising
    """

    def __init__(self, size=Config.IMAGE_SIZE):
        self.size  = size
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def preprocess(self, path: str, bbox=None) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            raise IOError(f"Cannot open: {path}")

        # 1. Crop
        if bbox is not None:
            x, y, w, h = [max(0, int(v)) for v in bbox]
            x2 = min(img.shape[1], x + w)
            y2 = min(img.shape[0], y + h)
            img = img[y:y2, x:x2]
            if img.size == 0:
                img = cv2.imread(path)   # fallback: full image

        # 2. Resize
        img = cv2.resize(img, self.size, interpolation=cv2.INTER_AREA)

        # 3. CLAHE on L channel
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # 4. Slight Gaussian smoothing
        img = cv2.GaussianBlur(img, (3, 3), sigmaX=0.5)

        return img   # BGR uint8


# ─────────────────────────────────────────────────────────────────────────────
# 4. GABOR FILTER BANK — PLUMAGE TEXTURE FEATURES
# ─────────────────────────────────────────────────────────────────────────────
class GaborFilterBank:
    """
    Captures plumage texture at multiple spatial frequencies and orientations.

    For each of the F×T filter combinations:
      → real response  mean + std
      → imag response  mean + std
    Total dims = F × T × 4  =  4 × 6 × 4  =  96
    """

    def __init__(self,
                 frequencies=Config.GABOR_FREQUENCIES,
                 thetas_deg=Config.GABOR_THETAS_DEG,
                 sigma=Config.GABOR_SIGMA):
        self.frequencies = frequencies
        self.thetas      = [np.deg2rad(t) for t in thetas_deg]
        self.sigma       = sigma

    @property
    def dim(self):
        return len(self.frequencies) * len(self.thetas) * 4

    def extract(self, img_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
        feats = []
        for freq in self.frequencies:
            for theta in self.thetas:
                real, imag = gabor(gray, frequency=freq, theta=theta,
                                   sigma_x=self.sigma, sigma_y=self.sigma)
                feats += [real.mean(), real.std(), imag.mean(), imag.std()]
        return np.array(feats, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 5. FULL FEATURE EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────
class FeatureExtractor:
    """
    Three complementary feature groups:
      A) Gabor texture  (plumage pattern & barring)  — 96 dims
      B) HSV histogram  (plumage color)              — 96 dims
      C) Gradient shape (body silhouette)            — 12 dims
                                               Total = 204 dims
    """

    def __init__(self):
        self.gabor = GaborFilterBank()

    @property
    def total_dim(self):
        return self.gabor.dim + 96 + 12   # 204

    # ── A ─ Gabor Texture ─────────────────────
    def _gabor(self, img):
        return self.gabor.extract(img)

    # ── B ─ HSV Color Histogram ───────────────
    def _color(self, img, bins=32):
        hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        feat = []
        for ch in range(3):
            h = cv2.calcHist([hsv], [ch], None, [bins], [0, 256]).flatten()
            h /= (h.sum() + 1e-7)
            feat.extend(h)
        return np.array(feat, dtype=np.float32)

    # ── C ─ Gradient / Shape ─────────────────
    def _shape(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gx   = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy   = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag  = np.hypot(gx, gy)
        ang  = np.arctan2(gy, gx) * 180 / np.pi % 180
        hist, _ = np.histogram(ang, bins=9, range=(0, 180),
                               weights=mag, density=True)
        stats = [mag.mean(), mag.std(), mag.max()]
        return np.concatenate([hist, stats]).astype(np.float32)

    def extract(self, img_bgr: np.ndarray) -> np.ndarray:
        return np.concatenate([
            self._gabor(img_bgr),
            self._color(img_bgr),
            self._shape(img_bgr)
        ])


# ─────────────────────────────────────────────────────────────────────────────
# 6. FEATURE SELECTION via PCA
# ─────────────────────────────────────────────────────────────────────────────
class FeatureSelector:
    """StandardScaler → PCA(n_components)"""

    def __init__(self, n=Config.N_COMPONENTS):
        self.scaler = StandardScaler()
        self.pca    = PCA(n_components=n, random_state=Config.RANDOM_STATE)

    def fit_transform(self, X):
        Xs = self.scaler.fit_transform(X)
        Xp = self.pca.fit_transform(Xs)
        ev = self.pca.explained_variance_ratio_.cumsum()[-1]
        print(f"  PCA {Config.N_COMPONENTS} components → {ev*100:.1f}% variance explained")
        return Xp

    def transform(self, X):
        return self.pca.transform(self.scaler.transform(X))

    def save(self, prefix):
        joblib.dump(self.scaler, f"{prefix}_scaler.pkl")
        joblib.dump(self.pca,    f"{prefix}_pca.pkl")
        print(f"  Selector saved → {prefix}_*.pkl")

    @classmethod
    def load(cls, prefix):
        obj = cls.__new__(cls)
        obj.scaler = joblib.load(f"{prefix}_scaler.pkl")
        obj.pca    = joblib.load(f"{prefix}_pca.pkl")
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# 7. SVM TRAINER
# ─────────────────────────────────────────────────────────────────────────────
class SVMTrainer:
    """
    RBF-SVM with optional GridSearchCV.
    probability=True enables confidence scores.
    class_weight='balanced' handles CUB's uneven per-class counts.
    """

    def __init__(self):
        self.clf = None

    def train(self, Xtr, ytr, grid_search=False):
        base = SVC(kernel='rbf', probability=True,
                   class_weight='balanced',
                   random_state=Config.RANDOM_STATE)

        if grid_search:
            print("  Running GridSearchCV  (C, gamma, kernel) …")
            pipe = Pipeline([('svm', base)])
            cv   = StratifiedKFold(5, shuffle=True, random_state=42)
            gs   = GridSearchCV(pipe, Config.SVM_PARAM_GRID,
                                cv=cv, scoring='accuracy',
                                n_jobs=-1, verbose=2)
            gs.fit(Xtr, ytr)
            self.clf = gs.best_estimator_
            print(f"  Best: {gs.best_params_}  CV-acc={gs.best_score_*100:.2f}%")
        else:
            print("  Training SVM (C=10, gamma=scale, kernel=rbf) …")
            self.clf = SVC(kernel='rbf', C=10, gamma='scale',
                           probability=True, class_weight='balanced',
                           random_state=42)
            self.clf.fit(Xtr, ytr)

    def evaluate(self, Xte, yte, class_names):
        yp    = self.clf.predict(Xte)
        proba = self.clf.predict_proba(Xte)
        acc   = accuracy_score(yte, yp)
        top5  = top_k_accuracy_score(yte, proba, k=5)

        print(f"\n{'='*58}")
        print(f"  Top-1 Accuracy : {acc*100:.2f}%")
        print(f"  Top-5 Accuracy : {top5*100:.2f}%")
        print(f"{'='*58}\n")
        rpt = classification_report(yte, yp, target_names=class_names, zero_division=0)
        print(rpt)

        with open(Config.REPORT_PATH, 'w') as f:
            f.write(f"Top-1: {acc*100:.2f}%\nTop-5: {top5*100:.2f}%\n\n{rpt}")

        self._plot_cm(yte, yp, class_names)
        return acc, top5

    def _plot_cm(self, yt, yp, names, top_n=20):
        cm  = confusion_matrix(yt, yp)
        err = cm.sum(1) - np.diag(cm)
        idx = np.argsort(err)[-top_n:]
        sub = cm[np.ix_(idx, idx)]
        lbl = [names[i].split('.')[-1].replace('_',' ') for i in idx]

        plt.figure(figsize=(16, 14))
        sns.heatmap(sub, annot=True, fmt='d', cmap='Blues',
                    xticklabels=lbl, yticklabels=lbl, linewidths=.3)
        plt.title(f'Confusion Matrix — Top-{top_n} Confused Classes',
                  fontweight='bold')
        plt.ylabel('True'); plt.xlabel('Predicted')
        plt.xticks(rotation=45, ha='right', fontsize=7)
        plt.yticks(fontsize=7)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=150)
        plt.close()
        print("  Saved → confusion_matrix.png")

    def save(self, path):
        joblib.dump(self.clf, path)
        print(f"  Model saved → {path}")

    @classmethod
    def load(cls, path):
        o = cls.__new__(cls)
        o.clf = joblib.load(path)
        return o


# ─────────────────────────────────────────────────────────────────────────────
# 8. HELPER: build feature matrix from a dataframe
# ─────────────────────────────────────────────────────────────────────────────
def _build_matrix(df, prep, ext, label):
    feats = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=label):
        bbox = (row['x'], row['y'], row['width'], row['height'])
        try:
            img  = prep.preprocess(row['full_path'], bbox=bbox)
            feat = ext.extract(img)
        except Exception as e:
            print(f"\n  [WARN] {row['full_path']}: {e}")
            feat = np.zeros(ext.total_dim, dtype=np.float32)
        feats.append(feat)
    return np.array(feats, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 9. TRAINING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def train_pipeline(grid_search=False):
    print("\n" + "="*60)
    print("   BIRD BIODIVERSITY CLASSIFIER — TRAINING")
    print("="*60)

    # Step 1 — Load dataset
    print("\n[1/5] Loading CUB-200-2011 …")
    loader = CUBDatasetLoader(Config.DATASET_ROOT)
    train_df, test_df, class_names = loader.load()

    # Step 2 — Preprocess + Extract features
    print(f"\n[2/5] Extracting features …")
    prep = Preprocessor()
    ext  = FeatureExtractor()
    print(f"  Feature dimensions: {ext.total_dim}  "
          f"(Gabor={ext.gabor.dim}, Color=96, Shape=12)")

    X_train = _build_matrix(train_df, prep, ext, "Train")
    y_train = train_df['class_id'].values - 1   # 0-indexed

    X_test  = _build_matrix(test_df, prep, ext, "Test ")
    y_test  = test_df['class_id'].values - 1

    # Step 3 — Feature selection (PCA)
    print(f"\n[3/5] PCA feature selection (n={Config.N_COMPONENTS}) …")
    sel = FeatureSelector()
    Xtr_pca = sel.fit_transform(X_train)
    Xte_pca = sel.transform(X_test)
    sel.save(Config.SELECTOR_PFX)

    # Step 4 — Train SVM
    print(f"\n[4/5] Training SVM …")
    trainer = SVMTrainer()
    trainer.train(Xtr_pca, y_train, grid_search=grid_search)
    trainer.save(Config.MODEL_PATH)

    # Step 5 — Evaluate
    print(f"\n[5/5] Evaluating …")
    trainer.evaluate(Xte_pca, y_test, class_names)

    joblib.dump(class_names, Config.CLASSES_PATH)
    print(f"\n  Classes saved → {Config.CLASSES_PATH}")
    print("\n[DONE] Training complete.\n")


# ─────────────────────────────────────────────────────────────────────────────
# 10. INFERENCE / PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────
class BirdBiodiversityPredictor:
    """
    Load trained artifacts and classify a user-uploaded bird image.
    Returns top-k species predictions with confidence percentages.
    """

    # Rough biodiversity metadata mapping for CUB classes
    BIODIVERSITY_NOTES = {
        "Albatross":    {"order":"Procellariiformes","family":"Diomedeidae","status":"Endangered"},
        "Warbler":      {"order":"Passeriformes",   "family":"Parulidae",   "status":"Least Concern"},
        "Hummingbird":  {"order":"Apodiformes",     "family":"Trochilidae", "status":"Least Concern"},
        "Sparrow":      {"order":"Passeriformes",   "family":"Passerellidae","status":"Least Concern"},
        "Woodpecker":   {"order":"Piciformes",      "family":"Picidae",     "status":"Least Concern"},
        "Flycatcher":   {"order":"Passeriformes",   "family":"Tyrannidae",  "status":"Least Concern"},
        "Finch":        {"order":"Passeriformes",   "family":"Fringillidae","status":"Least Concern"},
        "Gull":         {"order":"Charadriiformes", "family":"Laridae",     "status":"Least Concern"},
        "Tern":         {"order":"Charadriiformes", "family":"Laridae",     "status":"Varies"},
        "Grebe":        {"order":"Podicipediformes","family":"Podicipedidae","status":"Least Concern"},
        "Pelican":      {"order":"Pelecaniformes",  "family":"Pelecanidae", "status":"Least Concern"},
        "Wren":         {"order":"Passeriformes",   "family":"Troglodytidae","status":"Least Concern"},
        "Oriole":       {"order":"Passeriformes",   "family":"Icteridae",   "status":"Least Concern"},
        "Vireo":        {"order":"Passeriformes",   "family":"Vireonidae",  "status":"Least Concern"},
        "Kingfisher":   {"order":"Coraciiformes",   "family":"Alcedinidae", "status":"Least Concern"},
    }

    def __init__(self):
        self.clf      = SVMTrainer.load(Config.MODEL_PATH).clf
        self.sel      = FeatureSelector.load(Config.SELECTOR_PFX)
        self.classes  = joblib.load(Config.CLASSES_PATH)
        self.prep     = Preprocessor()
        self.ext      = FeatureExtractor()
        print("[Predictor] Loaded. Ready to classify.")

    def _bio_info(self, species_name: str) -> dict:
        for keyword, info in self.BIODIVERSITY_NOTES.items():
            if keyword.lower() in species_name.lower():
                return info
        return {"order": "Passeriformes", "family": "Unknown",
                "status": "Data Deficient"}

    def predict(self, image_path: str, top_k: int = 5) -> list:
        img     = self.prep.preprocess(image_path)
        feat    = self.ext.extract(img).reshape(1, -1)
        feat_p  = self.sel.transform(feat)
        proba   = self.clf.predict_proba(feat_p)[0]
        top_idx = np.argsort(proba)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_idx, 1):
            raw       = self.classes[idx]
            num, name = (raw.split('.', 1) + [raw])[:2]
            species   = name.replace('_', ' ')
            bio       = self._bio_info(species)
            results.append({
                'rank'      : rank,
                'class_id'  : num,
                'species'   : species,
                'order'     : bio['order'],
                'family'    : bio['family'],
                'status'    : bio['status'],
                'confidence': float(proba[idx]),
            })
        return results

    def visualize(self, image_path: str, top_k=5):
        results = self.predict(image_path, top_k)
        img_rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        fig = plt.figure(figsize=(15, 6), facecolor='#0D1117')
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        # ── Left: image ──
        ax1.imshow(img_rgb)
        ax1.set_title("Input Image", color='white', fontsize=13, fontweight='bold')
        ax1.axis('off')
        ax1.set_facecolor('#0D1117')

        # ── Right: bar chart ──
        species = [r['species'] for r in results]
        confs   = [r['confidence'] * 100 for r in results]
        colors  = ['#00FF9F' if i == 0 else '#4FC3F7' for i in range(len(species))]

        bars = ax2.barh(species[::-1], confs[::-1],
                        color=colors[::-1], edgecolor='#1F2937')
        for bar, val in zip(bars, confs[::-1]):
            ax2.text(bar.get_width() + 0.5,
                     bar.get_y() + bar.get_height() / 2,
                     f'{val:.1f}%', va='center', color='white', fontsize=9)

        ax2.set_facecolor('#161B22')
        ax2.set_xlabel('Confidence (%)', color='white')
        ax2.set_title(f'Top-{top_k} Bird Species', color='white',
                      fontsize=13, fontweight='bold')
        ax2.tick_params(colors='white')
        ax2.set_xlim(0, 115)
        for spine in ax2.spines.values():
            spine.set_edgecolor('#30363D')

        fig.suptitle('🐦 Bird Biodiversity Classification',
                     color='white', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('prediction_result.png', dpi=150,
                    facecolor='#0D1117', bbox_inches='tight')
        plt.show()

        print("\n" + "="*60)
        print("  BIODIVERSITY CLASSIFICATION RESULT")
        print("="*60)
        for r in results:
            print(f"  #{r['rank']}  {r['species']:<35} {r['confidence']*100:6.2f}%"
                  f"  |  Order: {r['order']}  |  IUCN: {r['status']}")
        print("="*60 + "\n")
        return results


# ─────────────────────────────────────────────────────────────────────────────
# 11. MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Bird Biodiversity Classifier — CUB-200-2011 + Gabor + SVM")
    ap.add_argument('--mode', choices=['train', 'predict'], default='train')
    ap.add_argument('--image', type=str, help="Image path for prediction")
    ap.add_argument('--grid_search', action='store_true',
                    help="Enable GridSearchCV (slower but better)")
    ap.add_argument('--top_k', type=int, default=5)
    args = ap.parse_args()

    if args.mode == 'train':
        train_pipeline(grid_search=args.grid_search)
    else:
        if not args.image:
            ap.error("--image required for predict mode")
        p = BirdBiodiversityPredictor()
        p.visualize(args.image, top_k=args.top_k)