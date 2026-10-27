import os
import sys
import glob
import re
import time
import random
import zipfile
import argparse
import threading
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")  # Headless mode for saving plots
import matplotlib.pyplot as plt
from PIL import Image, ImageTk

from skimage.morphology import skeletonize
from skimage.feature import hog, local_binary_pattern
from skimage.metrics import structural_similarity
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import joblib

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ==================================================================================================
# CONFIGURATION & REPRODUCIBILITY
# ==================================================================================================
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DATASET_PATH = "archive.zip"
EXTRACTED_DIR = "dataset_extracted"
OUTPUTS_DIR = "outputs"
MODEL_PATH = "signature_svm_model.joblib"
TARGET_SIZE = (128, 256)  # Height=128, Width=256
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# ==================================================================================================
# MODULE 1: PREPROCESSING MODULE
# ==================================================================================================
def load_image(source: Union[str, np.ndarray]) -> np.ndarray:
    """Safely loads an image from file path or array (supports Unicode/Windows paths)."""
    if isinstance(source, np.ndarray):
        return source.copy()
    if not os.path.isfile(source):
        raise FileNotFoundError(f"Image not found: {source}")
    with open(source, "rb") as f:
        buf = f.read()
    img = cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to decode image: {source}")
    return img


def preprocess_signature(image_source: Union[str, np.ndarray], target_size: Tuple[int, int] = TARGET_SIZE) -> Dict[str, Any]:
    """
    Executes the full image preprocessing pipeline and returns intermediate images:
    1. Grayscale conversion
    2. Median & Gaussian noise reduction
    3. Otsu & Adaptive binarization (inverted: ink=255, paper=0)
    4. Morphological opening (speck removal) and closing (gap reconnection)
    5. Auto-crop to signature bounding box
    6. Aspect-ratio preserving resize with zero padding
    7. Skeletonization (thinning to 1-pixel medial axis)
    """
    # 1. Load BGR and convert to Grayscale
    orig = load_image(image_source)
    gray = orig if len(orig.shape) == 2 else cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    
    # 2. Noise reduction
    med = cv2.medianBlur(gray, 3)
    blurred = cv2.GaussianBlur(med, (3, 3), 0)
    
    # 3. Binarization (Otsu + Adaptive, inverted so ink is white)
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    adapt = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 8)
    binary = cv2.bitwise_or(otsu, adapt)
    
    # 4. Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_sqr, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    
    # 5. Auto-crop bounding box
    pts = cv2.findNonZero(cleaned)
    if pts is not None:
        x, y, w, h = cv2.boundingRect(pts)
        cropped = cleaned[max(0, y-2):min(cleaned.shape[0], y+h+2), max(0, x-2):min(cleaned.shape[1], x+w+2)]
    else:
        cropped = cleaned
        x, y, w, h = 0, 0, cleaned.shape[1], cleaned.shape[0]
        
    crop_h, crop_w = max(1, cropped.shape[0]), max(1, cropped.shape[1])
    
    # 6. Aspect-ratio preserving resize with padding to target_size (128x256)
    th, tw = target_size
    scale = min((tw - 16) / crop_w, (th - 16) / crop_h)
    nw, nh = max(1, int(crop_w * scale)), max(1, int(crop_h * scale))
    resized = cv2.resize(cropped, (nw, nh), interpolation=cv2.INTER_AREA)
    
    canvas = np.zeros((th, tw), dtype=np.uint8)
    yo, xo = (th - nh) // 2, (tw - nw) // 2
    canvas[yo:yo + nh, xo:xo + nw] = resized
    
    # 7. Skeletonization (1-pixel medial axis thinning)
    skeleton = (skeletonize(canvas > 0) * 255).astype(np.uint8)
    
    return {
        "original": orig,
        "grayscale": gray,
        "binarized": binary,
        "cleaned": cleaned,
        "cropped": cropped,
        "normalized": canvas,
        "skeleton": skeleton,
        "crop_shape": (crop_h, crop_w),
        "orig_shape": orig.shape[:2]
    }


# ==================================================================================================
# MODULE 2: FEATURE EXTRACTION MODULE
# ==================================================================================================
def extract_geometric_features(prep: Dict[str, Any]) -> np.ndarray:
    """Extracts 31 geometric features (aspect ratio, density, centroid, contours, Hu moments, projections)."""
    canvas = prep["normalized"]
    skeleton = prep["skeleton"]
    crop_h, crop_w = prep["crop_shape"]
    orig_h, orig_w = prep["orig_shape"]
    th, tw = canvas.shape
    
    aspect_ratio = float(crop_w) / float(crop_h + 1e-6)
    ink_density = float(np.count_nonzero(canvas)) / float(canvas.size)
    skel_density = float(np.count_nonzero(skeleton)) / float(skeleton.size)
    
    # Centroid from spatial moments
    M = cv2.moments(canvas)
    cx = (M["m10"] / (M["m00"] + 1e-6)) / float(tw)
    cy = (M["m01"] / (M["m00"] + 1e-6)) / float(th)
    
    # Contours (number of strokes, pen lifts)
    cnts, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_cnts = float(len(cnts))
    max_cnt_area = max([cv2.contourArea(c) for c in cnts], default=0.0) / float(canvas.size)
    
    # 7 Hu Moments (scale, translation, and 2D rotation invariant; log transformed)
    hu = cv2.HuMoments(M).flatten()
    hu_feats = [-1.0 * np.sign(h) * np.log10(np.abs(h) + 1e-10) for h in hu]
    
    # Horizontal & Vertical projection profiles (8 bins each)
    h_proj = [float(np.mean(b)) / (th + 1e-6) for b in np.array_split(np.sum(canvas > 0, axis=0), 8)]
    v_proj = [float(np.mean(b)) / (tw + 1e-6) for b in np.array_split(np.sum(canvas > 0, axis=1), 8)]
    
    geom = [aspect_ratio, ink_density, skel_density, cx, cy, num_cnts, max_cnt_area] + hu_feats + h_proj + v_proj
    return np.array(geom, dtype=np.float32)


def extract_features(image_or_prep: Union[str, np.ndarray, Dict[str, Any]]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Extracts a 712-dimensional unified feature vector:
      [Geometric (31)] + [HOG Gradients (672)] + [LBP Texture (10)]
    """
    prep = image_or_prep if isinstance(image_or_prep, dict) and "normalized" in image_or_prep else preprocess_signature(image_or_prep)
    canvas = prep["normalized"]
    
    # 1. Geometric features
    geom = extract_geometric_features(prep)
    
    # 2. HOG features (orientations=8, cell=32x32)
    hog_feat = hog(canvas, orientations=8, pixels_per_cell=(32, 32), cells_per_block=(2, 2), block_norm="L2-Hys").astype(np.float32)
    
    # 3. LBP micro-texture histogram (P=8, R=1 uniform patterns)
    lbp = local_binary_pattern(canvas, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)
    
    combined = np.hstack([geom, hog_feat, lbp_hist.astype(np.float32)])
    return combined, prep


# ==================================================================================================
# MODULE 3: VERIFICATION MODULE (WRITER-INDEPENDENT SVM & BASELINE)
# ==================================================================================================
class SignatureVerifier:
    """Offline signature verification using Pairwise SVM (Mode A) or Baseline SSIM/Distance (Mode B)."""
    def __init__(self, model_file: Optional[str] = MODEL_PATH):
        self.model_file = model_file
        self.model: Optional[Pipeline] = None
        self.is_trained = False
        if model_file and os.path.exists(model_file):
            self.load(model_file)

    def train(self, X_pairs: np.ndarray, y_pairs: np.ndarray) -> float:
        """Trains writer-independent SVM on absolute pairwise difference vectors |f_A - f_B|."""
        print(f"[Verifier] Training SVM on {len(X_pairs)} signature pairs...")
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=1.5, gamma="scale", random_state=RANDOM_SEED))
        ])
        self.model.fit(X_pairs, y_pairs)
        self.is_trained = True
        acc = float(accuracy_score(y_pairs, self.model.predict(X_pairs)))
        print(f"[Verifier] Training complete. Train Accuracy: {acc * 100:.2f}%")
        return acc

    def verify(self, ref_input: Union[str, np.ndarray, List], test_input: Union[str, np.ndarray], mode: str = "A", threshold: float = 0.65) -> Dict[str, Any]:
        """Verifies a questioned signature against one or more reference genuine signatures."""
        ref_list = ref_input if isinstance(ref_input, list) else [ref_input]
        ref_feats, ref_preps = zip(*[extract_features(r) for r in ref_list])
        test_feat, test_prep = extract_features(test_input)
        
        # Average reference features for multi-enrollment
        mean_ref_feat = np.mean(ref_feats, axis=0)
        canvas_ref = ref_preps[0]["normalized"]
        canvas_test = test_prep["normalized"]
        
        # Mode B: Baseline SSIM and Cosine Similarity
        cos_sim = float(np.dot(mean_ref_feat, test_feat) / (np.linalg.norm(mean_ref_feat) * np.linalg.norm(test_feat) + 1e-7))
        cos_sim_norm = float(np.clip((cos_sim + 1.0) / 2.0, 0.0, 1.0))
        ssim_val, _ = structural_similarity(canvas_ref, canvas_test, full=True)
        ssim_norm = float(np.clip((ssim_val + 1.0) / 2.0, 0.0, 1.0))
        base_score = 0.5 * cos_sim_norm + 0.5 * ssim_norm
        
        if mode.upper() == "A" and self.is_trained:
            # Mode A: Writer-Independent SVM
            diff = np.abs(mean_ref_feat - test_feat).reshape(1, -1)
            dec_val = float(self.model.decision_function(diff)[0])
            conf = float(1.0 / (1.0 + np.exp(-dec_val)))  # Sigmoid confidence
            is_gen = bool(conf >= 0.5)
            mode_desc = "Mode A (Writer-Independent SVM)"
        else:
            conf = base_score
            is_gen = bool(base_score >= threshold)
            mode_desc = "Mode B (SSIM + Cosine Baseline)"
            
        return {
            "prediction": "GENUINE" if is_gen else "FORGED",
            "is_genuine": is_gen,
            "confidence": conf,
            "similarity_score": base_score,
            "cosine_sim": cos_sim_norm,
            "ssim": ssim_norm,
            "mode": mode_desc,
            "ref_prep": ref_preps[0],
            "test_prep": test_prep
        }

    def save(self, path: str = MODEL_PATH):
        joblib.dump(self.model, path)
        print(f"[Verifier] Saved model to '{path}'")

    def load(self, path: str = MODEL_PATH) -> bool:
        if os.path.exists(path):
            self.model = joblib.load(path)
            self.is_trained = True
            print(f"[Verifier] Loaded model from '{path}'")
            return True
        return False


# ==================================================================================================
# MODULE 4: DATASET HANDLING (ZIP EXTRACTION & PAIR GENERATION)
# ==================================================================================================
def extract_dataset(zip_path: str = DATASET_PATH, extract_dir: str = EXTRACTED_DIR, max_writers: Optional[int] = None) -> str:
    """Unzips dataset into extract_dir if not already extracted."""
    if not os.path.isfile(zip_path):
        return extract_dir
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        members = [m for m in z.namelist() if not m.startswith("__MACOSX/") and not m.endswith("/")]
        if max_writers:
            writers = set()
            filtered = []
            for m in members:
                p = m.replace("\\", "/").split("/")
                folder = p[-2] if len(p) > 1 else ""
                w = re.sub(r"[_-]+(forg|fake|gen|orig).*$", "", folder, flags=re.I)
                if w: writers.add(w)
                if len(writers) <= max_writers: filtered.append(m)
            members = filtered
            
        existing = sum(len(files) for _, _, files in os.walk(extract_dir))
        if existing < len(members):
            print(f"[Dataset] Extracting {len(members)} files from '{zip_path}'...")
            z.extractall(extract_dir, members=members)
            print(f"[Dataset] Extraction complete: '{extract_dir}' ready.")
    return extract_dir


def load_dataset(dataset_root: str, max_writers: Optional[int] = None) -> Tuple[Dict[str, Dict[str, List[str]]], str]:
    """Scans dataset directory, identifies genuine vs forged signatures, and groups them by writer ID."""
    dataset: Dict[str, Dict[str, List[str]]] = {}
    layout_name = "Per-Writer Folders"
    
    for root, _, files in os.walk(dataset_root):
        if "__MACOSX" in root: continue
        for f in files:
            if not f.lower().endswith(IMAGE_EXTS): continue
            full_path = os.path.join(root, f)
            rel = os.path.relpath(full_path, dataset_root).replace("\\", "/")
            parts = rel.split("/")
            fname, parent = parts[-1], parts[-2] if len(parts) > 1 else ""
            
            # Layout matching: CEDAR, Per-Writer sibling folders, or Subfolders
            if "original_" in fname.lower() or "full_org" in parent.lower():
                m = re.search(r"original_(\d+)_", fname, re.I)
                w_id, cat, layout_name = (m.group(1) if m else "1"), "genuine", "CEDAR"
            elif "forgeries_" in fname.lower() or "full_forg" in parent.lower():
                m = re.search(r"forgeries_(\d+)_", fname, re.I)
                w_id, cat, layout_name = (m.group(1) if m else "1"), "forged", "CEDAR"
            elif any(k in parent.lower() for k in ("forg", "fake")):
                w_id = re.sub(r"[_-]+(forg|fake|forgery).*$", "", parent, flags=re.I)
                cat = "forged"
            else:
                w_id = re.sub(r"[_-]+(gen|real|orig).*$", "", parent, flags=re.I)
                cat = "genuine"
                
            if w_id not in dataset:
                dataset[w_id] = {"genuine": [], "forged": []}
            dataset[w_id][cat].append(full_path)
            
    # Filter writers with genuine samples
    valid_ds = {w: d for w, d in dataset.items() if len(d["genuine"]) > 0}
    if max_writers and len(valid_ds) > max_writers:
        valid_ds = {k: valid_ds[k] for k in sorted(list(valid_ds.keys()))[:max_writers]}
        
    num_gen = sum(len(d["genuine"]) for d in valid_ds.values())
    num_forg = sum(len(d["forged"]) for d in valid_ds.values())
    print(f"[Dataset] Loaded {layout_name}: {len(valid_ds)} writers, {num_gen} genuine, {num_forg} forged.")
    return valid_ds, layout_name


def generate_synthetic_dataset(output_dir: str = "dataset_synthetic", num_writers: int = 8) -> str:
    """Generates synthetic handwriting signatures using Bezier curves so the project always runs."""
    print(f"[Synthetic] Generating synthetic dataset ({num_writers} writers)...")
    os.makedirs(output_dir, exist_ok=True)
    
    for w in range(1, num_writers + 1):
        w_id = f"{w:04d}"
        gen_dir = os.path.join(output_dir, w_id)
        forg_dir = os.path.join(output_dir, f"{w_id}_forg")
        os.makedirs(gen_dir, exist_ok=True); os.makedirs(forg_dir, exist_ok=True)
        
        # Base writer template anchors
        rng = np.random.RandomState(w * 10)
        anchors = [((rng.randint(30, 80), rng.randint(40, 110)),
                    (rng.randint(110, 180), rng.randint(30, 120)),
                    (rng.randint(200, 270), rng.randint(50, 110))) for _ in range(4)]
        base_slant = rng.uniform(-0.1, 0.1)
        
        for idx in range(6):
            # Genuine: subtle variation
            img_g = np.ones((140, 300), dtype=np.uint8) * 255
            for p1, p2, p3 in anchors:
                pts = [((1-t)**2 * p1[0] + 2*(1-t)*t * p2[0] + t**2 * p3[0] + base_slant * (140-((1-t)**2 * p1[1] + 2*(1-t)*t * p2[1] + t**2 * p3[1])) + np.random.normal(0, 0.5),
                        (1-t)**2 * p1[1] + 2*(1-t)*t * p2[1] + t**2 * p3[1] + np.random.normal(0, 0.5)) for t in np.linspace(0, 1, 30)]
                for k in range(len(pts)-1):
                    cv2.line(img_g, (int(pts[k][0]), int(pts[k][1])), (int(pts[k+1][0]), int(pts[k+1][1])), 0, 2, cv2.LINE_AA)
            cv2.imwrite(os.path.join(gen_dir, f"{w_id}_{idx+1:02d}.jpg"), img_g)
            
            # Forged: distorted slant and stroke tremor
            img_f = np.ones((140, 300), dtype=np.uint8) * 255
            slant_f = base_slant + np.random.uniform(-0.25, 0.25)
            for p1, p2, p3 in anchors:
                pts = [((1-t)**2 * p1[0] + 2*(1-t)*t * p2[0] + t**2 * p3[0] + slant_f * (140-((1-t)**2 * p1[1] + 2*(1-t)*t * p2[1] + t**2 * p3[1])) + np.random.normal(0, 2.0),
                        (1-t)**2 * p1[1] + 2*(1-t)*t * p2[1] + t**2 * p3[1] + np.random.normal(0, 2.0)) for t in np.linspace(0, 1, 30)]
                for k in range(len(pts)-1):
                    cv2.line(img_f, (int(pts[k][0]), int(pts[k][1])), (int(pts[k+1][0]), int(pts[k+1][1])), 0, 2, cv2.LINE_AA)
            cv2.imwrite(os.path.join(forg_dir, f"{w_id}F_{idx+1:02d}.jpg"), img_f)
            
    print(f"[Synthetic] Synthetic signatures created in '{output_dir}'.")
    return output_dir


def build_pairs(dataset: Dict[str, Dict[str, List[str]]], writer_ids: List[str], max_pairs: int = 3000) -> Tuple[np.ndarray, np.ndarray]:
    """Generates balanced pairs: Genuine-Genuine = 1, Genuine-Forged = 0."""
    t0 = time.time()
    feat_cache: Dict[str, np.ndarray] = {}
    gen_diffs, forg_diffs = [], []
    target_each = max_pairs // 2
    
    shuffled_writers = list(writer_ids)
    random.seed(RANDOM_SEED)
    random.shuffle(shuffled_writers)
    
    for w in shuffled_writers:
        g_paths = dataset[w]["genuine"]
        f_paths = dataset[w]["forged"]
        
        for p in g_paths + f_paths:
            if p not in feat_cache:
                try: feat_cache[p] = extract_features(p)[0]
                except Exception: pass
                
        vg = [p for p in g_paths if p in feat_cache]
        vf = [p for p in f_paths if p in feat_cache]
        
        # Genuine-Genuine pairs
        for i in range(len(vg)):
            for j in range(i + 1, len(vg)):
                gen_diffs.append(np.abs(feat_cache[vg[i]] - feat_cache[vg[j]]))
        # Genuine-Forged pairs
        for g in vg:
            for f in vf:
                forg_diffs.append(np.abs(feat_cache[g] - feat_cache[f]))
                
        if len(gen_diffs) >= target_each and len(forg_diffs) >= target_each:
            break
            
    n = min(len(gen_diffs), len(forg_diffs), target_each)
    random.seed(RANDOM_SEED)
    X = np.vstack([random.sample(gen_diffs, n), random.sample(forg_diffs, n)])
    y = np.array([1] * n + [0] * n, dtype=np.int32)
    
    p_idx = np.random.RandomState(RANDOM_SEED).permutation(len(X))
    print(f"[Pairs] Created {len(X)} balanced pairs ({n} genuine, {n} forged) in {time.time()-t0:.2f}s")
    return X[p_idx], y[p_idx]


# ==================================================================================================
# MODULE 5: EVALUATION MODULE (METRICS & PLOTS)
# ==================================================================================================
def evaluate_model(verifier: SignatureVerifier, X_test: np.ndarray, y_test: np.ndarray, output_dir: str = OUTPUTS_DIR) -> Dict[str, Any]:
    """Computes Accuracy, Precision, Recall, F1, FAR, FRR, EER, and saves ROC & Confusion Matrix plots."""
    os.makedirs(output_dir, exist_ok=True)
    d_vals = verifier.model.decision_function(X_test)
    y_scores = 1.0 / (1.0 + np.exp(-d_vals))
    y_pred = (y_scores >= 0.5).astype(np.int32)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    
    # Biometric Error Rates
    far = float(fp / (fp + tn + 1e-10))  # Forgeries falsely accepted
    frr = float(fn / (fn + tp + 1e-10))  # Genuine falsely rejected
    
    fpr, tpr, thresh = roc_curve(y_test, y_scores, pos_label=1)
    roc_auc = float(auc(fpr, tpr))
    eer_idx = int(np.nanargmin(np.abs(fpr - (1.0 - tpr))))
    eer = float((fpr[eer_idx] + (1.0 - tpr[eer_idx])) / 2.0)
    
    # 1. Confusion Matrix Plot
    plt.figure(figsize=(5, 4))
    plt.imshow([[tn, fp], [fn, tp]], cmap=plt.cm.Blues)
    plt.title("Confusion Matrix", fontweight="bold")
    plt.xticks([0, 1], ["Forged", "Genuine"]); plt.yticks([0, 1], ["Forged", "Genuine"])
    for i in range(2):
        for j in range(2):
            v = [[tn, fp], [fn, tp]][i][j]
            plt.text(j, i, f"{v}", ha="center", va="center", color="white" if v > len(y_test)/4 else "black", fontweight="bold")
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150); plt.close()
    
    # 2. ROC Curve Plot
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, color="#2563eb", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    plt.scatter([fpr[eer_idx]], [tpr[eer_idx]], color="#dc2626", s=80, zorder=5, label=f"EER = {eer*100:.1f}%")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Acceptance Rate (FAR)"); plt.ylabel("True Acceptance Rate (1 - FRR)")
    plt.title("ROC Curve & Equal Error Rate (EER)", fontweight="bold")
    plt.legend(loc="lower right"); plt.grid(True, linestyle=":", alpha=0.6); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=150); plt.close()
    
    print("\n" + "="*55 + "\nEVALUATION RESULTS (Unseen Test Writers)\n" + "="*55)
    print(f"  Accuracy : {acc*100:.2f}%     |  Precision : {prec*100:.2f}%")
    print(f"  Recall   : {rec*100:.2f}%     |  F1-Score  : {f1*100:.2f}%")
    print(f"  FAR      : {far*100:.2f}%     |  FRR       : {frr*100:.2f}%")
    print(f"  EER      : {eer*100:.2f}%     |  ROC AUC   : {roc_auc:.4f}")
    print("="*55 + f"\nPlots saved to '{output_dir}/'\n")
    
    return {"accuracy": acc, "far": far, "frr": frr, "eer": eer, "auc": roc_auc}


# ==================================================================================================
# MODULE 6: VIDEO PROCESSING MODULE
# ==================================================================================================
def process_video(video_source: Union[str, int], verifier: Optional[SignatureVerifier] = None, ref_path: Optional[str] = None, output_dir: str = OUTPUTS_DIR) -> Dict[str, Any]:
    """Samples video frames, selects sharpest frame using Variance of Laplacian, crops ink, and verifies."""
    os.makedirs(output_dir, exist_ok=True)
    if isinstance(video_source, str) and video_source.isdigit(): video_source = int(video_source)
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened(): raise IOError(f"Could not open video source: {video_source}")
    
    best_frame, best_score, count = None, -1.0, 0
    while cap.isOpened() and count < 90:
        ret, frame = cap.read()
        if not ret: break
        count += 1
        if count % 2 != 0: continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score = float(cv2.Laplacian(gray, cv2.CV_64F).var())  # Sharpness metric
        if score > best_score:
            best_score, best_frame = score, frame.copy()
    cap.release()
    
    if best_frame is None: raise ValueError("No valid frame captured.")
    
    # Segment signature: Otsu thresholding + largest bounding contour
    gray = cv2.cvtColor(best_frame, cv2.COLOR_BGR2GRAY)
    _, bin_inv = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(bin_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hf, wf = best_frame.shape[:2]
    boxes = [cv2.boundingRect(c) for c in cnts if 200 < cv2.contourArea(c) < (hf * wf * 0.7)]
    
    if boxes:
        bx, by, bw, bh = max(boxes, key=lambda b: b[2] * b[3])
        crop = best_frame[max(0, by-10):min(hf, by+bh+10), max(0, bx-10):min(wf, bx+bw+10)]
    else:
        crop = best_frame[int(hf*0.25):int(hf*0.75), int(wf*0.25):int(wf*0.75)]
        
    sharp_p = os.path.join(output_dir, "video_sharpest_frame.jpg")
    crop_p = os.path.join(output_dir, "video_signature_crop.png")
    cv2.imwrite(sharp_p, best_frame); cv2.imwrite(crop_p, crop)
    print(f"[Video] Sharpest frame saved (Laplacian Var = {best_score:.1f}). Crop saved to '{crop_p}'")
    
    v_res = verifier.verify(ref_path, crop) if (ref_path and verifier and verifier.is_trained) else None
    return {"crop_path": crop_p, "sharp_score": best_score, "crop_img": crop, "result": v_res}


# ==================================================================================================
# MODULE 7: GUI (TKINTER INTERFACE)
# ==================================================================================================
class SimpleSignatureGUI:
    """Clean, beginner-friendly Tkinter GUI for Offline Signature Verification."""
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Handwritten Signature Forgery Detection (IVP Project)")
        self.root.geometry("1080x700")
        self.root.configure(bg="#f8fafc")
        
        self.verifier = SignatureVerifier(MODEL_PATH)
        self.dataset, self.dataset_layout = {}, "Not Loaded"
        self.ref_path, self.test_path = None, None
        self.photo_cache = []
        
        # 1. Header
        header = tk.Frame(root, bg="#1e293b", height=60)
        header.pack(fill="x")
        tk.Label(header, text="HANDWRITTEN SIGNATURE FORGERY DETECTION", fg="white", bg="#1e293b", font=("Segoe UI", 14, "bold")).pack(anchor="w", padx=20, pady=(8, 0))
        tk.Label(header, text="Image and Video Processing (IVP) • Offline Verification", fg="#94a3b8", bg="#1e293b", font=("Segoe UI", 9)).pack(anchor="w", padx=20, pady=(0, 8))
        
        # 2. Result Banner
        self.result_frame = tk.Frame(root, bg="#ffffff", relief="groove", bd=2, pady=8)
        self.result_frame.pack(fill="x", padx=15, pady=8)
        self.res_lbl = tk.Label(self.result_frame, text="READY TO VERIFY", fg="#475569", bg="#ffffff", font=("Segoe UI", 14, "bold"))
        self.res_lbl.pack()
        self.detail_lbl = tk.Label(self.result_frame, text="Load a reference and test signature, then click Verify.", fg="#64748b", bg="#ffffff", font=("Segoe UI", 9))
        self.detail_lbl.pack()
        
        # 3. Visual Stage Viewer (Left: Ref, Right: Test)
        vis_frame = tk.Frame(root, bg="#f8fafc")
        vis_frame.pack(fill="both", expand=True, padx=15)
        
        self.ref_box = tk.LabelFrame(vis_frame, text="Reference Signature (Genuine)", bg="#ffffff", font=("Segoe UI", 10, "bold"), padx=5, pady=5)
        self.ref_box.pack(side="left", fill="both", expand=True, padx=(0, 5))
        self.ref_labels = self._make_stage_labels(self.ref_box)
        
        self.test_box = tk.LabelFrame(vis_frame, text="Test Signature (Questioned)", bg="#ffffff", font=("Segoe UI", 10, "bold"), padx=5, pady=5)
        self.test_box.pack(side="right", fill="both", expand=True, padx=(5, 0))
        self.test_labels = self._make_stage_labels(self.test_box)
        
        # 4. Action Toolbar
        tb = tk.Frame(root, bg="#e2e8f0", pady=8, padx=15)
        tb.pack(fill="x", side="bottom")
        
        tk.Button(tb, text="✍️ Load Reference", command=self.on_load_ref, bg="#ffffff", font=("Segoe UI", 9, "bold")).pack(side="left", padx=4)
        tk.Button(tb, text="🔍 Load Test", command=self.on_load_test, bg="#ffffff", font=("Segoe UI", 9, "bold")).pack(side="left", padx=4)
        tk.Button(tb, text="✅ VERIFY", command=self.on_verify, bg="#2563eb", fg="white", font=("Segoe UI", 10, "bold"), padx=12).pack(side="left", padx=10)
        tk.Button(tb, text="📁 Load Dataset", command=self.on_load_dataset, bg="#ffffff").pack(side="left", padx=4)
        tk.Button(tb, text="⚙️ Train", command=self.on_train, bg="#ffffff").pack(side="left", padx=4)
        tk.Button(tb, text="📊 Evaluate", command=self.on_eval, bg="#ffffff").pack(side="left", padx=4)
        tk.Button(tb, text="🎥 Video", command=self.on_video, bg="#ffffff").pack(side="left", padx=4)
        
        self.status_lbl = tk.Label(tb, text="Status: Ready", bg="#e2e8f0", font=("Segoe UI", 9), anchor="e")
        self.status_lbl.pack(side="right", padx=10)

    def _make_stage_labels(self, parent: tk.Widget) -> Dict[str, tk.Label]:
        labels = {}
        stages = [("original", "Original"), ("binarized", "Binarized (Otsu)"), ("normalized", "Normalized (Padded)"), ("skeleton", "Skeleton (Thinning)")]
        grid_f = tk.Frame(parent, bg="#ffffff")
        grid_f.pack(fill="both", expand=True)
        for i, (k, title) in enumerate(stages):
            box = tk.Frame(grid_f, bg="#f8fafc", relief="solid", bd=1)
            box.grid(row=i//2, column=i%2, padx=4, pady=4, sticky="nsew")
            grid_f.grid_rowconfigure(i//2, weight=1); grid_f.grid_columnconfigure(i%2, weight=1)
            tk.Label(box, text=title, font=("Segoe UI", 8, "bold"), bg="#f8fafc", fg="#475569").pack(side="top", pady=2)
            lbl = tk.Label(box, text="[Empty]", bg="#f8fafc", fg="#94a3b8")
            lbl.pack(fill="both", expand=True, pady=4)
            labels[k] = lbl
        return labels

    def _show_stages(self, prep: Dict[str, Any], labels: Dict[str, tk.Label]):
        for k in ["original", "binarized", "normalized", "skeleton"]:
            img = prep[k]
            pil_img = Image.fromarray(img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            pil_img.thumbnail((180, 95), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(pil_img)
            self.photo_cache.append(photo)
            labels[k].configure(image=photo, text="")

    def on_load_ref(self):
        p = filedialog.askopenfilename(title="Select Reference Genuine Signature", filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
        if p:
            self.ref_path = p
            _, prep = extract_features(p)
            self._show_stages(prep, self.ref_labels)
            self.status_lbl.configure(text=f"Loaded Reference: {os.path.basename(p)}")

    def on_load_test(self):
        p = filedialog.askopenfilename(title="Select Test Signature", filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
        if p:
            self.test_path = p
            _, prep = extract_features(p)
            self._show_stages(prep, self.test_labels)
            self.status_lbl.configure(text=f"Loaded Test: {os.path.basename(p)}")

    def on_verify(self):
        if not self.ref_path or not self.test_path:
            messagebox.showwarning("Missing Files", "Please load both a Reference and a Test signature first.")
            return
        res = self.verifier.verify(self.ref_path, self.test_path)
        is_gen = res["is_genuine"]
        color = "#15803d" if is_gen else "#b91c1c"
        bg_col = "#dcfce7" if is_gen else "#fee2e2"
        self.result_frame.configure(bg=bg_col)
        self.res_lbl.configure(text=f"RESULT: {res['prediction']} ({res['confidence']*100:.1f}%)", fg=color, bg=bg_col)
        self.detail_lbl.configure(text=f"Similarity: {res['similarity_score']:.3f} | Cosine: {res['cosine_sim']:.3f} | SSIM: {res['ssim']:.3f} | {res['mode']}", bg=bg_col)
        self.status_lbl.configure(text=f"Verification: {res['prediction']}")

    def on_load_dataset(self):
        target = filedialog.askopenfilename(title="Select Dataset (zip or folder)", filetypes=[("Zip Archives", "*.zip"), ("All", "*.*")]) or filedialog.askdirectory()
        if not target: return
        def run():
            root_dir = extract_dataset(target) if target.lower().endswith(".zip") else target
            self.dataset, self.dataset_layout = load_dataset(root_dir)
            self.root.after(0, lambda: self.status_lbl.configure(text=f"Dataset: {len(self.dataset)} writers ({self.dataset_layout})"))
            self.root.after(0, lambda: messagebox.showinfo("Dataset Loaded", f"Loaded {len(self.dataset)} writers."))
        threading.Thread(target=run, daemon=True).start()

    def on_train(self):
        if not self.dataset:
            target = DATASET_PATH if os.path.exists(DATASET_PATH) else generate_synthetic_dataset()
            self.dataset, _ = load_dataset(extract_dataset(target, max_writers=20) if target.endswith(".zip") else target)
        def run():
            writers = sorted(list(self.dataset.keys()))
            n_tr = max(1, int(len(writers) * 0.8))
            X_tr, y_tr = build_pairs(self.dataset, writers[:n_tr], max_pairs=3000)
            acc = self.verifier.train(X_tr, y_tr)
            self.verifier.save(MODEL_PATH)
            self.root.after(0, lambda: messagebox.showinfo("Trained", f"Trained SVM Model. Train Accuracy: {acc*100:.1f}%"))
            self.root.after(0, lambda: self.status_lbl.configure(text=f"Model Ready (Acc: {acc*100:.1f}%)"))
        threading.Thread(target=run, daemon=True).start()

    def on_eval(self):
        if not self.verifier.is_trained:
            messagebox.showwarning("Not Trained", "Please train the model first.")
            return
        def run():
            writers = sorted(list(self.dataset.keys()))
            n_tr = max(1, int(len(writers) * 0.8))
            X_te, y_te = build_pairs(self.dataset, writers[n_tr:] if len(writers) > 1 else writers, max_pairs=1500)
            m = evaluate_model(self.verifier, X_te, y_te, OUTPUTS_DIR)
            msg = f"Evaluation (Unseen Writers):\nAccuracy: {m['accuracy']*100:.1f}%\nFAR: {m['far']*100:.1f}%\nFRR: {m['far']*100:.1f}%\nEER: {m['eer']*100:.1f}%\nPlots saved to 'outputs/'"
            self.root.after(0, lambda: messagebox.showinfo("Evaluation Results", msg))
        threading.Thread(target=run, daemon=True).start()

    def on_video(self):
        v = filedialog.askopenfilename(title="Select Video File (or Cancel for Webcam)", filetypes=[("Videos", "*.mp4 *.avi")])
        source = v if v else 0
        def run():
            res = process_video(source, self.verifier, self.ref_path, OUTPUTS_DIR)
            self.test_path = res["crop_path"]
            _, prep = extract_features(res["crop_img"])
            self.root.after(0, lambda: self._show_stages(prep, self.test_labels))
            msg = f"Sharp frame captured (Variance: {res['sharp_score']:.1f})."
            if res["result"]: msg += f"\nVerification: {res['result']['prediction']} ({res['result']['confidence']*100:.1f}%)"
            self.root.after(0, lambda: messagebox.showinfo("Video Processed", msg))
        threading.Thread(target=run, daemon=True).start()


# ==================================================================================================
# MODULE 8: MAIN & CLI PARSING
# ==================================================================================================
def main():
    parser = argparse.ArgumentParser(description="Handwritten Signature Forgery Detection (IVP Project)")
    parser.add_argument("--gui", action="store_true", help="Launch GUI interface")
    parser.add_argument("--dataset", type=str, default=None, help="Path to dataset zip or folder")
    parser.add_argument("--train", action="store_true", help="Train Writer-Independent SVM")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model and output plots")
    parser.add_argument("--verify", nargs=2, metavar=("REF", "TEST"), help="Verify test against reference")
    parser.add_argument("--video", type=str, default=None, help="Process video file or webcam (0)")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic dataset generation")
    parser.add_argument("--max-writers", type=int, default=None, help="Limit number of writers")
    args = parser.parse_args()

    cli_active = any([args.train, args.evaluate, args.verify, args.video, args.synthetic])
    if args.gui or not cli_active:
        root = tk.Tk()
        SimpleSignatureGUI(root)
        root.mainloop()
        return

    verifier = SignatureVerifier(MODEL_PATH)
    dataset = None

    if args.synthetic:
        s_dir = generate_synthetic_dataset(num_writers=args.max_writers or 8)
        dataset, _ = load_dataset(s_dir)
    elif args.dataset or args.train or args.evaluate:
        dpath = args.dataset or DATASET_PATH
        root_dir = extract_dataset(dpath, max_writers=args.max_writers) if (os.path.isfile(dpath) and dpath.endswith(".zip")) else dpath
        if os.path.exists(root_dir):
            dataset, _ = load_dataset(root_dir, max_writers=args.max_writers)
        else:
            s_dir = generate_synthetic_dataset(num_writers=args.max_writers or 8)
            dataset, _ = load_dataset(s_dir)

    if args.train and dataset:
        writers = sorted(list(dataset.keys()))
        n_tr = max(1, int(len(writers) * 0.8))
        X_tr, y_tr = build_pairs(dataset, writers[:n_tr], max_pairs=3000)
        verifier.train(X_tr, y_tr)
        verifier.save(MODEL_PATH)

    if args.evaluate and dataset:
        if not verifier.is_trained: verifier.load(MODEL_PATH)
        writers = sorted(list(dataset.keys()))
        n_tr = max(1, int(len(writers) * 0.8))
        X_te, y_te = build_pairs(dataset, writers[n_tr:] if len(writers) > 1 else writers, max_pairs=1500)
        evaluate_model(verifier, X_te, y_te, OUTPUTS_DIR)

    if args.verify:
        r_p, t_p = args.verify
        res = verifier.verify(r_p, t_p)
        print(f"\nResult: {res['prediction']} (Confidence: {res['confidence']*100:.2f}%)")
        print(f"Similarity: {res['similarity_score']:.4f} | Cosine: {res['cosine_sim']:.4f} | SSIM: {res['ssim']:.4f} | {res['mode']}\n")

    if args.video:
        ref_p = args.verify[0] if args.verify else None
        process_video(args.video, verifier, ref_p, OUTPUTS_DIR)


if __name__ == "__main__":
    main()
