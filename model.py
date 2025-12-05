import os
import numpy as np
import cv2

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

import xgboost as xgb

# Optional OCR
try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision import models

# Limit thread usage (helps avoid segfault/weirdness on macOS)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

# ==============================
# 1. CONFIG
# ==============================
DATA_ROOT = "/Users/miaonodera/Desktop/UIUC/FALL2025/ECE549/CV/out"
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

USE_OCR = True    # set False if OCR is too slow
BATCH_SIZE_CNN = 16


# ==============================
# 2. HANDCRAFTED FEATURE FUNCTIONS
# ==============================

def compute_texture_feature(gray):
    """Texture: variance of Laplacian (clipped)."""
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    texture = lap.var()
    texture = min(texture, 5000.0)
    return texture


def compute_edge_composition(gray):
    """
    Edge composition:
      vertical edge ratio from Sobel_x and Sobel_y
    """
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magx = np.mean(np.abs(sobelx))
    magy = np.mean(np.abs(sobely))
    vert_edge_ratio = magx / (magx + magy + 1e-8)
    return vert_edge_ratio


def compute_color_and_brightness_features(img_bgr):
    """
    Color + brightness:
      - mean hue, mean saturation
      - mean and std of brightness (V channel)
      - sky ratio (bright + low-saturation pixels)
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    hue_mean = float(h.mean())
    sat_mean = float(s.mean())
    v_mean = float(v.mean())
    v_std = float(v.std())

    sky_mask = (v > 180) & (s < 60)
    sky_ratio = float(sky_mask.mean())

    return hue_mean, sat_mean, v_mean, v_std, sky_ratio


def compute_horizon_feature(gray):
    """
    Horizon/layout:
      row with largest vertical gradient magnitude (normalized).
    """
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    row_grad = np.mean(np.abs(sobely), axis=1)
    horizon_row = int(np.argmax(row_grad))
    horizon_height = horizon_row / gray.shape[0]
    return horizon_height


def compute_keypoint_density(gray):
    """Keypoint density using ORB."""
    orb = cv2.ORB_create(nfeatures=500)
    keypoints = orb.detect(gray, None)
    h, w = gray.shape
    density = len(keypoints) / (h * w)
    return density


def compute_edge_density(gray):
    """Edge density: fraction of pixels that are Canny edges."""
    edges = cv2.Canny(gray, 100, 200)
    edge_density = (edges > 0).mean()
    return float(edge_density)


# ==============================
# 3. LANGUAGE / SCRIPT FEATURES
# ==============================

def detect_script_from_text(text):
    """
    Rough script detector based on unicode ranges.
    Returns 5-dim: [latin, devanagari, cjk, arabic, has_text].
    """
    if not text or text.strip() == "":
        return np.array([0, 0, 0, 0, 0], dtype=np.float32)

    has_latin = False
    has_deva = False
    has_cjk = False
    has_arabic = False

    for ch in text:
        code = ord(ch)
        # Latin
        if (0x0041 <= code <= 0x007A) or (0x00C0 <= code <= 0x024F):
            has_latin = True
        # Devanagari
        if 0x0900 <= code <= 0x097F:
            has_deva = True
        # CJK (Han + Hiragana + Katakana)
        if (0x4E00 <= code <= 0x9FFF) or (0x3040 <= code <= 0x30FF):
            has_cjk = True
        # Arabic
        if 0x0600 <= code <= 0x06FF:
            has_arabic = True

    arr = np.array([
        1.0 if has_latin else 0.0,
        1.0 if has_deva else 0.0,
        1.0 if has_cjk else 0.0,
        1.0 if has_arabic else 0.0,
        1.0,  # has_text flag
    ], dtype=np.float32)
    return arr


def compute_language_features(img_bgr):
    """
    OCR-based language/script indicator.
    If OCR not available or USE_OCR=False, returns all zeros.
    """
    if not OCR_AVAILABLE or not USE_OCR:
        return np.zeros(5, dtype=np.float32)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    try:
        text = pytesseract.image_to_string(img_rgb, config="--psm 6")
    except Exception:
        return np.zeros(5, dtype=np.float32)

    return detect_script_from_text(text)


# ==============================
# 4. HANDCRAFTED+LANG FEATURES PER IMAGE
# ==============================

def extract_handcrafted_features(image_path):
    """
    Compute handcrafted + language features for one image.
    Returns a 15-dim numpy array.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")

    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    texture = compute_texture_feature(gray)
    vert_edge_ratio = compute_edge_composition(gray)
    hue_mean, sat_mean, v_mean, v_std, sky_ratio = compute_color_and_brightness_features(img)
    horizon_height = compute_horizon_feature(gray)
    keypoint_density = compute_keypoint_density(gray)
    edge_density = compute_edge_density(gray)
    lang_feats = compute_language_features(img)  # 5 dims

    # 10 original + 5 language = 15 features
    feats_10 = np.array([
        texture,
        vert_edge_ratio,
        hue_mean,
        sat_mean,
        v_mean,
        v_std,
        horizon_height,
        keypoint_density,
        edge_density,
        sky_ratio,
    ], dtype=np.float32)

    features = np.concatenate([feats_10, lang_feats], axis=0)
    return features


def build_handcrafted_matrix(paths):
    X = []
    for p in paths:
        try:
            feats = extract_handcrafted_features(p)
            X.append(feats)
        except Exception as e:
            print(f"  [HC] Skipping {p}: {e}")
    return np.vstack(X)


# ==============================
# 5. LIST IMAGE PATHS + LABELS
# ==============================

def list_image_paths_and_labels(root):
    paths = []
    labels = []
    for city_name in sorted(os.listdir(root)):
        city_folder = os.path.join(root, city_name)
        if not os.path.isdir(city_folder):
            continue
        for fname in os.listdir(city_folder):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(city_folder, fname))
                labels.append(city_name)
    paths = np.array(paths)
    labels = np.array(labels)
    print(f"Found {len(paths)} images across {len(np.unique(labels))} cities.")
    return paths, labels


def split_paths(paths, labels):
    p_train, p_temp, y_train, y_temp = train_test_split(
        paths, labels,
        test_size=0.4,
        random_state=RANDOM_SEED,
        stratify=labels
    )
    p_val, p_test, y_val, y_test = train_test_split(
        p_temp, y_temp,
        test_size=0.5,
        random_state=RANDOM_SEED,
        stratify=y_temp
    )

    print("Image split counts:")
    print("  Train:", len(p_train))
    print("  Val  :", len(p_val))
    print("  Test :", len(p_test))

    return p_train, p_val, p_test, y_train, y_val, y_test


# ==============================
# 6. CNN FEATURE EXTRACTOR (ResNet18)
# ==============================

class ImagePathDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = list(paths)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((256, 256, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            img = self.transform(img)
        return img, idx  # idx so we can put features in correct position


def build_resnet18_feature_extractor(device):
    # Pretrained ResNet-18
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    # Replace final fc with identity to get 512-dim embedding
    model.fc = nn.Identity()
    model = model.to(device)
    model.eval()
    return model


def extract_cnn_features(paths, model, transform, device):
    """
    paths: numpy array of image paths
    Returns: [N, 512] CNN embedding matrix
    """
    dataset = ImagePathDataset(paths, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE_CNN, shuffle=False, num_workers=0)

    N = len(paths)
    feats = np.zeros((N, 512), dtype=np.float32)

    with torch.no_grad():
        for imgs, idxs in loader:
            imgs = imgs.to(device)
            emb = model(imgs)  # [B, 512]
            emb = emb.cpu().numpy().astype(np.float32)
            for i, idx in enumerate(idxs):
                feats[idx] = emb[i]

    return feats


# ==============================
# 7. XGBOOST ON HYBRID FEATURES
# ==============================

def train_xgb_hybrid(X_train, y_train, X_val, y_val, X_test, y_test, label_encoder, num_classes):
    print("\n=== XGBoost on hybrid features ===")

    xgb_clf = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softmax",
        num_class=num_classes,
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    # Train with validation
    xgb_clf.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Retrain on train+val
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])
    xgb_clf.fit(X_trainval, y_trainval, verbose=False)

    # Test
    y_test_pred = xgb_clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    y_test_pred_str = label_encoder.inverse_transform(y_test_pred)
    y_test_str = label_encoder.inverse_transform(y_test)

    print("\n=== XGBOOST HYBRID TEST RESULTS ===")
    print(f"XGBoost hybrid test accuracy: {test_acc:.4f}\n")
    print("Classification report (XGBoost hybrid):")
    print(classification_report(y_test_str, y_test_pred_str))
    print("Confusion matrix (XGBoost hybrid):")
    print(confusion_matrix(y_test_str, y_test_pred_str))

    return xgb_clf


# ==============================
# 8. MAIN
# ==============================

def main():
    # 1) List images + labels and make splits
    paths, labels_str = list_image_paths_and_labels(DATA_ROOT)
    p_train, p_val, p_test, y_train_str, y_val_str, y_test_str = split_paths(paths, labels_str)

    # Encode labels to ints for XGB
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_str)
    y_val = le.transform(y_val_str)
    y_test = le.transform(y_test_str)
    num_classes = len(le.classes_)
    print("Classes:", list(le.classes_))

    # 2) Handcrafted+language features
    print("\nExtracting handcrafted+language features...")
    X_train_hc = build_handcrafted_matrix(p_train)
    X_val_hc = build_handcrafted_matrix(p_val)
    X_test_hc = build_handcrafted_matrix(p_test)

    assert X_train_hc.shape[0] == len(p_train)
    assert X_val_hc.shape[0] == len(p_val)
    assert X_test_hc.shape[0] == len(p_test)

    # 3) CNN features
    print("\nExtracting CNN features (ResNet18)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = build_resnet18_feature_extractor(device)

    cnn_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((256, 256)),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    X_train_cnn = extract_cnn_features(p_train, resnet, cnn_transform, device)
    X_val_cnn   = extract_cnn_features(p_val,   resnet, cnn_transform, device)
    X_test_cnn  = extract_cnn_features(p_test,  resnet, cnn_transform, device)

    # 4) Hybrid features = [handcrafted+language, cnn]
    X_train_hybrid = np.concatenate([X_train_hc, X_train_cnn], axis=1)
    X_val_hybrid   = np.concatenate([X_val_hc,   X_val_cnn],   axis=1)
    X_test_hybrid  = np.concatenate([X_test_hc,  X_test_cnn],  axis=1)

    print("\nHybrid feature dims:")
    print("  Train:", X_train_hybrid.shape)
    print("  Val  :", X_val_hybrid.shape)
    print("  Test :", X_test_hybrid.shape)

    # 5) Train + evaluate XGBoost on hybrid
    _ = train_xgb_hybrid(
        X_train_hybrid, y_train,
        X_val_hybrid, y_val,
        X_test_hybrid, y_test,
        le,
        num_classes
    )


if __name__ == "__main__":
    main()
