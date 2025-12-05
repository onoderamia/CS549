import os
import numpy as np
import cv2

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

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

# ==============================
# 1. CONFIG
# ==============================
DATA_ROOT = "/Users/miaonodera/Desktop/UIUC/FALL2025/ECE549/CV/out"
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

USE_OCR = True    # set False if OCR is too slow
BATCH_SIZE = 16
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# ImageNet normalization for ResNet18
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Data augmentation for training CNN features
train_transform = T.Compose([
    T.ToPILImage(),
    T.RandomResizedCrop(224, scale=(0.8, 1.0)),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.2),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Deterministic transform for val/test CNN features
eval_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((256, 256)),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


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
    sobely = cv2.CV_64F
    sobely = cv2.Sobel(gray, sobely, 0, 1, ksize=3)
    magx = np.mean(np.abs(sobelx))
    magy = np.mean(np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)))
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


def build_handcrafted_dict(paths):
    """
    Build a dict: image_path -> handcrafted feature vector.
    Assumes all images are valid; will raise if any fails.
    """
    feat_dict = {}
    for p in paths:
        feats = extract_handcrafted_features(p)
        feat_dict[p] = feats
    return feat_dict


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
# 6. HYBRID DATASET (IMAGE + HANDCRAFTED)
# ==============================

class HybridDataset(Dataset):
    def __init__(self, paths, labels_int, hc_features_dict, transform):
        self.paths = list(paths)
        self.labels_int = np.array(labels_int, dtype=np.int64)
        self.hc_features_dict = hc_features_dict
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels_int[idx]

        img_bgr = cv2.imread(path)
        if img_bgr is None:
            # Fallback: black image
            img_bgr = np.zeros((256, 256, 3), dtype=np.uint8)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        img_tensor = self.transform(img_rgb)

        hc_feats = self.hc_features_dict[path]  # numpy array (15,)
        hc_feats = torch.from_numpy(hc_feats.astype(np.float32))

        label_tensor = torch.tensor(label, dtype=torch.long)
        return img_tensor, hc_feats, label_tensor


# ==============================
# 7. HYBRID MODEL (ResNet18 + Handcrafted)
# ==============================

class HybridNet(nn.Module):
    def __init__(self, num_hc_features, num_classes):
        super().__init__()
        # Pretrained ResNet-18 backbone
        weights = models.ResNet18_Weights.DEFAULT
        self.cnn = models.resnet18(weights=weights)
        in_feats = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity()  # output is (B, in_feats)

        # Classifier that takes [cnn_feats || hc_feats]
        self.fc = nn.Sequential(
            nn.Linear(in_feats + num_hc_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x_img, x_hc):
        cnn_feats = self.cnn(x_img)         # (B, 512)
        x = torch.cat([cnn_feats, x_hc], dim=1)  # (B, 512 + num_hc_features)
        out = self.fc(x)
        return out


# ==============================
# 8. TRAINING / EVAL LOOPS
# ==============================

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for imgs, hc_feats, labels in loader:
        imgs = imgs.to(device)
        hc_feats = hc_feats.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs, hc_feats)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for imgs, hc_feats, labels in loader:
            imgs = imgs.to(device)
            hc_feats = hc_feats.to(device)
            labels = labels.to(device)

            outputs = model(imgs, hc_feats)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    return epoch_loss, epoch_acc, all_labels, all_preds


# ==============================
# 9. MAIN
# ==============================

def main():
    # 1) List images + labels and make splits
    paths, labels_str = list_image_paths_and_labels(DATA_ROOT)
    p_train, p_val, p_test, y_train_str, y_val_str, y_test_str = split_paths(paths, labels_str)

    # 2) Build handcrafted feature dict for ALL images (so it's shared)
    print("\nComputing handcrafted+language features for all images...")
    hc_dict = build_handcrafted_dict(paths)
    num_hc_features = len(next(iter(hc_dict.values())))
    print("Handcrafted feature dimension:", num_hc_features)

    # 3) Encode labels to ints
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_str)
    y_val = le.transform(y_val_str)
    y_test = le.transform(y_test_str)
    num_classes = len(le.classes_)
    print("Classes:", list(le.classes_))

    # 4) Build datasets + loaders
    train_dataset = HybridDataset(p_train, y_train, hc_dict, train_transform)
    val_dataset   = HybridDataset(p_val,   y_val,   hc_dict, eval_transform)
    test_dataset  = HybridDataset(p_test,  y_test,  hc_dict, eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 5) Model, loss, optimizer
    model = HybridNet(num_hc_features=num_hc_features, num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_val_acc = 0.0
    best_state = None

    # 6) Training loop
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_acc, _, _ = eval_one_epoch(model, val_loader, criterion, DEVICE)

        print(f"[Epoch {epoch:02d}] "
              f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
              f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

    # 7) Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nLoaded best model with val acc = {best_val_acc:.4f}")

    # 8) Evaluate on test set
    test_loss, test_acc, y_true, y_pred = eval_one_epoch(model, test_loader, criterion, DEVICE)
    print("\n=== TEST RESULTS (Hybrid Deep + Handcrafted) ===")
    print(f"Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")

    # Decode labels to city names
    y_true_str = le.inverse_transform(y_true)
    y_pred_str = le.inverse_transform(y_pred)

    print("\nClassification report:")
    print(classification_report(y_true_str, y_pred_str))
    print("Confusion matrix:")
    print(confusion_matrix(y_true_str, y_pred_str))

    # 9) Save model + label classes
    save_path = "hybrid_cnn_handcrafted.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": le.classes_.tolist(),
            "num_hc_features": num_hc_features,
        },
        save_path,
    )
    print(f"\nSaved model to {save_path}")


if __name__ == "__main__":
    main()
