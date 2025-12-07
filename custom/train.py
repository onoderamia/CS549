import os
import numpy as np
import cv2

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision import models

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------

DATA_ROOT = "../out"
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

BATCH_SIZE = 16
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("xpu") if torch.xpu.is_available() else DEVICE
print("Using device:", DEVICE)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = T.Compose([
    T.ToPILImage(),
    T.RandomResizedCrop(224, scale=(0.8, 1.0)),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.2),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

eval_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((256, 256)),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# -------------------------------------------------------------------
# CV Features
# -------------------------------------------------------------------

def feature_texture(gray):
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return min(lap.var(), 5000.0)


def feature_edges(gray):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    magx = np.mean(np.abs(sobelx))
    magy = np.mean(np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)))
    return magx / (magx + magy + 1e-8)


def feature_color(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    sky_mask = (v > 180) & (s < 60)
    return float(h.mean()), float(s.mean()), float(v.mean()), float(v.std()), float(sky_mask.mean())


def feature_horizon(gray):
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    row_grad = np.mean(np.abs(sobely), axis=1)
    horizon_row = int(np.argmax(row_grad))
    return horizon_row / gray.shape[0]


def feature_keypoints(gray):
    orb = cv2.ORB_create(nfeatures=500)
    keypoints = orb.detect(gray, None)
    h, w = gray.shape
    return len(keypoints) / (h * w)


def feature_edge_density(gray):
    edges = cv2.Canny(gray, 100, 200)
    return float((edges > 0).mean())


def extract_scene_features(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image at {path}")

    img = cv2.resize(img, (256, 256))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    texture    = feature_texture(gray)
    edge_ratio = feature_edges(gray)
    h_mean, s_mean, v_mean, v_std, sky_ratio = feature_color(img)
    horizon    = feature_horizon(gray)
    kp_density = feature_keypoints(gray)
    ed_density = feature_edge_density(gray)

    scene10 = np.array([
        texture,
        edge_ratio,
        h_mean,
        s_mean,
        v_mean,
        v_std,
        horizon,
        kp_density,
        ed_density,
        sky_ratio
    ], dtype=np.float32)

    return scene10


def build_scene_dict(paths):
    return {p: extract_scene_features(p) for p in paths}

# -------------------------------------------------------------------
# Data utilities
# -------------------------------------------------------------------

def list_image_paths_and_labels(root):
    paths, labels = [], []
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
        stratify=labels,
        random_state=RANDOM_SEED
    )

    p_val, p_test, y_val, y_test = train_test_split(
        p_temp, y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=RANDOM_SEED
    )

    print("Image split counts:")
    print("  Train:", len(p_train))
    print("  Val  :", len(p_val))
    print("  Test :", len(p_test))
    return p_train, p_val, p_test, y_train, y_val, y_test

# -------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------

class SceneDataset(Dataset):
    def __init__(self, paths, labels_int, scene_dict, transform):
        self.paths = list(paths)
        self.labels_int = np.array(labels_int, dtype=np.int64)
        self.scene_dict = scene_dict
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels_int[idx]

        img_bgr = cv2.imread(path)
        if img_bgr is None:
            img_bgr = np.zeros((256, 256, 3), dtype=np.uint8)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(img_rgb)

        scene_feats = torch.from_numpy(self.scene_dict[path].astype(np.float32))
        label_tensor = torch.tensor(label, dtype=torch.long)

        return img_tensor, scene_feats, label_tensor

# -------------------------------------------------------------------
# Model
# -------------------------------------------------------------------

class GeoSceneNet(nn.Module):
    def __init__(self, num_scene_features, num_classes):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT
        self.cnn = models.resnet18(weights=weights)
        in_feats = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity()
        self.fc = nn.Sequential(
            nn.Linear(in_feats + num_scene_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x_img, x_scene):
        cnn_feats = self.cnn(x_img)
        x = torch.cat([cnn_feats, x_scene], dim=1)
        return self.fc(x)

# -------------------------------------------------------------------
# Train / Eval loops
# -------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for imgs, scene_feats, labels in loader:
        imgs = imgs.to(device)
        scene_feats = scene_feats.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs, scene_feats)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    labels_all = []
    preds_all = []

    with torch.no_grad():
        for imgs, scene_feats, labels in loader:
            imgs = imgs.to(device)
            scene_feats = scene_feats.to(device)
            labels = labels.to(device)

            outputs = model(imgs, scene_feats)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

            labels_all.append(labels.cpu().numpy())
            preds_all.append(preds.cpu().numpy())

    return total_loss / total, correct / total, np.concatenate(labels_all), np.concatenate(preds_all)

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    # 1. List & split
    paths, labels_str = list_image_paths_and_labels(DATA_ROOT)
    p_train, p_val, p_test, y_train_str, y_val_str, y_test_str = split_paths(paths, labels_str)

    # 2. Scene features
    print("\nExtracting scene features...")
    scene_dict = build_scene_dict(paths)
    num_scene_features = len(next(iter(scene_dict.values())))
    print("Scene feature dimension:", num_scene_features)

    # 3. Label encoding
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_str)
    y_val = le.transform(y_val_str)
    y_test = le.transform(y_test_str)
    num_classes = len(le.classes_)
    print("Classes:", list(le.classes_))

    # 4. Datasets / Loaders
    train_ds = SceneDataset(p_train, y_train, scene_dict, train_transform)
    val_ds   = SceneDataset(p_val, y_val, scene_dict, eval_transform)
    test_ds  = SceneDataset(p_test, y_test, scene_dict, eval_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 5. Model / Optim
    model = GeoSceneNet(num_scene_features, num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # 6. Train
    best_acc = 0.0
    best_state = None

    for epoch in range(1, NUM_EPOCHS + 1):
        loss_tr, acc_tr = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        loss_val, acc_val, _, _ = eval_one_epoch(model, val_loader, criterion, DEVICE)

        print(f"[Epoch {epoch:02d}] Train loss: {loss_tr:.4f}, acc: {acc_tr:.4f} | "
              f"Val loss: {loss_val:.4f}, acc: {acc_val:.4f}")

        if acc_val > best_acc:
            best_acc = acc_val
            best_state = model.state_dict()

    # 7. Load best and evaluate on test
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nLoaded best model with val acc = {best_acc:.4f}")

    loss_test, acc_test, y_true, y_pred = eval_one_epoch(model, test_loader, criterion, DEVICE)

    print("\n=== TEST RESULTS (Scene-Fusion Model) ===")
    print(f"Test loss: {loss_test:.4f}, Test acc: {acc_test:.4f}")

    y_true_str = le.inverse_transform(y_true)
    y_pred_str = le.inverse_transform(y_pred)

    print("\nClassification report:")
    print(classification_report(y_true_str, y_pred_str))

    print("Confusion matrix:")
    print(confusion_matrix(y_true_str, y_pred_str))

    # 8. Save model
    save_path = "../models/custom.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": le.classes_.tolist(),
            "num_scene_features": num_scene_features,
        },
        save_path,
    )
    print(f"\nSaved model to {save_path}")


if __name__ == "__main__":
    main()
