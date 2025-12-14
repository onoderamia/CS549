import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.append('gsv-cities')
from main import VPRModel

DEVICE = "xpu" if torch.xpu.is_available() else "cpu"
DEVICE = "cuda" if torch.cuda.is_available() else DEVICE

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])

CITIES = None
DATA_DIR = None

def load_cities(root="../data"):
    global CITIES, DATA_DIR
    CITIES = sorted([city for city in os.listdir(root) if os.path.isdir(os.path.join(root, city))])
    DATA_DIR = root
    return CITIES

def get_city_to_id():
    return {c: i for i, c in enumerate(CITIES)}

def load_dataset(root="../data"):
    global CITIES, DATA_DIR
    if CITIES is None or DATA_DIR is None:
        load_cities(root)
        
    image_paths = []
    labels = []

    for city in CITIES:
        city_dir = os.path.join(DATA_DIR, city)
        if not os.path.isdir(city_dir):
            continue
        for fname in sorted(os.listdir(city_dir)):
            if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                image_paths.append(os.path.join(city_dir, fname))
                labels.append(city)

    print(f"Loaded {len(image_paths)} images from {len(set(labels))} cities.")
    return image_paths, labels

def load_model(checkpoint_path=None):
    backbone = VPRModel(
        backbone_arch='resnet50',
        layers_to_crop=[],
        agg_arch='ConvAP',
        agg_config={
            'in_channels': 2048,
            'out_channels': 128,
            's1': 2,
            's2': 2
        },
    )
    
    # classification head
    model = torch.nn.Sequential(
        backbone,
        torch.nn.Linear(128*2*2, len(CITIES))
    ).to(DEVICE)
    
    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(state)
    
    return model

def get_logits(model, image_paths, batch_size=16, progress=True):
    model.eval()
    all_logits = []

    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Classifying", disable=not progress):
            batch_paths = image_paths[i:i + batch_size]
            images = [transform(Image.open(p).convert("RGB")).to(DEVICE) for p in batch_paths]
            images = torch.stack(images)
            logits = model(images)
            all_logits.append(logits.cpu().numpy())

    return np.concatenate(all_logits, axis=0)

def get_predictions(model, image_paths, batch_size=16, progress=True):
    logits = get_logits(model, image_paths, batch_size, progress)
    return np.argmax(logits, axis=1)

def classify_image(model, image_path):
    logits = get_logits(model, [image_path], 1, False)[0]
    probs = np.exp(logits) / np.sum(np.exp(logits))
    sorted_indices = probs.argsort()[::-1]
    preds = []
    for idx in sorted_indices:
        city = CITIES[idx]
        confidence = probs[idx]
        preds.append((city, confidence))
    return preds