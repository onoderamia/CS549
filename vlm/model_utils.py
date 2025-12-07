import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import CLIPVisionModel, CLIPProcessor

import torch.nn as nn
import torch.nn.functional as F

DEVICE = "xpu" if torch.xpu.is_available() else "cpu"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
SIGLIP_MODEL_NAME = "google/siglip-base-patch16-224"

CITIES = None
DATA_DIR = None


def load_cities(root="../out"):
    global CITIES, DATA_DIR
    CITIES = sorted([city for city in os.listdir(root) if os.path.isdir(os.path.join(root, city))])
    DATA_DIR = root
    return CITIES


def get_city_to_id():
    return {c: i for i, c in enumerate(CITIES)}

def load_dataset(root="../out"):
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

class CLIPCityClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.clip = CLIPVisionModel.from_pretrained(CLIP_MODEL_NAME)
        self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME, use_fast=True)
        
        hidden_size = self.clip.config.hidden_size
        
        # classification head
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, images):
        pixel_values = self.processor(images=images, return_tensors="pt")['pixel_values'].to(DEVICE)
        outputs = self.clip(pixel_values)
        image_features = outputs.pooler_output
        return self.classifier(image_features)
    
    def get_probabilities(self, pixel_values):
        logits = self.forward(pixel_values)
        return F.softmax(logits, dim=-1)

def load_model(checkpoint_path=None):
    num_classes = len(CITIES)
    model = CLIPCityClassifier(num_classes)
    
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
    
    return model.to(DEVICE)

def get_logits(model, image_paths, batch_size=16, progress=True):
    model.eval()
    all_logits = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Classifying", disable=not progress):
            batch_paths = image_paths[i:i + batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
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
