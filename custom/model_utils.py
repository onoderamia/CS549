import numpy as np
import torch
from tqdm import tqdm
import cv2

from train import *

CITIES = None

def load_cities(root="../out"):
    global CITIES
    CITIES = sorted([city for city in os.listdir(root) if os.path.isdir(os.path.join(root, city))])
    return CITIES

def load_model(checkpoint_path=None):
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        num_classes = len(checkpoint["classes"])
        num_scene_features = checkpoint["num_scene_features"]
        
        global CITIES
        CITIES = checkpoint["classes"]
        
        model = GeoSceneNet(num_scene_features, num_classes).to(DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])

        return model.to(DEVICE)

    return None


def get_logits(model, image_paths, batch_size=16, progress=True):
    model.eval()
    all_logits = []

    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Classifying", disable=not progress):
            batch_paths = image_paths[i:i + batch_size]
            
            images = []
            scene_feats = []
            for p in batch_paths:
                img_bgr = cv2.imread(p)
                if img_bgr is None:
                    img_bgr = np.zeros((256, 256, 3), dtype=np.uint8)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img_tensor = eval_transform(img_rgb)
                images.append(img_tensor)
                
                scene_feat = extract_scene_features(p)
                scene_feats.append(torch.from_numpy(scene_feat))
            
            images = torch.stack(images).to(DEVICE)
            scene_feats = torch.stack(scene_feats).to(DEVICE)
            
            logits = model(images, scene_feats)
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
