"""
Training script for fine-tuning CLIP on city classification.
Uses HuggingFace's CLIP model with a custom classification head.
"""

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
import sys
from random import shuffle
from model_utils import *

RANDOM_VARIATION = True

train_transform = T.Compose([
    T.RandomResizedCrop(224, scale=(0.8, 1.0)),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.2),
])

def train_epoch(model, image_paths, labels, optimizer, img_transform=None, batch_size=16):
    model.train()
    total_loss = 0
    num_correct = 0
    num_samples = 0

    loss_fn = torch.nn.CrossEntropyLoss()

    data = list(zip(image_paths, labels))
    shuffle(data)

    num_batches = len(data) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Training"):
        batch = data[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        paths, label_ids = zip(*batch)

        def load_img(path):
            img = Image.open(path).convert("RGB")
            if img_transform is not None:
                img = img_transform(img)
            return img

        images = [load_img(p) for p in paths]
        targets = torch.tensor(label_ids, dtype=torch.long, device=DEVICE)

        logits = model(images)

        loss = loss_fn(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = logits.argmax(dim=1)
        num_correct += (preds == targets).sum().item()
        num_samples += len(targets)

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    accuracy = num_correct / num_samples if num_samples > 0 else 0
    return avg_loss, accuracy

def evaluate(model, test_paths, test_labels, batch_size=16):
    preds = get_predictions(model, test_paths, batch_size)
    return np.mean(preds == np.array(test_labels))

if __name__ == "__main__":
    cities = load_cities("../data")
    image_paths, labels = load_dataset("../data")

    city_to_id = get_city_to_id()
    labels = [city_to_id[l] for l in labels]

    il = list(zip(image_paths, labels))
    shuffle(il)
    image_paths, labels = zip(*il)

    N = len(image_paths)
    split = 4 * N // 5
    train_paths = image_paths[:split]
    train_labels = labels[:split]

    test_paths = image_paths[split:]
    test_labels = labels[split:]

    model = load_model(sys.argv[1]) if len(sys.argv) > 1 else load_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    EPOCHS = 10
    active_train_transform = train_transform if RANDOM_VARIATION else None

    print(f"Training with augmentation: {RANDOM_VARIATION}")
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        loss, train_acc = train_epoch(model, train_paths, train_labels, optimizer, active_train_transform)
        test_acc = evaluate(model, test_paths, test_labels)
        print(f"Avg loss: {loss:.4f}, Train acc: {train_acc:.4f}, Test acc: {test_acc:.4f}")

    save_path = "../models/vlm.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")

    print("\nFinal evaluation on test set...")

    preds = get_predictions(model, test_paths, batch_size=16)
    print(f"Test accuracy: {np.mean(preds == np.array(test_labels)):.4f}")
    all_targets = np.array(test_labels)
    all_preds = preds
    print(f"Test accuracy: {np.mean(preds == np.array(test_labels)):.4f}")
    for city in cities:
        city_mask = [i for i, t in enumerate(test_labels) if t == city_to_id[city]]
        num_correct = sum([preds[i] == city_to_id[city] for i in city_mask])
        acc = num_correct / len(city_mask)
        print(f"{city}: {acc:.4f} ({num_correct}/{len(city_mask)})")
