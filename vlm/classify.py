import os
import sys
from PIL import Image
import matplotlib.pyplot as plt
from model_utils import load_model, classify_image, load_cities

def display_results(image, predictions, image_path):
    fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(14, 6))
    
    # show image
    ax_img.imshow(image)
    ax_img.set_title(f"Input: {os.path.basename(image_path)}", fontsize=12)
    ax_img.axis('off')
    
    # show preds
    cities = [p[0] for p in predictions]
    confidences = [p[1] for p in predictions]
    # green for top pred
    colors = ['#2ecc71'] + ["#4b555c" for i in range(len(predictions) - 1)]
    
    y_pos = range(len(cities) - 1, -1, -1)  # reverse
    bars = ax_bar.barh(y_pos, confidences, color=colors)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(cities)
    ax_bar.set_xlabel('Confidence')
    ax_bar.set_title('Predictions', fontsize=12)
    ax_bar.set_xlim(0, 1)
    
    # add labels on bars
    for bar, conf in zip(bars, confidences):
        width = bar.get_width()
        ax_bar.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                   f'{conf:.1%}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image_path = sys.argv[1]
    checkpoint_path = sys.argv[2] if len(sys.argv) > 2 else "../models/vlm.pth"
    
    print(f"Loading model from: {checkpoint_path}")
    load_cities("../data")
    model = load_model(checkpoint_path)

    print(f"Classifying: {image_path}")
    predictions = classify_image(model, image_path)
    img = Image.open(image_path).convert("RGB")
    
    display_results(img, predictions, image_path)

