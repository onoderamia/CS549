import os
import numpy as np
from sklearn.metrics import confusion_matrix
from model_utils import load_dataset, get_predictions, load_model, load_cities, get_city_to_id


if __name__ == "__main__":
    cities = load_cities("../data")

    image_paths, labels = load_dataset()
    city_to_id = get_city_to_id()
    y_true = np.array([city_to_id[c] for c in labels])

    model = load_model("/home/neil/git/UIUC/CS549/CS549/vlm/vlm_city_classifier.pth")
    
    y_pred = get_predictions(model, image_paths)

    cm = confusion_matrix(y_true, y_pred) 
    
    # convert to prob
    cm_prob = cm.astype(float)
    row_sums = cm_prob.sum(axis=1, keepdims=True)
    cm_prob = np.divide(cm_prob, row_sums, where=row_sums != 0)

    cities_sorted = [c for c, _ in sorted(city_to_id.items(), key=lambda x: x[1])]
    max_city_len = max(len(c) for c in cities_sorted)

    print("CONFUSION MATRIX")
    print("Rows: Actual city")
    print("Cols: Predicted city\n")

    header = " " * (max_city_len + 2)
    for c in cities_sorted:
        header += f"{c:>{max_city_len+2}}"
    print(header)

    for i, true_city in enumerate(cities_sorted):
        row_str = f"{true_city:<{max_city_len+2}}"
        for j in range(len(cities_sorted)):
            row_str += f"{cm_prob[i, j]:>{max_city_len+2}.2f}"
        print(row_str)

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # clear old results
    for city in cities_sorted:
        open(os.path.join(results_dir, f"{city}.txt"), "w").close()

    for img_path, pred_idx in zip(image_paths, y_pred):
        pred_city = cities[pred_idx]
        out_path = os.path.join(results_dir, f"{pred_city}.txt")

        with open(out_path, "a") as f:
            f.write(f"{img_path}\n")

    num_classes = cm.shape[0]

    # overall accuracy
    accuracy = np.trace(cm) / np.sum(cm)

    # TPR = TP / (TP + FN)
    tpr = np.zeros(num_classes)
    for i in range(num_classes):
        TP = cm[i, i]
        FN = np.sum(cm[i, :]) - TP
        tpr[i] = TP / (TP + FN)

    # TNR = TN / (FP + TN)
    tnr = np.zeros(num_classes)
    for i in range(num_classes):
        FP = np.sum(cm[:, i]) - cm[i, i]
        TN = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
        tnr[i] = TN / (FP + TN)

    print()
    print("\nOVERALL METRICS")
    print(f"Accuracy: {accuracy:.4f}\n")

    print("\nTPR:")
    for i, city in enumerate(cities_sorted):
        print(f"{city:<{max_city_len}} : {tpr[i]:.4f}")

    print("\nTNR:")
    for i, city in enumerate(cities_sorted):
        print(f"{city:<{max_city_len}} : {tnr[i]:.4f}")
