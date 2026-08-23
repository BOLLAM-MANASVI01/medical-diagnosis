"""
Standalone inference script.

Usage:
    python predict.py --image path/to/xray.jpg
"""

import argparse
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

IMG_SIZE = (224, 224)


def predict(image_path, model_path="model/medical_model.h5", class_map_path="model/class_indices.json"):
    model = load_model(model_path)

    with open(class_map_path) as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}

    img = load_img(image_path, target_size=IMG_SIZE)
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    prob = float(model.predict(arr)[0][0])
    pred_idx = int(prob > 0.5)
    label = idx_to_class[pred_idx]
    confidence = prob if pred_idx == 1 else 1 - prob

    return label, confidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()

    label, confidence = predict(args.image)
    print(f"Prediction: {label} (confidence: {confidence*100:.2f}%)")
