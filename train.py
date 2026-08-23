"""
Medical Diagnosis - Pneumonia Detection from Chest X-Rays
Improved training script: transfer learning + class balancing + proper evaluation.

Usage:
    python train.py --data_dir dataset --epochs 15
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import json

IMG_SIZE = (224, 224)  # standard input size for MobileNetV2
BATCH_SIZE = 32


def build_model():
    """Transfer learning model: frozen MobileNetV2 backbone + custom classification head."""
    base_model = MobileNetV2(
        weights="imagenet", include_top=False, input_shape=IMG_SIZE + (3,)
    )
    base_model.trainable = False  # freeze pretrained feature extractor

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def get_data_generators(data_dir):
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )
    val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_data = train_datagen.flow_from_directory(
        os.path.join(data_dir, "train"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
    )
    val_data = val_test_datagen.flow_from_directory(
        os.path.join(data_dir, "val"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
    )
    test_data = val_test_datagen.flow_from_directory(
        os.path.join(data_dir, "test"),
        target_size=IMG_SIZE,
        batch_size=1,
        class_mode="binary",
        shuffle=False,
    )
    return train_data, val_data, test_data


def main(args):
    os.makedirs("model", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    train_data, val_data, test_data = get_data_generators(args.data_dir)

    # Handle class imbalance (pneumonia datasets are typically imbalanced)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_data.classes),
        y=train_data.classes,
    )
    class_weights = dict(enumerate(class_weights))
    print("Class weights:", class_weights)

    model = build_model()
    model.summary()

    callbacks = [
        ModelCheckpoint("model/best_model.h5", save_best_only=True, monitor="val_accuracy"),
        EarlyStopping(patience=4, restore_best_weights=True, monitor="val_loss"),
        ReduceLROnPlateau(factor=0.2, patience=2, monitor="val_loss"),
    ]

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    # Save final model + class index mapping (needed at inference time)
    model.save("model/medical_model.h5")
    with open("model/class_indices.json", "w") as f:
        json.dump(train_data.class_indices, f)

    # ---- Evaluation on held-out test set ----
    test_data.reset()
    y_true = test_data.classes
    y_pred_probs = model.predict(test_data).ravel()
    y_pred = (y_pred_probs > 0.5).astype(int)

    print("\nClassification Report:\n")
    report = classification_report(y_true, y_pred, target_names=list(test_data.class_indices.keys()))
    print(report)
    with open("outputs/classification_report.txt", "w") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)

    try:
        auc = roc_auc_score(y_true, y_pred_probs)
        print(f"ROC-AUC: {auc:.4f}")
    except ValueError:
        auc = None

    # ---- Plot training curves ----
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="train_acc")
    plt.plot(history.history["val_accuracy"], label="val_acc")
    plt.legend(); plt.title("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.legend(); plt.title("Loss")
    plt.tight_layout()
    plt.savefig("outputs/training_curves.png")

    print("\nSaved: model/medical_model.h5, outputs/classification_report.txt, outputs/training_curves.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="dataset")
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()
    main(args)
