"""
Streamlit demo app - deployable to Streamlit Community Cloud or Hugging Face Spaces.

Run locally:
    streamlit run streamlit_app.py
"""

import json
import numpy as np
import streamlit as st
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

IMG_SIZE = (224, 224)
MODEL_PATH = "model/medical_model.h5"
CLASS_MAP_PATH = "model/class_indices.json"


@st.cache_resource
def load_assets():
    model = load_model(MODEL_PATH)
    with open(CLASS_MAP_PATH) as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}
    return model, idx_to_class


def preprocess(image: Image.Image):
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = img_to_array(image) / 255.0
    return np.expand_dims(arr, axis=0), image


def find_last_conv_layer(model):
    """Auto-detect the last Conv2D layer in the model (works regardless of backbone)."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in this model for Grad-CAM.")


def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None):
    """Grad-CAM: highlights the image regions that most influenced the prediction."""
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)

    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        raise ValueError(f"No gradient could be computed for layer '{last_conv_layer_name}'.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(heatmap, original_image, alpha=0.4):
    """Overlay a Grad-CAM heatmap on the original image using OpenCV's JET colormap."""
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_resized = cv2.resize(heatmap_uint8, original_image.size)  # (width, height)
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR, convert to RGB

    original_array = np.array(original_image).astype(np.float32)
    overlayed = original_array * (1 - alpha) + heatmap_color.astype(np.float32) * alpha
    return Image.fromarray(np.uint8(overlayed))


st.set_page_config(page_title="Chest X-Ray Diagnosis", layout="centered")
st.title("🩻 Medical Diagnosis: Pneumonia Detection")
st.write(
    "Upload a chest X-ray image. The model predicts **Normal** vs **Disease (Pneumonia)** "
    "and highlights the regions it focused on using Grad-CAM."
)
st.caption(
    "⚠️ Educational/portfolio project only — not a medical device, not for clinical use."
)

uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    model, idx_to_class = load_assets()
    image = Image.open(uploaded_file)
    img_array, resized_image = preprocess(image)

    prob = float(model.predict(img_array)[0][0])
    pred_idx = int(prob > 0.5)
    label = idx_to_class[pred_idx]
    confidence = prob if pred_idx == 1 else 1 - prob

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded X-Ray", use_container_width=True)
    with col2:
        try:
            heatmap = make_gradcam_heatmap(img_array, model)
            overlayed = overlay_heatmap(heatmap, resized_image)
            st.image(overlayed, caption="Grad-CAM: model focus area", use_container_width=True)
        except Exception as e:
            st.warning("Grad-CAM failed — showing error for debugging:")
            st.exception(e)

    st.markdown(f"### Prediction: **{label}**")
    st.progress(confidence)
    st.write(f"Confidence: **{confidence*100:.2f}%**")