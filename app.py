import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

st.set_page_config(page_title="Pneumonia Detection", page_icon="🩺", layout="centered")

st.title("🩺 Pneumonia Detection using MobileNetV2")
st.write("Upload a chest X-ray image to get the prediction.")

# Load the trained model
model = load_model("MobileNetV2_BEST_SGD.keras")
st.success("✅ Model loaded successfully!")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 224, 224, 3)

    # Predict
    pred_prob = model.predict(img_array)[0][0]
    pred_class = "Pneumonia" if pred_prob > 0.5 else "Normal"

    # Display results
    st.subheader(f"Prediction: {pred_class}")
    st.write(f"Confidence: {pred_prob*100:.2f}%")
