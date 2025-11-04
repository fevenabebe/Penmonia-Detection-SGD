import streamlit as st
import pickle
import numpy as np
from PIL import Image

st.title("🩺 Pneumonia Detection using ML (SVM)")
st.write("Upload a chest X-ray image to get the prediction.")

# Load model
with open("ml_baseline_svm_model.pkl", "rb") as f:
    model = pickle.load(f)
st.success("✅ Model loaded successfully!")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img = image.resize((224,224))
    img_array = np.array(img).flatten().reshape(1, -1)  # flatten for ML SVM

    # Predict
    pred_class = model.predict(img_array)[0]

    # Display results
    st.subheader(f"Prediction: {pred_class}")
