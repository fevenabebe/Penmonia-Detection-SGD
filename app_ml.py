%%writefile app_ml.py
import streamlit as st
import joblib
import numpy as np
from PIL import Image

st.title("🩺 Pneumonia Detection using ML (SVM)")
st.write("Upload a chest X-ray image to get the prediction using the trained SVM model.")

# Load the SVM model
model = joblib.load("ml_baseline_svm_model.pkl")
st.success("✅ ML SVM Model loaded successfully!")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert to grayscale
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess to match training (64x64, flatten, normalize)
    img = image.resize((64, 64))
    img_array = np.array(img).flatten() / 255.0
    img_array = img_array.reshape(1, -1)

    # Predict
    pred_class = model.predict(img_array)[0]
    pred_prob = model.predict_proba(img_array)[0][1] if hasattr(model, "predict_proba") else None
    label = "Pneumonia" if pred_class == 1 else "Normal"

    st.subheader(f"Prediction: {label}")
    if pred_prob is not None:
        confidence = pred_prob*100 if pred_class == 1 else (1-pred_prob)*100
        st.write(f"Confidence: {confidence:.2f}%")
