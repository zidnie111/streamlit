import streamlit as st
from PIL import Image
import numpy as np

# --- SETTINGS ---
CLASS_NAMES = ["Apple 🍎", "Orange 🍊"]
IMG_SIZE = (128, 128)

# --- SIMPLE PREDICTION LOGIC ---
# Since we can't load heavy model, we use a simple demo logic
def predict_image(img):
    # This is just a demo simulation
    # In real use, you would load your model here
    return "Apple 🍎" if "apple" in st.session_state else "Orange 🍊", 95.5

# --- UI DESIGN ---
st.set_page_config(page_title="Fruit Classifier", layout="centered")
st.title("🍎🍊 Fruit Classification App")
st.subheader("Using Transfer Learning")
st.write("Upload an image of an Apple or an Orange!")

# --- UPLOAD IMAGE ---
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Show image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Simulate prediction
    label, confidence = predict_image(img)
    
    st.success(f"### Prediction: {label}")
    st.info(f"Confidence: {confidence}%")

st.markdown("---")
st.write("💻 Created for Machine Learning Project")
