import streamlit as st
from PIL import Image
import numpy as np

# --- SETTINGS ---
CLASS_NAMES = ["Apple 🍎", "Orange 🍊"]
IMG_SIZE = (128, 128)

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

    # For now, we show a friendly message
    # Later we can add the model prediction here
    st.success("✅ Image uploaded successfully!")
    st.info("This is your working Streamlit app interface.")

st.markdown("---")
st.write("💻 Created for Machine Learning Project")
