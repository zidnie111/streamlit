import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# --- SETTINGS ---
MODEL_PATH = "EfficientNetB0_fruit_model.h5"
CLASS_NAMES = ["Apple 🍎", "Orange 🍊"]
IMG_SIZE = (128, 128)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# --- UI DESIGN ---
st.set_page_config(page_title="Fruit Classifier", layout="centered")
st.title("🍎🍊 Fruit Classification App")
st.subheader("Using Transfer Learning")
st.write("Upload an image of an Apple or an Orange and the AI will identify it!")

# --- UPLOAD IMAGE ---
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Show image
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize(IMG_SIZE)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Predict
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array, verbose=0)[0][0]
    
    if prediction < 0.5:
        label = CLASS_NAMES[0]
        confidence = round((1 - prediction) * 100, 2)
        st.success(f"### Prediction: {label}")
        st.info(f"Confidence: {confidence}%")
    else:
        label = CLASS_NAMES[1]
        confidence = round(prediction * 100, 2)
        st.success(f"### Prediction: {label}")
        st.info(f"Confidence: {confidence}%")

st.markdown("---")
st.write("💻 Created for Machine Learning Project")
