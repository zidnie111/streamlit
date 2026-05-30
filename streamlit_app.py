import streamlit as st
from PIL import Image
import numpy as np
from tflite_runtime.interpreter import Interpreter

# --- SETTINGS ---
MODEL_PATH = "fruit_model.tflite"
CLASS_NAMES = ["Apple 🍎", "Orange 🍊"]
IMG_SIZE = (128, 128)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

# --- PREDICT FUNCTION ---
def predict_image(img):
    # Preprocess image
    img = img.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Run prediction
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

    # Return result
    if prediction < 0.5:
        return CLASS_NAMES[0], round((1 - prediction) * 100, 2)
    else:
        return CLASS_NAMES[1], round(prediction * 100, 2)

# --- UI DESIGN ---
st.set_page_config(page_title="Fruit Classifier", layout="centered")
st.title("🍎🍊 Fruit Classification App")
st.subheader("Machine Learning | Transfer Learning")
st.write("Upload an image — AI will detect Apple or Orange!")

# --- UPLOAD IMAGE ---
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    label, confidence = predict_image(img)
    
    st.success(f"### Prediction: {label}")
    st.info(f"Confidence: {confidence}%")

st.markdown("---")
st.write("✅ Powered by TensorFlow Lite — Lightweight & Fast")
