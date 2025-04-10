import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import json
import os
import requests
import time

# =============================================
# CONFIGURATION
# =============================================
MODEL_URL = "https://www.dropbox.com/scl/fi/i4mm96vcvo582peo8o6bt/pneumonia_detection-alpha.h5?rlkey=w3z1zuffakbuyqpynltbqs51c&st=jau9mi95&dl=1"
MODEL_PATH = "pneumonia_detection.h5"
CLASS_INDICES_PATH = "class_indices.json"

# =============================================
# MODEL LOADING WITH ERROR HANDLING
# =============================================
@st.cache_resource
def load_pneumonia_model():
    """Download and load model from Dropbox with fallback handling"""
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", MODEL_PATH)

    # Download model if not exists
    if not os.path.exists(model_path):
        with st.spinner("🚀 Downloading AI model from Dropbox..."):
            try:
                response = requests.get(MODEL_URL, stream=True)
                response.raise_for_status()
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            except Exception as e:
                st.error(f"❌ Model download failed: {str(e)}")
                return None, None

    # Load model and class indices
    try:
        model = load_model(model_path)

        if os.path.exists(CLASS_INDICES_PATH):
            with open(CLASS_INDICES_PATH, 'r') as f:
                class_indices = json.load(f)
        else:
            class_indices = {"Normal": 0, "Pneumonia": 1}  # Fallback

        return model, class_indices

    except Exception as e:
        st.error(f"🔥 Model loading error: {str(e)}")
        return None, None

# =============================================
# STREAMLIT UI
# =============================================
def main():
    st.set_page_config(page_title="Pneumonia Detector", page_icon="🩺")

    model, class_indices = load_pneumonia_model()

    if model is None:
        st.error("Critical error - cannot load AI model")
        return

    st.title("🩺 Pneumonia Detection from Chest X-rays")
    st.write("Upload a chest X-ray image to check for pneumonia signs")

    uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            img = Image.open(uploaded_file).convert('RGB')

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📤 Uploaded Image")
                display_img = img.copy()
                display_img.thumbnail((400, 400))
                st.image(display_img, use_container_width=True)

            with col2:
                st.subheader("🔍 AI Analysis")

                img = img.resize((224, 224))
                img_array = image.img_to_array(img)
                if img_array.shape[-1] != 3:
                    img_array = np.stack((img_array.squeeze(),)*3, axis=-1)

                img_array = np.expand_dims(img_array, axis=0) / 255.0

                prediction = model.predict(img_array)
                prob = float(prediction[0][0])

                if prob > 0.5:
                    st.error(f"🚨 Pneumonia detected (confidence: {prob:.2%})")
                else:
                    st.success(f"✅ Normal (confidence: {1 - prob:.2%})")

                st.progress(prob)
                st.metric("Pneumonia Probability", f"{prob:.2%}")
                st.caption("🖼️ Model Input Preview (224×224 pixels)")
                st.image(img, width=150)

        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")

    st.markdown("---")
    st.caption("⚠️ This AI tool is for educational/research purposes only. Consult a doctor for medical decisions.")

# =============================================
# RUN THE APP
# =============================================
if __name__ == "__main__":
    main()
