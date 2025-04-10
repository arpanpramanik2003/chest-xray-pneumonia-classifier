import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import json
import gdown
import os
import requests
import time

# =============================================
# CONFIGURATION
# =============================================
MODEL_ID = "13gVDE1x9Pd3PKXcXEiU4w8tl1Ej7ArzQ" 
MODEL_PATH = "pneumonia_detection.h5"
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_ID}&export=download"
CLASS_INDICES_PATH = "class_indices.json"

# =============================================
# MODEL LOADING WITH ERROR HANDLING
# =============================================
@st.cache_resource
def load_pneumonia_model():
    """Load model with multiple fallback download methods"""
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", MODEL_PATH)
    
    # Download model if not exists
    if not os.path.exists(model_path):
        with st.spinner("🚀 Downloading AI model (this may take 2-3 minutes)..."):
            # Try Google Drive download
            try:
                gdown.download(MODEL_URL, model_path, quiet=False)
                if not os.path.exists(model_path):
                    raise RuntimeError("Google Drive download failed")
                    
            except Exception as e:
                st.warning(f"⚠️ Primary download failed: {str(e)}")
                time.sleep(2)
                
                # Try direct download fallback
                try:
                    response = requests.get(MODEL_URL, stream=True)
                    response.raise_for_status()
                    with open(model_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                except Exception as e:
                    st.error(f"❌ All download methods failed: {str(e)}")
                    st.info("ℹ️ Please check your internet connection or contact support")
                    return None, None
    
    # Load model and class indices
    try:
        model = load_model(model_path)
        with open(CLASS_INDICES_PATH, 'r') as f:
            class_indices = json.load(f)
        return model, class_indices
    except Exception as e:
        st.error(f"🔥 Model loading error: {str(e)}")
        return None, None

# =============================================
# STREAMLIT UI
# =============================================
def main():
    st.set_page_config(page_title="Pneumonia Detector", page_icon="🩺")
    
    # Load model (with caching)
    model, class_indices = load_pneumonia_model()
    
    if model is None:
        st.error("Critical error - cannot load AI model")
        return
    
    st.title("🩺 Pneumonia Detection from Chest X-rays")
    st.write("Upload a chest X-ray image to check for pneumonia signs")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an X-ray image...", 
                                   type=["jpg", "jpeg", "png"],
                                   accept_multiple_files=False)
    
    if uploaded_file is not None:
        try:
            # Load and convert image to RGB (3 channels)
            img = Image.open(uploaded_file).convert('RGB')
            
            # Create two columns for layout
            col1, col2 = st.columns(2)
            
            with col1:
                # Display original image (resized for display)
                st.subheader("📤 Uploaded Image")
                display_img = img.copy()
                display_img.thumbnail((400, 400))
                st.image(display_img, use_container_width=True)
            
            with col2:
                st.subheader("🔍 AI Analysis")
                
                # Preprocess image for model
                img = img.resize((224, 224))
                img_array = image.img_to_array(img)
                
                # Ensure we have 3 channels
                if img_array.shape[-1] != 3:
                    img_array = np.stack((img_array.squeeze(),)*3, axis=-1)
                
                img_array = np.expand_dims(img_array, axis=0) / 255.0
                
                # Make prediction
                prediction = model.predict(img_array)
                prob = float(prediction[0][0])
                
                # Display results
                if prob > 0.5:
                    st.error(f"🚨 Pneumonia detected (confidence: {prob:.2%})")
                else:
                    st.success(f"✅ Normal (confidence: {1-prob:.2%})")
                
                # Visual indicators
                st.progress(prob)
                st.metric("Pneumonia Probability", f"{prob:.2%}")
                
                # Model input preview
                st.caption("🖼️ Model Input Preview (224×224 pixels)")
                st.image(img, width=150)
        
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")
            st.info("ℹ️ Please try another image file")

    # Footer
    st.markdown("---")
    st.caption("⚠️ Note: This AI tool is for research purposes only. Always consult a medical professional for diagnosis.")

# =============================================
# RUN THE APP
# =============================================
if __name__ == "__main__":
    main()
