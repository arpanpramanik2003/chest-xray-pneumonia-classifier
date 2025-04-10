import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import json
import gdown
import os

# Configure Google Drive model link
MODEL_URL = "https://drive.google.com/uc?id=13gVDE1x9Pd3PKXcXEiU4w8tl1Ej7ArzQ"
MODEL_PATH = "pneumonia_detection.h5"

# Load the model and class indices
@st.cache_resource
def load_pneumonia_model():
    # Download model if not exists
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model from Google Drive..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    
    # Load model and class indices
    model = load_model(MODEL_PATH)
    with open('class_indices.json', 'r') as f:
        class_indices = json.load(f)
    return model, class_indices

model, class_indices = load_pneumonia_model()

# Streamlit interface
st.title("Pneumonia Detection from Chest X-rays")
st.write("Upload a chest X-ray image to check for pneumonia")

# File uploader
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load and convert image to RGB (3 channels)
        img = Image.open(uploaded_file).convert('RGB')
        
        # Create two columns for layout
        col1, col2 = st.columns(2)
        
        with col1:
            # Display original image (resized for display)
            st.subheader("Uploaded Image")
            display_img = img.copy()
            display_img.thumbnail((400, 400))
            st.image(display_img, use_container_width=True)
        
        with col2:
            st.subheader("Analysis")
            
            # Preprocess image
            img = img.resize((224, 224))
            img_array = image.img_to_array(img)
            
            # Ensure 3 channels
            if img_array.shape[-1] != 3:
                img_array = np.stack((img_array.squeeze(),)*3, axis=-1)
            
            img_array = np.expand_dims(img_array, axis=0) / 255.0
            
            # Predict
            prediction = model.predict(img_array)
            prob = prediction[0][0]
            
            # Display results
            if prob > 0.5:
                st.error(f"Pneumonia detected (confidence: {prob:.2%})")
            else:
                st.success(f"Normal (confidence: {1-prob:.2%})")
            
            st.progress(float(prob))
            st.metric("Pneumonia Probability", f"{prob:.2%}")
            
            st.caption("Model Input (224×224 pixels)")
            st.image(img, width=150)
    
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
