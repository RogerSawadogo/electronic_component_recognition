import streamlit as st
from PIL import Image
import torch
import numpy as np
import tempfile
from pathlib import Path

# Charger le modèle YOLOv5
@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
    return model

model = load_model()

# App
st.title("🔍 Electronic Components Detector with YOLOv5")
st.write("Upload an image containing electronic components to detect them.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Sauvegarde temporaire
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        img.save(tmp.name)
        results = model(tmp.name)

    # Affichage des résultats
    results.render()  # ajoute les boîtes sur l’image
    for img in results.ims:
        st.image(img, caption="Detected Components", use_container_width=True)

    # Afficher les labels détectés
    st.write("### Detections:")
    for *box, conf, cls in results.xyxy[0].tolist():
        label = model.names[int(cls)]
        st.write(f"- **{label}** (confidence: {conf:.2f})")
