from ultralytics import YOLO
import streamlit as st
import cv2
from PIL import Image
import numpy as np

# Cargar modelo
model = YOLO("yolov5s.pt")

st.title("Detector de Objetos")
uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "png"])
if uploaded_file:
    # Procesar imagen
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    results = model(image_np)
    # Mostrar resultados
    annotated_img = results[0].plot()
    st.image(annotated_img, caption="Objetos detectados")
    st.write("Objetos encontrados:")
    for det in results[0].boxes:
        label = model.names[int(det.cls)]
        confidence = det.conf.item()
        st.write(f"{label}: {confidence:.2f}")