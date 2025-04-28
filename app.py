# Importaciones (explicación de cada una)
from ultralytics import YOLO  # Framework YOLO para detección de objetos
import streamlit as st  # Para crear la interfaz web
from PIL import Image  # Procesamiento de imágenes
import numpy as np  # Manipulación de arrays (imágenes)
import time  # Medir tiempo de procesamiento (opcional)

# --- CONFIGURACIÓN INICIAL ---
# Cargar el modelo YOLO (versión pequeña 's' para CPU)
# Se descargará automáticamente 'yolov5s.pt' la primera vez
model = YOLO("yolov5s.pt").cpu()  # Forzar uso de CPU (aunque ya es el default)

# --- INTERFAZ CON STREAMLIT ---
st.title("🖼️ Detector de Objetos (YOLOv5)")  # Título con emoji
st.markdown("""
Sube una imagen y el modelo identificará los objetos presentes.
*Funciona mejor con fotos claras y objetos comunes.*
""")

# Widget para subir archivos (solo imágenes)
uploaded_file = st.file_uploader(
    "Elige una imagen...", 
    type=["jpg", "jpeg", "png"],
    help="Formatos soportados: JPG, PNG"
)

# --- PROCESAMIENTO CUANDO SE SUBE UNA IMAGEN ---
if uploaded_file is not None:
    # 1. Leer y mostrar la imagen original
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen original", use_column_width=True)
    
    # 2. Convertir a formato numpy (requerido por YOLO)
    image_np = np.array(image)
    
    # 3. Procesar con YOLO (con spinner para feedback)
    with st.spinner("🔍 Buscando objetos..."):
        start_time = time.time()  # Opcional: medir tiempo
        results = model(image_np)  # ¡Aquí ocurre la magia!
        process_time = time.time() - start_time
    
    # 4. Mostrar imagen con las detecciones
    annotated_img = results[0].plot()  # Dibuja cuadros y etiquetas
    st.image(annotated_img, caption="Objetos detectados", use_column_width=True)
    
    # 5. Mostrar resultados numéricos
    st.success(f"✅ Procesado en {process_time:.2f} segundos")
    st.subheader("📊 Resultados:")
    
    # Contador de objetos detectados
    detections_count = {}
    
    for det in results[0].boxes:
        label = model.names[int(det.cls)]  # Nombre del objeto
        confidence = det.conf.item()  # Confianza (0-1)
        
        # Filtrar detecciones con confianza baja (<50%)
        if confidence >= 0.5:
            # Contar objetos por tipo
            detections_count[label] = detections_count.get(label, 0) + 1
            
            # Opcional: Mostrar coordenadas del bounding box
            # x1, y1, x2, y2 = det.xyxy[0].tolist()
    
    # Mostrar resumen ordenado
    if detections_count:
        st.write("**Objetos encontrados:**")
        for obj, count in detections_count.items():
            st.write(f"- {obj.capitalize()}: {count}")
    else:
        st.warning("No se encontraron objetos con confianza suficiente (>=50%).")

# --- SECCIÓN INFORMATIVA ---
st.sidebar.markdown("""
### ℹ️ Información
- **Modelo:** YOLOv5s (optimizado para CPU)
- **Clases detectadas:** 80 (personas, coches, animales, etc.)
- **Precisión:** Variable según tamaño/calidad de imagen
- **Hardware:** Solo CPU (sin GPU)
""")