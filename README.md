### Proyecto desde 0 (cero) con máquina de bajos recursos Compaq Presario CQ40 con Intel Pentium, 4GB de Ram y 240GB de SSD, 64bits para realizar Apps Python con librerías de modelos preentrenados de IA. 
# python_ia

# **¿Qué hace el código?
Carga el modelo YOLO preentrenado (yolov5s.pt):

Usa la versión pequeña (s) de YOLOv5, optimizada para velocidad y rendimiento en CPU.

El modelo ya sabe detectar objetos comunes (personas, coches, sillas, etc.) porque fue entrenado con el dataset COCO.

Crea una interfaz web con Streamlit:

Muestra un título: "Detector de Objetos".

Permite subir una imagen (formatos JPG o PNG) mediante un botón de carga.

Procesa la imagen y detecta objetos:

Convierte la imagen subida a un array de NumPy (formato que OpenCV y YOLO entienden).

Pasa la imagen por el modelo YOLO, que devuelve las detecciones (objetos encontrados, sus posiciones y confianza).

Muestra los resultados:

Dibuja rectángulos y etiquetas sobre la imagen original (usando results[0].plot()).

Muestra la imagen anotada en la web.

Lista los objetos detectados junto con su confianza (ejemplo: "persona: 0.92" significa 92% de seguridad).

# Ejemplo de uso
Subes una foto de una calle con coches y peatones.

El código devuelve:

La misma imagen con los objetos marcados (cada uno con un cuadro y su nombre).

Una lista como:

coche: 0.85
persona: 0.92
semáforo: 0.76

# Limitaciones (por no tener GPU)
Velocidad: En una CPU antigua (como tu Pentium T4200), el procesamiento puede tardar unos segundos (especialmente en imágenes grandes).

Precisión: El modelo yolov5s es ligero pero menos preciso que versiones más grandes (yolov5m, yolov5l).

Calentamiento: Si procesas muchas imágenes seguidas, la CPU podría calentarse.

# Recomendaciones para mejorarlo
Optimizar para CPU:

Usa imágenes más pequeñas (ej: 640x640 píxeles) para acelerar el procesamiento.

Añade model = YOLO("yolov5s.pt").cpu() para forzar el uso de CPU (aunque Ultralytics ya lo hace por defecto si no detecta GPU).

Filtrar detecciones:

Ignora objetos con confianza baja (ej: < 0.5):

 for det in results[0].boxes:
    if det.conf.item() > 0.5:  # Solo muestra si confianza > 50%
        label = model.names[int(det.cls)]
        st.write(f"{label}: {det.conf.item():.2f}")

# Interfaz más amigable:

Agrega un spinner mientras procesa:

with st.spinner("Analizando imagen..."):
    results = model(image_np)

# ¿Qué tipo de imágenes funcionan mejor?
Escenas con objetos comunes (COCO incluye 80 categorías: https://cocodataset.org/#home).

Fotos con buena iluminación y objetos no muy pequeños.

Evita imágenes borrosas o con muchos objetos superpuestos.

Si quieres probarlo, ¡sube una foto y verás cómo detecta los objetos! 🚀

