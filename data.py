import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from PIL import Image # ¡Nueva librería necesaria!

# --- 1. Definiciones y Configuración ---
# Nuevo tamaño de salida solicitado
TAMANO_SALIDA = (80, 80)
NUM_IMAGENES_TOTAL = 80
CARPETA_SALIDA = 'mini_dataset_80x80_imagenes'

# Etiquetas de Fashion-MNIST requeridas
ETIQUETAS_REQUERIDAS = [0, 1, 5, 7, 9]

# Mapeo de etiquetas a nombres de subcarpetas
NOMBRE_CARPETA = {
    0: 'Camisetas',
    1: 'Pantalones',
    5: 'Zapatos_Sandalias',
    7: 'Zapatos_Zapatillas',
    9: 'Zapatos_Botines'
}

# Definición de la distribución (puede ajustarla)
CUOTAS = {
    0: 27, # Camisetas
    1: 27, # Pantalones
    5: 8,  # Zapatos (Sandalias)
    7: 8,  # Zapatos (Zapatillas)
    9: 10  # Zapatos (Botines)
}

# --- 2. Cargar y Filtrar el Dataset ---
print(f"Cargando el dataset Fashion-MNIST...")
(imagenes, etiquetas), _ = tf.keras.datasets.fashion_mnist.load_data()

# Crear la carpeta principal si no existe
os.makedirs(CARPETA_SALIDA, exist_ok=True)

# Crear las subcarpetas
for nombre in NOMBRE_CARPETA.values():
    os.makedirs(os.path.join(CARPETA_SALIDA, nombre), exist_ok=True)

imagenes_guardadas = 0

print("Comenzando el proceso de filtrado, redimensionamiento y guardado...")

# --- 3. Iterar, Seleccionar, Redimensionar y Guardar ---

for etiqueta, cuota in CUOTAS.items():
    indices = np.where(etiquetas == etiqueta)[0]
    indices_seleccionados = random.sample(list(indices), min(cuota, len(indices)))
    
    imagenes_a_guardar = imagenes[indices_seleccionados]
    nombre_carpeta = NOMBRE_CARPETA[etiqueta]
    
    print(f"  -> Procesando y guardando {len(indices_seleccionados)} imágenes de: {nombre_carpeta}")

    for i, imagen_np in enumerate(imagenes_a_guardar):
        # 1. Convertir el array numpy (28x28) a un objeto Image de PIL
        imagen_pil = Image.fromarray(imagen_np)
        
        # 2. Redimensionar la imagen a 80x80 píxeles
        imagen_redimensionada = imagen_pil.resize(TAMANO_SALIDA, Image.Resampling.LANCZOS)
        
        # 3. La ruta completa del archivo
        ruta_archivo = os.path.join(CARPETA_SALIDA, nombre_carpeta, f"{nombre_carpeta.lower()}_{i}_80x80.png")
        
        # 4. Guardar la imagen redimensionada
        imagen_redimensionada.save(ruta_archivo)
        imagenes_guardadas += 1

print("-" * 40)
print(f"✅ Proceso completado. Se guardaron {imagenes_guardadas} imágenes de {TAMANO_SALIDA[0]}x{TAMANO_SALIDA[1]} píxeles.")
print(f"Las imágenes están en la carpeta: './{CARPETA_SALIDA}'")