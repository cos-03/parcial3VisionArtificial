#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parcial: Visión Artificial con Redes Convolucionales
Archivo: parcial_cnn.py
Descripción: Carga un dataset de imágenes, aplica preprocesamiento,
usa modelo preentrenado (mobilenet_v2, resnet50, vgg16) para predecir,
guarda resultados y genera evidencias (imágenes antes/después y CSV).
Autor: (tu nombre)
"""

import os
import sys
import argparse
from PIL import Image, ImageOps, ImageFilter
import torch
import torchvision
from torchvision import transforms, models
import torchvision.transforms.functional as TF
import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import datetime
import urllib.request

# -------------------------
# Utilidades
# -------------------------
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_imagenet_labels(target_path="imagenet_classes.txt"):
    """
    Intenta cargar labels de ImageNet desde archivo local; si no existe,
    intenta descargarlas desde el repo de PyTorch hub.
    Retorna lista de 1000 nombres (strings). Si falla, retorna None.
    """
    if os.path.exists(target_path):
        with open(target_path, "r", encoding="utf-8") as f:
            labels = [l.strip() for l in f.readlines()]
            if len(labels) >= 1000:
                return labels
    # intentar descargar
    urls = [
        "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
        "https://raw.githubusercontent.com/pytorch/vision/main/torchvision/models/imagenet_classes.txt"
    ]
    for url in urls:
        try:
            print(f"Intentando descargar etiquetas ImageNet desde {url} ...")
            urllib.request.urlretrieve(url, target_path)
            with open(target_path, "r", encoding="utf-8") as f:
                labels = [l.strip() for l in f.readlines()]
                if len(labels) >= 1000:
                    return labels
        except Exception as e:
            print("Descarga fallida:", e)
    print("No se pudieron cargar las etiquetas de ImageNet. Se mostrarán índices en lugar de nombres.")
    return None

# -------------------------
# Preprocesamientos
# -------------------------
def preprocess_pil(img_pil, preprocess_name="default"):
    """
    Recibe PIL image; devuelve PIL image preprocesada según opción.
    preprocess_name: "default" (resize+center crop), "hist_eq", "blur", "grayscale", "contrast"
    """
    # Normal resize+center crop 256 -> 224 center crop (standard)
    if preprocess_name == "default":
        img = ImageOps.fit(img_pil, (224, 224), Image.BILINEAR, centering=(0.5, 0.5))
        return img.convert("RGB")
    elif preprocess_name == "hist_eq":
        # Convert to YCbCr, equalize Y channel
        img_rgb = ImageOps.fit(img_pil, (224, 224), Image.BILINEAR)
        ycbcr = img_rgb.convert("YCbCr")
        y, cb, cr = ycbcr.split()
        y_eq = ImageOps.equalize(y)
        img_eq = Image.merge("YCbCr", (y_eq, cb, cr)).convert("RGB")
        return img_eq
    elif preprocess_name == "blur":
        img = ImageOps.fit(img_pil, (224, 224), Image.BILINEAR)
        return img.filter(ImageFilter.GaussianBlur(radius=1.5)).convert("RGB")
    elif preprocess_name == "grayscale":
        img = ImageOps.fit(img_pil.convert("L"), (224, 224), Image.BILINEAR)
        return img.convert("RGB")
    elif preprocess_name == "contrast":
        img = ImageOps.fit(img_pil, (224, 224), Image.BILINEAR)
        # simple contrast stretch using ImageOps.autocontrast
        return ImageOps.autocontrast(img).convert("RGB")
    else:
        # fallback to default
        return preprocess_pil(img_pil, "default")

# -------------------------
# Transformación a tensor y normalización (para modelos preentrenados)
# -------------------------
imagenet_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -------------------------
# Funcion principal
# -------------------------
def run(dataset_dir, model_name, extra_preprocess, out_dir, device, show_examples):
    # directorios de salida
    ensure_dir(out_dir)
    resultados_dir = os.path.join(out_dir, "resultados")
    ensure_dir(resultados_dir)
    evidencias_dir = os.path.join(out_dir, "evidencias")
    ensure_dir(evidencias_dir)

    # cargar etiquetas imagenet
    labels = load_imagenet_labels()

    # Seleccionar modelo
    model_name = model_name.lower()
    if model_name == "resnet50":
        model = models.resnet50(pretrained=True)
    elif model_name == "vgg16":
        model = models.vgg16(pretrained=True)
    elif model_name == "mobilenet_v2" or model_name == "mobilenetv2":
        model = models.mobilenet_v2(pretrained=True)
    else:
        raise ValueError("Modelo no soportado. Elige resnet50, vgg16 o mobilenet_v2")

    model.eval()
    model.to(device)

    # Recorrer imágenes en dataset_dir (archivos comunes de imagen)
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    image_files = []
    for root, _, files in os.walk(dataset_dir):
        for f in files:
            if os.path.splitext(f.lower())[1] in valid_ext:
                image_files.append(os.path.join(root, f))
    image_files.sort()

    if len(image_files) == 0:
        print("No se encontraron imágenes en", dataset_dir)
        return

    csv_path = os.path.join(out_dir, "predicciones.csv")
    csv_fields = ["archivo", "pred_idx", "pred_class", "confidence", "preprocesamiento"]

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
        writer.writeheader()

        for i, img_path in enumerate(image_files):
            try:
                img_orig = Image.open(img_path).convert("RGB")
            except Exception as e:
                print("No se pudo abrir:", img_path, e)
                continue

            # guardar miniatura original para evidencia
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            orig_thumb_path = os.path.join(evidencias_dir, f"{base_name}_original.jpg")
            img_orig.resize((256,256)).save(orig_thumb_path)

            # preprocesamiento inicial (el requerido en el parcial)
            img_pre = preprocess_pil(img_orig, "default")
            before_img_path = os.path.join(evidencias_dir, f"{base_name}_before_after.png")

            # aplicar preprocesamiento adicional solicitado por el estudiante
            if extra_preprocess != "none":
                img_pre2 = preprocess_pil(img_orig, extra_preprocess)
            else:
                img_pre2 = img_pre

            # transformar para modelo
            tensor_in = imagenet_transform(img_pre2).unsqueeze(0).to(device)

            # inferencia
            with torch.no_grad():
                outputs = model(tensor_in)
                probs = torch.nn.functional.softmax(outputs[0], dim=0)
                conf, idx = torch.max(probs, dim=0)
                conf = float(conf.cpu().item())
                idx = int(idx.cpu().item())

            pred_class = labels[idx] if labels and idx < len(labels) else str(idx)

            # guardar imagen procesada
            processed_path = os.path.join(resultados_dir, f"{base_name}_processed.jpg")
            img_pre2.save(processed_path)

            # guardar evidencia gráfica (side-by-side)
            # crear canvas con antes (resized) y despues
            a = img_orig.copy().resize((224,224))
            b = img_pre2.copy().resize((224,224))
            canvas = Image.new("RGB", (224*2+10, 224))
            canvas.paste(a, (0,0))
            canvas.paste(b, (224+10,0))
            # escribir texto simple con matplotlib para incluir predicción
            plt.figure(figsize=(6,3))
            plt.imshow(canvas)
            plt.axis('off')
            title = f"{base_name}  → {pred_class} ({conf*100:.2f}%)"
            plt.title(title, fontsize=9)
            ev_path = os.path.join(evidencias_dir, f"{base_name}_evidence.png")
            plt.savefig(ev_path, bbox_inches='tight', dpi=150)
            plt.close()

            # escribir fila en CSV
            writer.writerow({
                "archivo": os.path.relpath(img_path, dataset_dir),
                "pred_idx": idx,
                "pred_class": pred_class,
                "confidence": f"{conf:.6f}",
                "preprocesamiento": extra_preprocess
            })

            # mostrar algunos ejemplos si se solicita
            if show_examples and i < 5:
                print(f"[{i+1}/{len(image_files)}] {base_name} -> {pred_class} ({conf*100:.2f}%)")
            elif i % 50 == 0:
                print(f"Procesadas {i+1}/{len(image_files)} imágenes...")

    print("Terminado. Resultados guardados en:", out_dir)
    print("- CSV:", csv_path)
    print("- Resultados (imágenes procesadas):", resultados_dir)
    print("- Evidencias (antes/después y capturas):", evidencias_dir)

# -------------------------
# CLI
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Parcial Vision Artificial - CNN (PyTorch)")
    parser.add_argument("--dataset", "-d", required=True,
                        help="Carpeta con imágenes (subcarpetas válidas).")
    parser.add_argument("--model", "-m", default="resnet50",
                        choices=["resnet50","vgg16","mobilenet_v2"],
                        help="Modelo preentrenado a usar.")
    parser.add_argument("--preproc", "-p", default="none",
                        choices=["none","default","hist_eq","blur","grayscale","contrast"],
                        help="Preprocesamiento adicional a aplicar (parte III).")
    parser.add_argument("--out", "-o", default="./salida_parcial",
                        help="Directorio de salida para resultados y evidencias.")
    parser.add_argument("--no-gpu", action="store_true", help="Forzar CPU (no usar GPU).")
    parser.add_argument("--show", action="store_true", help="Mostrar en consola algunas predicciones (ejemplos).")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if (torch.cuda.is_available() and not args.no_gpu) else "cpu")
    print("Usando dispositivo:", device)
    run(args.dataset, args.model, args.preproc, args.out, device, args.show)
