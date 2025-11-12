#!/usr/bin/env python3
"""
build_fashion_mints.py

Qué hace:
- Descarga imágenes desde una lista de URLs de páginas (Unsplash/Pexels/etc).
- Extrae la URL real de la imagen (og:image / meta tags).
- Descarga la imagen, quita el fondo (usando rembg), redimensiona a 80x80 px,
  guarda como PNG con transparencia en /fashion-mints/<categoria>/.
- Crea metadata.csv con: filename, category, color_dominante, source_url
- Empaqueta todo en fashion-mints.zip

Requisitos:
pip install requests beautifulsoup4 pillow rembg tqdm numpy scikit-image
(rembg usa modelos; requiere pip >= build tools)
"""

import os
import csv
import io
import zipfile
import requests
from bs4 import BeautifulSoup
from PIL import Image
from rembg import remove
from tqdm import tqdm
from urllib.parse import urlparse

OUT_ROOT = "fashion-mints"
CATEGORIES = {"camiseta": [], "pantalon": [], "zapato": []}
TARGET_SIZE = (80, 80)
MAX_IMAGES = 80

# --- Lista inicial de páginas con imágenes (curadas: Unsplash / Pexels)
# Puedes añadir/editar URLs aquí para alcanzar 80 imágenes.
SOURCE_PAGES = [
    # Camisetas / tops (ejemplos)
    ("camiseta", "https://www.pexels.com/photo/flat-lay-of-mint-green-t-shirt-with-cotton-stems-34156905/"),
    ("camiseta", "https://www.pexels.com/photo/minimalist-flat-lay-of-cotton-t-shirts-34156906/"),
    ("camiseta", "https://unsplash.com/photos/woman-wearing-a-personal-trainer-t-shirt-on-black-background-MSDPqpFDTvU"),
    ("camiseta", "https://unsplash.com/s/photos/tshirt"),
    # Pantalones
    ("pantalon", "https://unsplash.com/photos/person-in-ripped-jeans-with-a-light-green-suitcase-HhPof9DmLnI"),
    ("pantalon", "https://unsplash.com/photos/a-pair-of-jeans-with-a-flower-CLwiuvrtKF4"),
    ("pantalon", "https://lefaire.co/cdn/shop/files/Pants_ClearlyAquaFlatlayFront.png"),
    # Zapatos
    ("zapato", "https://unsplash.com/photos/person-in-green-nike-athletic-shoes-4-hLHgJnLqg"),
    ("zapato", "https://unsplash.com/photos/a-pair-of-green-shoes-sitting-on-top-of-a-wooden-table-1kYX0v4DdrQ"),
    ("zapato", "https://unsplash.com/photos/white-and-green-nike-athletic-shoes-PoRfPVPPL4I"),
    # Más resultados de búsquedas (puedes añadir otros enlaces de Pexels/Unsplash):
    ("camiseta", "https://www.pexels.com/search/green%20t-shirt/"),
    ("camiseta", "https://unsplash.com/s/photos/tee"),
    ("pantalon", "https://unsplash.com/photos/flat-lay-of-summer-clothing-and-accessories-on-green-background-C6sC6C-8pJM"),
    ("zapato", "https://unsplash.com/photos/pair-of-teal-converse-all-star-high-tops-34EmbPeDBRg"),
    # (Añade más enlaces hasta alcanzar 80)
]

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; dataset-builder/1.0)"}

def ensure_dirs():
    os.makedirs(OUT_ROOT, exist_ok=True)
    for c in CATEGORIES:
        os.makedirs(os.path.join(OUT_ROOT, c), exist_ok=True)

def fetch_page_image_url(page_url):
    """
    Intenta extraer la URL real de la imagen de la página:
    - Busca meta property="og:image"
    - Busca <link rel="image_src">
    - Busca <img> con src que tenga 'images' o 'unsplash' o 'pexels'
    Devuelve None si no encuentra.
    """
    try:
        r = requests.get(page_url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return None
        soup = BeautifulSoup(r.text, "html.parser")
        # og:image
        tag = soup.find("meta", property="og:image")
        if tag and tag.get("content"):
            return tag["content"]
        tag2 = soup.find("link", rel="image_src")
        if tag2 and tag2.get("href"):
            return tag2["href"]
        # fallback: primera imagen grande
        imgs = soup.find_all("img")
        candidate = None
        maxw = 0
        for img in imgs:
            src = img.get("src") or img.get("data-src")
            if not src: continue
            # prefer urls from unsplash/pexels
            score = 0
            if "unsplash" in src: score += 3
            if "pexels" in src: score += 3
            if len(src) > maxw:
                candidate = src
                maxw = len(src)
        return candidate
    except Exception as e:
        print("Error fetch_page_image_url:", e)
        return None

def download_image(url):
    try:
        r = requests.get(url, headers=HEADERS, stream=True, timeout=20)
        if r.status_code == 200:
            return r.content
    except Exception as e:
        return None
    return None

def process_and_save(img_bytes, out_path):
    # 1) load
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    except Exception as e:
        print("PIL open error:", e)
        return False
    # 2) remove background via rembg
    try:
        result = remove(img)
        if isinstance(result, bytes):
            img = Image.open(io.BytesIO(result)).convert("RGBA")
        else:
            # rembg may return PIL Image; ensure RGBA
            img = result.convert("RGBA")
    except Exception as e:
        print("rembg error, continuing with original image:", e)
    # 3) resize & center-crop or pad to target while keeping aspect
    img.thumbnail((TARGET_SIZE[0], TARGET_SIZE[1]), Image.LANCZOS)
    # create transparent background and paste centered
    bg = Image.new("RGBA", TARGET_SIZE, (0,0,0,0))
    x = (TARGET_SIZE[0] - img.width) // 2
    y = (TARGET_SIZE[1] - img.height) // 2
    bg.paste(img, (x,y), img)
    bg.save(out_path, format="PNG")
    return True

def guess_color_dominant(img_path):
    # Simple heuristic: open, resize small, get most common non-transparent pixel
    try:
        im = Image.open(img_path).convert("RGBA").resize((32,32))
        pixels = [p for p in im.getdata() if p[3] > 50]
        if not pixels: return "transparent"
        # count rgb
        from collections import Counter
        counter = Counter([(p[0], p[1], p[2]) for p in pixels])
        dominant_rgb = counter.most_common(1)[0][0]
        return "#{:02x}{:02x}{:02x}".format(*dominant_rgb)
    except Exception as e:
        print("Color detection error:", e)
        return "unknown"
