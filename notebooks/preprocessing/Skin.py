import sys
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Idi dva koraka unazad (iz preprocessing u project), pa uđi u scripts/
sys.path.append(os.path.abspath(os.path.join("..", "..", "scripts")))
from MonkScale import (
    load_image,
    process_image,
    DEFAULT_TONE_PALETTE,
    default_tone_labels
)


def display_skin_analysis(image_path):
    """
    Učitava sliku, analizira ton kože i prikazuje rezultate

    Args:
        image_path (str): Putanja do slike za analizu
    """
    # Učitaj sliku
    image, base_filename, _ = load_image(image_path)

    # Pripremi Monk paletu i oznake
    monk_palette = DEFAULT_TONE_PALETTE["monk"]
    labels = default_tone_labels(monk_palette, "Monk ")

    # Obradi sliku
    records, report_images = process_image(
        image,
        is_bw=False,  # Pretpostavljamo da je slika u boji
        to_bw=False,  # Ne konvertujemo u crno-belu
        skin_tone_palette=monk_palette,
        tone_labels=labels,
        verbose=True  # Da bismo dobili slike izveštaja
    )

    # Prikaz rezultata
    if records:
        record = records[0]
        print(f"Filename: {base_filename}")
        print(f"Monk skala: {record['tone_label']}")
        print(f"Dominantna boja: {record['dominant_colors'][0]['color']}")
        print(f"Tačnost: {record['accuracy']}%")

    # Prikaz izveštaja
    key = list(report_images.keys())[0] if report_images else None
    if key:
        # Konvertuj BGR u RGB za Matplotlib
        rgb_image = cv2.cvtColor(report_images[key], cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(12, 8))
        plt.imshow(rgb_image)
        plt.title(f"Analiza tona kože: {base_filename}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        return record

    return None


# Funkcija za analizu više slika iz foldera
def analyze_folder(folder_path, num_images=10):
    """
    Analizira više slika iz foldera

    Args:
        folder_path (str): Putanja do foldera sa slikama
        num_images (int): Maksimalan broj slika za analizu
    """
    folder_path = Path(folder_path)
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in folder_path.iterdir()
                   if f.is_file() and
                   any(f.suffix.lower() == ext for ext in valid_extensions)]

    # Ograničavanje na traženi broj slika
    image_files = image_files[:min(num_images, len(image_files))]

    if not image_files:
        print(f"Nema slika u folderu: {folder_path}")
        return

    results = []
    for image_file in image_files:
        print(f"\nAnaliziranje slike: {image_file.name}")
        try:
            record = display_skin_analysis(str(image_file))
            if record:
                results.append({
                    "filename": image_file.name,
                    "monk_scale": record["tone_label"],
                    "dominant_color": record["dominant_colors"][0]["color"],
                    "accuracy": record["accuracy"]
                })
        except Exception as e:
            print(f"Greška pri obradi slike {image_file.name}: {str(e)}")

    return results


# Putanja do foldera sa slikama za analizu
path = os.path.abspath(os.path.join("..", "..", "data1", "raw", "train"))

# Analiziraj prvih 5 slika iz foldera
results = analyze_folder(path, num_images=5)

# Opciono: prikazi rezultate u tabeli
if results:
    import pandas as pd

    df = pd.DataFrame(results)
    print("\nTabela rezultata:")
    print(df)