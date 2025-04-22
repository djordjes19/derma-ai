import os
import shutil
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

class Augmentation:
    def __init__(self):
        pass


    def brightness(self, img):
        random.seed()
        enhance = ImageEnhance.Brightness(img)
        return enhance.enhance(random.uniform(1, 3))

    def contrast(self, img):
        random.seed()
        enhance = ImageEnhance.Contrast(img)
        return enhance.enhance(random.uniform(0.5, 3))

    def color_jitter(self, img):
        random.seed()
        img_hsv = img.convert('HSV')
        np_img = np.array(img_hsv)

        # Izvući kanale
        h = np_img[..., 0].astype(np.int16)
        s = np_img[..., 1].astype(np.float32)
        v = np_img[..., 2]

        # Primeni pomeraj i skaliranje
        hue_shift = random.randint(-10, 10)
        sat_scale = random.uniform(0.7, 1.3)

        h = (h + hue_shift) % 256
        s = np.clip(s * sat_scale, 0, 255)

        # Rekombinuj i konvertuj nazad
        np_img[..., 0] = h.astype(np.uint8)
        np_img[..., 1] = s.astype(np.uint8)

        jittered = Image.fromarray(np_img, 'HSV').convert('RGB')
        return jittered
    def gaussian_noise(self, img):
        random.seed()
        np_img = np.array(img).astype(np.float32)
        noise = np.random.normal(5, 70, np_img.shape)
        np_img += noise
        np_img = np.clip(np_img, 0, 255).astype(np.uint8)
        return Image.fromarray(np_img)

    def blur(self, img):
        random.seed()
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 3)))

    def crop_resize(self, img):
        random.seed()
        w, h = img.size
        crop_size = random.uniform(0.5, 0.85)
        left = int((1 - crop_size) * w * random.random())
        upper = int((1 - crop_size) * h * random.random())
        right = left + int(w * crop_size)
        lower = upper + int(h * crop_size)
        cropped = img.crop((left, upper, right, lower))
        return cropped.resize((w, h))

    def zoom(self, img):
        random.seed()
        w, h = img.size
        zoom_factor = random.uniform(0.55, 2.15)
        new_w = int(w * zoom_factor)
        new_h = int(h * zoom_factor)
        img_zoomed = img.resize((new_w, new_h), resample=Image.BICUBIC)
        if zoom_factor > 1:
            # central crop
            left = (new_w - w) // 2
            top = (new_h - h) // 2
            img_zoomed = img_zoomed.crop((left, top, left + w, top + h))
        else:
            # central pad
            pad_w = (w - new_w) // 2
            pad_h = (h - new_h) // 2
            img_zoomed = ImageOps.expand(img_zoomed, border=(pad_w, pad_h), fill='black')
        return img_zoomed


img_path = '../../data/ISIC_0052212.jpg'
img = Image.open(img_path)

# Inicijalizuj klasu
augmentor = Augmentation()

# Pokupi sve metode augmentacije (ručno ih pozivamo redom)
augmented_images = [
    ("Original", img),
    ("Color jitter", augmentor.color_jitter(img)),
    ("Brightness", augmentor.brightness(img)),
    ("Contrast", augmentor.contrast(img)),
    ("Gaussian noise", augmentor.gaussian_noise(img)),
    ("Blur", augmentor.blur(img)),
    ("CropResize", augmentor.crop_resize(img)),
    ("Zoom", augmentor.zoom(img))
]

# Prikaz slika
plt.figure(figsize=(16, 10))
for i, (title, aug_img) in enumerate(augmented_images):
    plt.subplot(2, 4, i + 1)
    plt.imshow(aug_img)
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.show()