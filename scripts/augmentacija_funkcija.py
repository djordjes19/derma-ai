import pandas as pd
import os
import random
from PIL import Image


def augment_images(df, data_dir, num_generated_per_image, images_to_augment, augmentations, prefix='aug_'):
    """
    Augmentuje slike i ažurira DataFrame sa novim redovima.

    :param df: Pandas DataFrame koji sadrži informacije o originalnim slikama (mora imati kolonu 'image_name')
    :param data_dir: Putanja do direktorijuma sa slikama
    :param num_generated_per_image: Broj augmentacija koje treba generisati po slici
    :param images_to_augment: Lista imena slika koje treba augmentovati (bez ekstenzije)
    :param augmentations: Lista funkcija za augmentaciju
    :param prefix: Prefiks za nazive novih augmentovanih slika
    :return: DataFrame sa dodatim redovima za nove augmentovane slike
    """
    augmented_data = []

    for original_image_name in images_to_augment:
        img_path = os.path.join(data_dir, f"{original_image_name}.jpg")
        original_image = Image.open(img_path).convert('RGB')
        filtered_rows = df[df['image_name'] == original_image_name]
        if filtered_rows.empty:
            print(f"[UPOZORENJE] Slika '{original_image_name}' nije pronađena u DataFrame-u.")
            continue  # ili nastavi sa sledećom slikom
        original_row = filtered_rows.iloc[0].copy()

        for i in range(num_generated_per_image):
            chosen_augmentation = random.choice(augmentations)
            augmented_image = original_image.copy()
            augmented_image = chosen_augmentation(augmented_image, **chosen_augmentation.__defaults__ or {})

            new_image_name = f"{prefix}{original_image_name}_{i}"
            new_image_path = os.path.join(data_dir, f"{new_image_name}.jpg")
            augmented_image.save(new_image_path)

            new_row = original_row.copy()
            new_row['image_name'] = new_image_name
            augmented_data.append(new_row)

    augmented_df = pd.DataFrame(augmented_data)
    return pd.concat([df, augmented_df], ignore_index=True)