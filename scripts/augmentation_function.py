import pandas as pd
import os
import random
from PIL import Image


def augment_images(df, data_dir, num_generated_per_image, images_to_augment, augmentations, prefix='aug_'):
    """
    This function is used to augment images using augmentations.

    :param df: Pandas DataFrame which contains images and their labels (image_name column)
    :param data_dir: File path of directory which contains images
    :param num_generated_per_image: Number of augmented images to generate
    :param images_to_augment: List of image names to augment (without .jpg extension)
    :param augmentations: List of augmentations
    :param prefix: Prefix of augmented images
    :return: DataFrame with additional rows for every augmented image
    """
    augmented_data = []

    for original_image_name in images_to_augment:
        img_path = os.path.join(data_dir, f"{original_image_name}.jpg")
        original_image = Image.open(img_path).convert('RGB')
        filtered_rows = df[df['image_name'] == original_image_name]
        if filtered_rows.empty:
            print(f"[WARNING] Image '{original_image_name}' is not found in DataFrame.")
            continue
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