import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import numpy as np


class MelanomaDataset(Dataset):
    """Dataset used for classification of dermatoscopic images."""

    def __init__(self, img_paths, labels, transform=None, metadata=None):
        """
        Args:
            img_paths (list): List od image paths
            labels (list): List of labels (targets)
            transform (callable, optional): Transformations being performed on an image
            metadata (pandas.DataFrame, optional): Additional metadata for every image
        """
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform
        self.metadata = metadata

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Image loading
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        # Transformations
        if self.transform:
            image = self.transform(image)

        # Return metadata if possible
        if self.metadata is not None:
            metadata = {key: self.metadata.iloc[idx][key] for key in self.metadata.columns}
            return image, torch.tensor(label, dtype=torch.long), metadata

        return image, torch.tensor(label, dtype=torch.long)


def get_transforms(img_size, use_augmentation=False):
    """Creating transformations."""

    # Base transformations for validation
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Transformations for training with augmentation
    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        train_transform = val_transform

    return train_transform, val_transform


def load_data(config, fold_indices=None):
    """Data loading and creating of DataLoader ."""

    # Load CSV file
    df = pd.read_csv(config['data']['csv_file'])
    if not all(df['image_name'].str.endswith('.jpg')):
        df['image_name'] = df['image_name'].apply(lambda x: x + '.jpg' if not x.endswith('.jpg') else x)
    if fold_indices is None:
        if 'skin_tone' in df.columns:
            df['stratify_key'] = df['target'].astype(str) + '_' + df['skin_tone']
        else:
            df['stratify_key'] = df['target']

        # Stratified split
        from sklearn.model_selection import train_test_split
        train_df, valid_df = train_test_split(
            df,
            test_size=config['data'].get('valid_size', 0.2),
            stratify=df['stratify_key'],
            random_state=42
        )
    else:
        # K-fold validation
        train_idx, val_idx = fold_indices
        train_df = df.iloc[train_idx]
        valid_df = df.iloc[val_idx]

    # Paths and labels
    train_paths = [os.path.join(config['data']['data_dir'], img_name) for img_name in train_df['image_name']]
    train_labels = train_df['target'].values

    val_paths = [os.path.join(config['data']['data_dir'], img_name) for img_name in valid_df['image_name']]
    val_labels = valid_df['target'].values

    # Transformations
    train_transform, val_transform = get_transforms(config['model']['img_size'], use_augmentation=True)

    train_metadata = train_df.drop(['image_name', 'target'], axis=1) if len(train_df.columns) > 2 else None
    val_metadata = valid_df.drop(['image_name', 'target'], axis=1) if len(valid_df.columns) > 2 else None

    # Creating datasets
    train_dataset = MelanomaDataset(train_paths, train_labels, train_transform, train_metadata)
    val_dataset = MelanomaDataset(val_paths, val_labels, val_transform, val_metadata)

    # Calculating class weights
    if config['training'].get('use_class_weights', False):
        class_counts = np.bincount(train_labels)
        class_weights = torch.FloatTensor(len(class_counts))

        for i in range(len(class_counts)):
            class_weights[i] = len(train_labels) / (class_counts[i] * len(class_counts))
    else:
        class_weights = None

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training'].get('pin_memory', True)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training'].get('pin_memory', True)
    )

    return train_loader, val_loader, class_weights


class TestDataset(Dataset):
    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image


def get_test_loader(img_dir, img_size, batch_size=32, num_workers=4):
    """Creating test DataLoader for predictions.

    Args:
        img_dir (str): Directory with test images
        img_size (int): Image size
        batch_size (int): Batch size
        num_workers (int): Number of parallel workers

    Returns:
        tuple: (test_loader, image_names)
    """
    # Transformations
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    #Image paths and names
    image_paths = []
    image_names = []

    for filename in os.listdir(img_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_paths.append(os.path.join(img_dir, filename))
            image_names.append(filename)

    #Creating test dataset and Loader
    test_dataset = TestDataset(image_paths, transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return test_loader, image_names