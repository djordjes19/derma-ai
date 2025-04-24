import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split


class MelanomaDataset(Dataset):
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def get_transforms(img_size):
    """Kreiranje osnovnih transformacija za slike"""
    # Pretpostavljamo da je augmentacija već urađena, samo normalizacija i resize
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, val_transform


def load_data(config):
    """Učitavanje podataka i priprema data loadera"""
    # Učitavanje CSV fajla za trening/validaciju
    df_train = pd.read_csv(config['csv_file'])

    # Kreiranje putanja do slika i oznaka za trening/validaciju
    train_val_paths = [os.path.join(config['data_dir'], img_name) for img_name in df_train['image_name']]
    train_val_labels = df_train['target'].values

    # Podela na trening i validaciju
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, test_size=config.get('val_size', 0.2),
        stratify=train_val_labels, random_state=42
    )

    # Učitavanje odvojenog test seta ako je definisan
    if 'test_csv_file' in config and config['test_csv_file']:
        df_test = pd.read_csv(config['test_csv_file'])
        test_paths = [os.path.join(config['test_dir'], img_name) for img_name in df_test['image_name']]
        test_labels = df_test['target'].values
    else:
        # Ako nema odvojenog test seta, koristimo deo validacionog
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            val_paths, val_labels, test_size=0.5, stratify=val_labels, random_state=42
        )

    # Kreiranje transformacija
    train_transform, val_transform = get_transforms(config['img_size'])

    # Kreiranje datasetova
    train_dataset = MelanomaDataset(train_paths, train_labels, train_transform)
    val_dataset = MelanomaDataset(val_paths, val_labels, val_transform)
    test_dataset = MelanomaDataset(test_paths, test_labels, val_transform)

    # Kreiranje data loadera
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )

    # Računanje težina klasa ako je potrebno
    if config.get('use_class_weights', False):
        class_counts = pd.Series(train_labels).value_counts().sort_index()
        class_weights = torch.FloatTensor(len(class_counts))

        for i in range(len(class_counts)):
            class_weights[i] = len(train_labels) / (class_counts[i] * len(class_counts))
    else:
        class_weights = None

    return train_loader, val_loader, test_loader, class_weights