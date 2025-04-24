import os
import yaml
import time
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from dataset import load_data
from model import get_model
from focal_loss import FocalLoss


def parse_args():
    parser = argparse.ArgumentParser(description='Trening modela za klasifikaciju melanoma')
    parser.add_argument('--config', type=str, default='config.yaml', help='Putanja do konfiguracionog fajla')
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for inputs, targets in tqdm(data_loader, desc='Trening'):
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass i optimizacija
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistika
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    # Metriks
    epoch_loss = running_loss / len(data_loader.dataset)
    accuracy = accuracy_score(all_targets, all_preds)

    return epoch_loss, accuracy


def validate(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc='Validacija'):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Statistika
            running_loss += loss.item() * inputs.size(0)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, preds = torch.max(outputs, 1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Metriks
    epoch_loss = running_loss / len(data_loader.dataset)
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)

    try:
        auc = roc_auc_score(all_targets, all_probs)
    except:
        auc = 0.0

    # Konfuziona matrica za specificity
    tn, fp, fn, tp = confusion_matrix(all_targets, all_preds).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = recall  # isto što i recall

    # Balanced accuracy
    balanced_acc = (sensitivity + specificity) / 2

    return {
        'loss': epoch_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'specificity': specificity,
        'balanced_acc': balanced_acc
    }


def main():
    args = parse_args()
    config = load_config(args.config)

    # Kreiranje output direktorijuma
    os.makedirs(config['output_dir'], exist_ok=True)

    # Postavljanje uređaja
    device = torch.device('cuda' if config['use_cuda'] and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Učitavanje podataka
    train_loader, val_loader, test_loader, class_weights = load_data(config)

    # Kreiranje modela
    model = get_model(config)
    model = model.to(device)

    # Loss function - izbor na osnovu konfiguracije
    if config.get('loss_type', 'cross_entropy') == 'focal':
        criterion = FocalLoss(alpha=class_weights, gamma=config.get('focal_gamma', 2.0))
        print("Using Focal Loss")
    elif config.get('use_class_weights', False) and class_weights is not None:
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"Using CrossEntropyLoss with class weights: {class_weights}")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using standard CrossEntropyLoss")

    # Optimizer
    optimizer = Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 0)
    )

    # Trening
    best_val_loss = float('inf')
    best_model_path = os.path.join(config['output_dir'], 'best_model.pt')

    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")

        # Trening
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

        # Validacija
        val_metrics = validate(model, val_loader, criterion, device)

        print(f"Val Loss: {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")
        print(
            f"Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}")
        print(f"Specificity: {val_metrics['specificity']:.4f}, Balanced Acc: {val_metrics['balanced_acc']:.4f}")

        # Čuvanje najboljeg modela
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model with val loss: {best_val_loss:.4f}")

    # Učitavanje najboljeg modela za testiranje
    model.load_state_dict(torch.load(best_model_path))

    # Testiranje
    print("\nEvaluating on test set:")
    test_metrics = validate(model, test_loader, criterion, device)

    print("\nTest Results:")
    print(f"Loss: {test_metrics['loss']:.4f}, Accuracy: {test_metrics['accuracy']:.4f}")
    print(
        f"Precision: {test_metrics['precision']:.4f}, Recall: {test_metrics['recall']:.4f}, F1: {test_metrics['f1']:.4f}")
    print(f"AUC: {test_metrics['auc']:.4f}, Balanced Acc: {test_metrics['balanced_acc']:.4f}")
    print(f"Specificity: {test_metrics['specificity']:.4f}")

    # Čuvanje test metrika
    import json
    with open(os.path.join(config['output_dir'], 'test_results.json'), 'w') as f:
        json.dump(test_metrics, f, indent=4)

    print(f"Model training complete. Best model saved to {best_model_path}")


if __name__ == "__main__":
    main()