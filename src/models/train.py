import os
import yaml
import time
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
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

        # Opciono: gradient clipping da sprečimo eksploziju gradijenata
        if 'grad_clip_value' in config:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip_value'])

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


def save_metrics_plot(train_losses, val_losses, val_metrics, save_path):
    """Funkcija za crtanje i čuvanje grafika metrika tokom treninga"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot loss
    axes[0, 0].plot(train_losses, label='Train Loss')
    axes[0, 0].plot(val_losses, label='Validation Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()

    # Plot accuracy
    axes[0, 1].plot([metrics['accuracy'] for metrics in val_metrics], label='Accuracy')
    axes[0, 1].plot([metrics['balanced_acc'] for metrics in val_metrics], label='Balanced Accuracy')
    axes[0, 1].set_title('Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()

    # Plot precision/recall
    axes[1, 0].plot([metrics['precision'] for metrics in val_metrics], label='Precision')
    axes[1, 0].plot([metrics['recall'] for metrics in val_metrics], label='Recall')
    axes[1, 0].plot([metrics['specificity'] for metrics in val_metrics], label='Specificity')
    axes[1, 0].set_title('Precision, Recall, and Specificity')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].legend()

    # Plot F1 and AUC
    axes[1, 1].plot([metrics['f1'] for metrics in val_metrics], label='F1 Score')
    axes[1, 1].plot([metrics['auc'] for metrics in val_metrics], label='AUC')
    axes[1, 1].set_title('F1 Score and AUC')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    args = parse_args()
    global config
    config = load_config(args.config)

    # Kreiranje output direktorijuma
    os.makedirs(config['output_dir'], exist_ok=True)

    # Čuvanje konfiguracije eksperimenta
    experiment_time = time.strftime("%Y%m%d-%H%M%S")
    experiment_dir = os.path.join(config['output_dir'], f"{experiment_time}_{config['model_type']}")
    os.makedirs(experiment_dir, exist_ok=True)

    # Kopiranje konfiguracionog fajla u eksperiment direktorijum
    with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

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
        criterion = FocalLoss(gamma=config.get('focal_gamma', 2.0))
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

    # Learning Rate Scheduler
    scheduler_type = config.get('scheduler', None)
    if scheduler_type == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.get('lr_factor', 0.5),
            patience=config.get('lr_patience', 3),
            verbose=True
        )
        print(f"Using ReduceLROnPlateau scheduler with patience {config.get('lr_patience', 3)}")
    elif scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config['epochs'],
            eta_min=config.get('min_lr', 1e-6)
        )
        print(f"Using CosineAnnealingLR scheduler with T_max={config['epochs']}")
    else:
        scheduler = None
        print("No learning rate scheduler")

    # Trening
    best_val_loss = float('inf')
    best_val_metric = 0.0  # Za praćenje metrike koju želimo da maksimizujemo
    metric_to_monitor = config.get('monitor_metric', 'loss')  # Metrika za early stopping
    patience = config.get('patience', 10)  # Broj epoha bez poboljšanja pre early stopping-a
    patience_counter = 0

    # Za praćenje istorije treninga
    train_losses = []
    val_losses = []
    val_metrics_history = []

    best_model_path = os.path.join(experiment_dir, 'best_model.pt')

    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")

        # Trening
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

        # Validacija
        val_metrics = validate(model, val_loader, criterion, device)
        val_loss = val_metrics['loss']

        print(f"Val Loss: {val_loss:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")
        print(
            f"Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}")
        print(f"Specificity: {val_metrics['specificity']:.4f}, Balanced Acc: {val_metrics['balanced_acc']:.4f}")

        # Ažuriranje istorije
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_metrics_history.append(val_metrics)

        # Ažuriranje learning rate scheduler-a
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)  # Za ReduceLROnPlateau
            else:
                scheduler.step()  # Za ostale scheduler-e

        # Određivanje vrednosti koja se prati za early stopping
        if metric_to_monitor == 'loss':
            current_metric = -val_loss  # Negativna jer želimo da minimizujemo loss
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
        else:
            # Za metrike koje maksimizujemo (accuracy, f1, itd.)
            current_metric = val_metrics.get(metric_to_monitor, 0.0)
            is_best = current_metric > best_val_metric
            if is_best:
                best_val_metric = current_metric

        # Čuvanje najboljeg modela
        if is_best:
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model with {metric_to_monitor}: {current_metric:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement in {metric_to_monitor} for {patience_counter} epochs")

        # Crtanje i čuvanje grafika metrika nakon svake epohe
        save_metrics_plot(
            train_losses,
            val_losses,
            val_metrics_history,
            os.path.join(experiment_dir, 'training_metrics.png')
        )

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement")
            break

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
    with open(os.path.join(experiment_dir, 'test_results.json'), 'w') as f:
        json.dump(test_metrics, f, indent=4)

    print(f"Model training complete. Best model saved to {best_model_path}")


if __name__ == "__main__":
    main()