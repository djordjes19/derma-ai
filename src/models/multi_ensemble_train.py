import os
import yaml
import time
import argparse
import torch
import json
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from dataset import load_data
from model import get_model
from focal_loss import FocalLoss
from ensemble import create_multi_architecture_ensemble, MultiArchitectureEnsemble


def parse_args():
    parser = argparse.ArgumentParser(description='Multi-Architecture Ensemble trening')
    parser.add_argument('--config', type=str, default='config.yaml', help='Osnovna konfiguracija')
    parser.add_argument('--architectures', nargs='+', default=['resnet50', 'efficientnet_b0', 'convnext_tiny'],
                        help='Lista arhitektura za ensemble')
    parser.add_argument('--output_dir', type=str, default='./multi_ensemble', help='Direktorijum za čuvanje modela')
    parser.add_argument('--ensemble_method', type=str, default='average',
                        choices=['average', 'weighted', 'max', 'vote'], help='Metod ensembling-a')
    parser.add_argument('--epochs', type=int, default=None, help='Broj epoha za treniranje (opciono)')
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_single_architecture(config, architecture, output_dir, device):
    """Treniranje pojedinačne arhitekture"""
    print(f"\n{'=' * 50}")
    print(f"Training {architecture} model")
    print(f"{'=' * 50}")

    # Kreiranje direktorijuma za model
    model_dir = os.path.join(output_dir, architecture)
    os.makedirs(model_dir, exist_ok=True)

    # Kreiranje konfiguracije za ovu arhitekturu
    arch_config = config.copy()
    arch_config['model_type'] = architecture

    # Čuvanje konfiguracije
    with open(os.path.join(model_dir, 'config.yaml'), 'w') as f:
        yaml.dump(arch_config, f)

    # Učitavanje podataka
    train_loader, val_loader, test_loader, class_weights = load_data(arch_config)

    # Kreiranje modela
    model = get_model(arch_config)
    model = model.to(device)

    # Loss function
    if arch_config.get('loss_type', 'cross_entropy') == 'focal':
        criterion = FocalLoss(gamma=arch_config.get('focal_gamma', 2.0))
        print("Using Focal Loss")
    elif arch_config.get('use_class_weights', False) and class_weights is not None:
        class_weights = class_weights.to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        print(f"Using CrossEntropyLoss with class weights")
    else:
        criterion = torch.nn.CrossEntropyLoss()
        print("Using standard CrossEntropyLoss")

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=arch_config['learning_rate'],
        weight_decay=arch_config.get('weight_decay', 0.0001)
    )

    # Learning Rate Scheduler
    scheduler_type = arch_config.get('scheduler', None)
    if scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=arch_config.get('lr_factor', 0.5),
            patience=arch_config.get('lr_patience', 3),
            verbose=True
        )
    elif scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=arch_config['epochs'],
            eta_min=arch_config.get('min_lr', 1e-6)
        )
    else:
        scheduler = None

    # Varijable za trening
    best_val_metric = 0.0 if arch_config.get('monitor_metric', 'loss') != 'loss' else float('inf')
    patience_counter = 0
    train_losses = []
    val_metrics_history = []

    best_model_path = os.path.join(model_dir, 'best_model.pt')
    use_mixed_precision = arch_config.get('use_mixed_precision', False)

    # Broj epoha
    epochs = arch_config['epochs']

    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Trening
        model.train()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        # Postavljanje scaler-a za mixed precision
        scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None

        for inputs, targets in tqdm(train_loader, desc='Train'):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            if use_mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                # Backward pass i optimizacija
                optimizer.zero_grad()
                scaler.scale(loss).backward()

                # Gradient clipping
                if 'grad_clip_value' in arch_config:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), arch_config['grad_clip_value'])

                scaler.step(optimizer)
                scaler.update()
            else:
                # Standardni forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # Backward pass i optimizacija
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                if 'grad_clip_value' in arch_config:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), arch_config['grad_clip_value'])

                optimizer.step()

            # Statistika
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

        # Računanje metrika
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = accuracy_score(all_targets, all_preds)
        train_losses.append(train_loss)

        print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

        # Validacija
        val_metrics = validate(model, val_loader, criterion, device)
        val_loss = val_metrics['loss']
        val_metrics_history.append(val_metrics)

        print(f"Val Loss: {val_loss:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")
        print(
            f"Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}")
        print(f"Specificity: {val_metrics['specificity']:.4f}, Balanced Acc: {val_metrics['balanced_acc']:.4f}")

        # Ažuriranje learning rate scheduler-a
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Early stopping i čuvanje najboljeg modela
        metric_to_monitor = arch_config.get('monitor_metric', 'loss')

        if metric_to_monitor == 'loss':
            current_metric = val_loss
            is_better = current_metric < best_val_metric
        else:
            current_metric = val_metrics.get(metric_to_monitor, 0.0)
            is_better = current_metric > best_val_metric

        if is_better:
            best_val_metric = current_metric
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model with {metric_to_monitor}: {current_metric:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement in {metric_to_monitor} for {patience_counter} epochs")

        # Early stopping
        if patience_counter >= arch_config.get('patience', 10):
            print(f"Early stopping triggered after {patience_counter} epochs without improvement")
            break

    # Crtanje metrika
    plot_training_metrics(train_losses, val_metrics_history, model_dir)

    # Učitavanje najboljeg modela i evaluacija na test setu
    model.load_state_dict(torch.load(best_model_path))
    test_metrics = validate(model, test_loader, criterion, device)

    # Čuvanje test metrika
    with open(os.path.join(model_dir, 'test_metrics.json'), 'w') as f:
        json.dump(test_metrics, f, indent=4)

    print(f"\nTest Results for {architecture}:")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1']:.4f}")
    print(f"AUC: {test_metrics['auc']:.4f}, Balanced Acc: {test_metrics['balanced_acc']:.4f}")

    return best_model_path, test_metrics


def validate(model, data_loader, criterion, device):
    """Validacija modela"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc='Validation'):
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

    # Metrike
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


def plot_training_metrics(train_losses, val_metrics, save_dir):
    """Funkcija za crtanje i čuvanje grafika metrika tokom treninga"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot loss
    axes[0, 0].plot(train_losses, label='Train Loss')
    axes[0, 0].plot([metrics['loss'] for metrics in val_metrics], label='Validation Loss')
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
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'))
    plt.close()


def test_ensemble(ensemble, test_loader, device):
    """Test ensemble modela na test skupu"""
    ensemble.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='Testing Ensemble'):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = ensemble(inputs)

            # Statistika
            probs = outputs[:, 1]
            preds = torch.argmax(outputs, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Računanje metrika
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

    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'specificity': specificity,
        'balanced_acc': balanced_acc
    }

    return results


def train_multi_ensemble(config, args):
    """Trening više arhitektura i kreiranje ensemble modela"""
    # Kreiranje direktorijuma za multi ensemble
    experiment_time = time.strftime("%Y%m%d-%H%M%S")
    ensemble_dir = os.path.join(args.output_dir, f"{experiment_time}_multi_ensemble")
    os.makedirs(ensemble_dir, exist_ok=True)

    # Čuvanje konfiguracije
    with open(os.path.join(ensemble_dir, 'base_config.yaml'), 'w') as f:
        yaml.dump(config, f)

    # Postavljanje uređaja
    device = torch.device('cuda' if config['use_cuda'] and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Čuvanje opcija za ensemble
    ensemble_options = {
        'architectures': args.architectures,
        'ensemble_method': args.ensemble_method,
        'output_dir': ensemble_dir
    }

    with open(os.path.join(ensemble_dir, 'ensemble_options.json'), 'w') as f:
        json.dump(ensemble_options, f, indent=4)

    # Ažuriranje broja epoha ako je prosleđeno kroz argumente
    if args.epochs is not None:
        config['epochs'] = args.epochs

    # Treniranje svake arhitekture
    model_paths = []
    config_paths = []
    test_metrics = []

    for architecture in args.architectures:
        print(f"\nTraining {architecture} model...")

        # Treniranje modela pojedinačne arhitekture
        model_path, metrics = train_single_architecture(config, architecture, ensemble_dir, device)

        # Čuvanje putanja i metrika
        model_paths.append(model_path)
        config_paths.append(os.path.join(ensemble_dir, architecture, 'config.yaml'))
        test_metrics.append({
            'architecture': architecture,
            'metrics': metrics
        })

    # Kreiranje multi-architecture ensemble modela
    create_multi_architecture_ensemble(
        config_paths,
        model_paths,
        ensemble_dir,
        ensemble_method=args.ensemble_method
    )

    # Učitavanje ensemble modela za testiranje
    ensemble_config_path = os.path.join(ensemble_dir, 'multi_architecture_ensemble_config.json')
    ensemble = MultiArchitectureEnsemble(
        args.architectures,
        model_paths,
        method=args.ensemble_method
    )
    ensemble = ensemble.to(device)

    # Učitavanje podataka za testiranje
    _, _, test_loader, _ = load_data(config)

    # Testiranje ensemble modela
    print("\nTesting Multi-Architecture Ensemble model...")
    ensemble_metrics = test_ensemble(ensemble, test_loader, device)

    # Čuvanje metrika ensemble modela
    with open(os.path.join(ensemble_dir, 'ensemble_metrics.json'), 'w') as f:
        json.dump(ensemble_metrics, f, indent=4)

    # Ispis rezultata
    print(f"\nEnsemble Results:")
    print(f"Accuracy: {ensemble_metrics['accuracy']:.4f}, F1: {ensemble_metrics['f1']:.4f}")
    print(f"AUC: {ensemble_metrics['auc']:.4f}, Balanced Acc: {ensemble_metrics['balanced_acc']:.4f}")

    # Kreiranje barchart poređenja
    create_comparison_chart(test_metrics, ensemble_metrics, ensemble_dir)

    return ensemble_metrics, model_paths, test_metrics


def create_comparison_chart(model_metrics, ensemble_metrics, save_dir):
    """Kreiranje grafikona za poređenje performansi modela"""
    metrics_to_plot = ['accuracy', 'f1', 'auc', 'balanced_acc']
    labels = ['Accuracy', 'F1 Score', 'AUC', 'Balanced Acc']

    # Priprema podataka
    architectures = [m['architecture'] for m in model_metrics]
    x = np.arange(len(metrics_to_plot))
    width = 0.15  # Širina barova

    # Kreiranje grafikona
    plt.figure(figsize=(12, 8))

    # Crtanje pojedinačnih modela
    for i, model_metric in enumerate(model_metrics):
        values = [model_metric['metrics'][metric] for metric in metrics_to_plot]
        plt.bar(x + i * width, values, width, label=model_metric['architecture'])

    # Crtanje ensemble modela
    ensemble_values = [ensemble_metrics[metric] for metric in metrics_to_plot]
    plt.bar(x + len(model_metrics) * width, ensemble_values, width, label='Ensemble', color='red')

    # Dodavanje oznaka
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x + width * (len(model_metrics) / 2), labels)
    plt.legend(loc='lower right')
    plt.ylim(0, 1.0)

    # Dodavanje vrednosti iznad barova
    for i, arch in enumerate(architectures + ['Ensemble']):
        values = [model_metrics[i]['metrics'][m] for m in metrics_to_plot] if i < len(
            architectures) else ensemble_values
        for j, v in enumerate(values):
            plt.text(j + i * width, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8, rotation=90)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_comparison.png'))
    plt.close()


def main():
    args = parse_args()
    config = load_config(args.config)

    # Kreiranje osnovnog direktorijuma
    os.makedirs(args.output_dir, exist_ok=True)

    # Trening i kreiranje ensemble modela
    train_multi_ensemble(config, args)


if __name__ == "__main__":
    main()