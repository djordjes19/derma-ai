"""
Fair Melanoma Detection - Training

Script used for training melanoma classification models.
Implemented k-fold validation and threshold optimization.
"""

import os
import yaml
import torch
import numpy as np
import json
import argparse
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from datetime import datetime

from src.dataset import load_data
from src.model import get_model
from src.utils import (
    calculate_metrics, optimize_threshold, calculate_fairness_metrics,
    plot_metrics_by_group, save_validation_predictions, plot_precision_recall_curve
)
from src.ensemble import EnsembleModel, ThresholdOptimizedEnsemble, save_ensemble


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Fair Melanoma Detection Training')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Putanja do konfiguracionog fajla')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Direktorijum za rezultate (ako None, kreira se na osnovu vremena)')
    return parser.parse_args()


def load_config(config_path):
    """Loads config file."""

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_epoch(model, data_loader, criterion, optimizer, device, scheduler=None, clip_value=None):
    """Trains the model for one epoch."""

    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for batch in tqdm(data_loader, desc="Training"):
        # Load data
        if len(batch) == 3:  # With metadata
            inputs, targets, _ = batch
        else:
            inputs, targets = batch

        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward i optimization
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping if specified
        if clip_value:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        optimizer.step()

        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step()

    # Metrics calculation
    train_loss = running_loss / len(data_loader.dataset)
    metrics = calculate_metrics(all_targets, all_preds)
    metrics['loss'] = train_loss

    return metrics


def validate(model, data_loader, criterion, device):
    """Evaluates model na validation set."""

    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []
    all_image_names = []
    group_data = {}

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validation"):
            # Load data
            if len(batch) == 3:  # With metadata
                inputs, targets, metadata = batch

                # Saving metadata for fairness analysis
                for key in metadata:
                    if key not in group_data:
                        group_data[key] = []
                    group_data[key].extend(metadata[key])
            else:
                inputs, targets = batch
                metadata = None

            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, preds = torch.max(outputs, 1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            if metadata is not None and 'image_name' in metadata:
                all_image_names.extend(metadata['image_name'])

    # Metrics calculation
    val_loss = running_loss / len(data_loader.dataset)
    metrics = calculate_metrics(all_targets, all_preds, all_probs)
    metrics['loss'] = val_loss

    # Analizing fairness if metadata is provided
    group_metrics = {}

    for key in group_data:
        if key in ['skin_tone', 'skin_type', 'ethnicity', 'sex', 'age_group']:
            unique_values = np.unique(group_data[key])

            for value in unique_values:
                group_indices = np.array(group_data[key]) == value
                if np.sum(group_indices) > 10:
                    group_name = f"{key}_{value}"
                    group_metrics[group_name] = calculate_metrics(
                        np.array(all_targets)[group_indices],
                        np.array(all_preds)[group_indices],
                        np.array(all_probs)[group_indices]
                    )

    return metrics, all_probs, all_targets, all_image_names, group_metrics


def train_model(config, architecture, fold=None, train_loader=None, val_loader=None, class_weights=None):
    """
    Trains the model of specified architecture.

    Args:
        config (dict): Configuration dictionary.
        architecture (str): Architecture type.
        fold (int, optional): Number of folds to use. Defaults to None.
        train_loader, val_loader: Training and validation loaders.

    Returns:
        tuple: (model, best_metrics, val_probs, val_targets, optimal_threshold)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} for {architecture}")

    # Create model
    model = get_model(
        model_type=architecture,
        pretrained=config['model']['pretrained'],
        dropout=config['model']['dropout']
    )
    model = model.to(device)

    # Define loss function
    if config['training'].get('use_class_weights', False) and class_weights is not None:
        class_weights = class_weights.to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        print(f"Using weighted CrossEntropyLoss with weights: {class_weights}")
    else:
        criterion = torch.nn.CrossEntropyLoss()
        print("Using standard CrossEntropyLoss")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Learning rate scheduler
    scheduler_type = config['training'].get('scheduler', None)
    if scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config['training'].get('lr_factor', 0.5),
            patience=config['training'].get('lr_patience', 3),
            verbose=True
        )
    elif scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs'],
            eta_min=config['training'].get('min_lr', 1e-6)
        )
    else:
        scheduler = None

    # Load data
    if train_loader is None or val_loader is None:
        train_loader, val_loader, class_weights = load_data(config)

    # Parameters for early stopping
    best_val_metric = float('inf') if config['training'].get('monitor_metric', 'loss') == 'loss' else 0
    best_val_metrics = None
    best_val_probs = None
    best_val_targets = None
    best_val_image_names = None
    patience_counter = 0

    # Model training
    for epoch in range(config['training']['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['training']['epochs']}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device,
            scheduler=None,
            clip_value=config['training'].get('grad_clip_value', None)
        )

        # Validation
        val_metrics, val_probs, val_targets, val_image_names, group_metrics = validate(
            model, val_loader, criterion, device
        )

        print(f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, "
              f"Recall: {train_metrics['recall']:.4f}")

        print(f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
              f"Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}")

        # Fairness analysis
        if group_metrics:
            fairness = calculate_fairness_metrics(group_metrics)
            print(f"Fairness - Recall Disparity: {fairness['recall_disparity']:.4f}, "
                  f"Precision Ratio: {fairness['precision_ratio']:.4f}")

        # Update learning rate scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()

        # Early stopping check
        monitor_metric = config['training'].get('monitor_metric', 'loss')
        current_metric = val_metrics['loss'] if monitor_metric == 'loss' else val_metrics[monitor_metric]

        if (monitor_metric == 'loss' and current_metric < best_val_metric) or \
                (monitor_metric != 'loss' and current_metric > best_val_metric):
            best_val_metric = current_metric
            best_val_metrics = val_metrics
            best_val_probs = val_probs
            best_val_targets = val_targets
            best_val_image_names = val_image_names

            # Save best model
            model_dir = os.path.join(config['data']['output_dir'], architecture)
            os.makedirs(model_dir, exist_ok=True)

            model_suffix = f"_fold{fold}" if fold is not None else ""
            model_path = os.path.join(model_dir, f"best_model{model_suffix}.pt")
            torch.save(model.state_dict(), model_path)

            print(f"Saved best model with {monitor_metric}: {current_metric:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement in {monitor_metric} for {patience_counter} epochs")

        # Early stopping
        if patience_counter >= config['training'].get('early_stopping_patience', 7):
            print(f"Early stopping triggered after {patience_counter} epochs without improvement")
            break

    # Load best model
    model_dir = os.path.join(config['data']['output_dir'], architecture)
    model_suffix = f"_fold{fold}" if fold is not None else ""
    model_path = os.path.join(model_dir, f"best_model{model_suffix}.pt")
    model.load_state_dict(torch.load(model_path))

    # Optimize threshold for recall
    if best_val_probs is not None and best_val_targets is not None:
        optimal_threshold = optimize_threshold(
            best_val_targets,
            best_val_probs,
            target_recall=config['ensemble'].get('target_recall', 0.9),
            min_precision=config['ensemble'].get('min_precision', 0.6)
        )

        # Calculate new metrics
        threshold_preds = (np.array(best_val_probs) >= optimal_threshold).astype(int)
        threshold_metrics = calculate_metrics(best_val_targets, threshold_preds, best_val_probs)

        print(f"\nOptimalni threshold: {optimal_threshold:.4f}")
        print(f"Metrike sa optimalnim threshold-om:")
        print(f"Precision: {threshold_metrics['precision']:.4f}, Recall: {threshold_metrics['recall']:.4f}, "
              f"F1: {threshold_metrics['f1']:.4f}, Balanced Acc: {threshold_metrics['balanced_acc']:.4f}")

        # Plot precision-recall curve
        plot_dir = os.path.join(config['data']['output_dir'], architecture, 'plots')
        os.makedirs(plot_dir, exist_ok=True)

        plot_suffix = f"_fold{fold}" if fold is not None else ""
        pr_curve_path = os.path.join(plot_dir, f"precision_recall_curve{plot_suffix}.png")
        plot_precision_recall_curve(best_val_targets, best_val_probs, pr_curve_path, optimal_threshold)
    else:
        optimal_threshold = 0.5

    return model, best_val_metrics, best_val_probs, best_val_targets, best_val_image_names, optimal_threshold


def train_with_kfold(config):
    """
    Trains models using k-fold cross validation.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        tuple: (models, metrics, thresholds)
    """
    # Load data
    df = pd.read_csv(config['data']['csv_file'])

    # Create combined variable for stratification
    if 'skin_tone' in df.columns:
        df['stratify_key'] = df['target'].astype(str) + '_' + df['skin_tone']
    else:
        df['stratify_key'] = df['target']

    # Stratified k-fold
    n_splits = config['kfold']['n_splits']
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=config['kfold'].get('shuffle', True),
        random_state=config['kfold'].get('random_state', 42)
    )

    # Lists for saving results
    models = []
    all_metrics = []
    all_thresholds = []
    all_val_probs = []
    all_val_targets = []
    all_val_image_names = []

    # For every architecture
    for architecture in config['model']['architectures']:
        print(f"\n{'=' * 50}")
        print(f"Training {architecture} with {n_splits}-fold cross validation")
        print(f"{'=' * 50}")

        architecture_models = []
        architecture_metrics = []
        architecture_thresholds = []
        architecture_val_probs = {}
        architecture_val_targets = {}
        architecture_val_image_names = {}

        # For every fold
        for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['stratify_key'])):
            print(f"\nFold {fold + 1}/{n_splits}")

            # Create data loader
            fold_config = config.copy()
            train_loader, val_loader, class_weights = load_data(config, fold_indices=(train_idx, val_idx))

            # Train model
            model, metrics, val_probs, val_targets, val_image_names, threshold = train_model(
                config, architecture, fold=fold,
                train_loader=train_loader, val_loader=val_loader,
                class_weights=class_weights
            )

            # Save results
            architecture_models.append(model)
            architecture_metrics.append(metrics)
            architecture_thresholds.append(threshold)

            # Save fold predictions
            for i, img_name in enumerate(val_image_names):
                architecture_val_probs[img_name] = val_probs[i]
                architecture_val_targets[img_name] = val_targets[i]
                architecture_val_image_names[img_name] = val_image_names[i]

        # Append results
        models.append(architecture_models)
        all_metrics.append(architecture_metrics)
        all_thresholds.append(architecture_thresholds)
        all_val_probs.append(architecture_val_probs)
        all_val_targets.append(architecture_val_targets)
        all_val_image_names.append(architecture_val_image_names)

        # Create validation_output.csv
        create_validation_output(architecture, architecture_val_image_names,
                                 architecture_val_probs, architecture_val_targets,
                                 architecture_thresholds, config)

    return models, all_metrics, all_thresholds, all_val_probs, all_val_targets, all_val_image_names


def create_validation_output(architecture, val_image_names, val_probs, val_targets, thresholds, config):
    """
    Creates validation_output.csv for specified architecture.

    Args:
        architecture (str): Architecture type
        val_image_names (dict): Validation image names
        val_probs (dict): Validation probabilities
        val_targets (dict): Real targets
        thresholds (list): List of thresholds
        config (dict): Configuration dictionary
    """
    # Average threshold
    avg_threshold = sum(thresholds) / len(thresholds)

    # Sort keys
    image_names = sorted(val_image_names.keys())

    # Create lists
    image_names_list = []
    predictions_list = []
    probabilities_list = []

    for img_name in image_names:
        # Make predictions list
        prob = val_probs[img_name]
        pred = 1 if prob >= avg_threshold else 0

        image_names_list.append(img_name)
        predictions_list.append(pred)
        probabilities_list.append(prob)

    # Save to CSV
    output_dir = os.path.join(config['data']['output_dir'], architecture)
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, 'validation_output.csv')
    save_validation_predictions(image_names_list, predictions_list, probabilities_list, output_path)


def create_ensemble(config, models, thresholds):
    """
    Creates ensemble model based on pretrained models.

    Args:
        config (dict): Configuration dictionary
        models (list): Models list - [architecture][fold]
        thresholds (list): Threshold list - [architecture][fold]

    Returns:
        ThresholdOptimizedEnsemble: Ensemble model
    """
    # Combine models from all architectures and folds
    all_models = []
    all_thresholds = []
    models_info = []

    for arch_idx, architecture_models in enumerate(models):
        architecture = config['model']['architectures'][arch_idx]

        for fold_idx, model in enumerate(architecture_models):
            all_models.append(model)
            all_thresholds.append(thresholds[arch_idx][fold_idx])

            # Model information
            model_dir = os.path.join(config['data']['output_dir'], architecture)
            model_path = os.path.join(model_dir, f"best_model_fold{fold_idx}.pt")

            models_info.append({
                'architecture': architecture,
                'fold': fold_idx,
                'path': model_path
            })

    # Create ensemble
    ensemble_method = config['ensemble'].get('method', 'average')
    ensemble = ThresholdOptimizedEnsemble(
        all_models,
        all_thresholds,
        method=ensemble_method
    )

    # Save ensemble
    ensemble_dir = os.path.join(config['data']['output_dir'], 'ensemble')
    save_ensemble(ensemble, config, models_info, all_thresholds, ensemble_dir)

    return ensemble


def create_global_validation_output(all_val_image_names, all_val_probs, all_val_targets, config, ensemble=None):
    """
    Creates global validation_output.csv based on all predictions.

    Args:
        all_val_image_names (list): Image names
        all_val_probs (list): Probabilities
        all_val_targets (list): Real targets
        config (dict): Configuration dictionary
        ensemble (nn.Module, optional): Ensemble model
    """
    # Combine all dictionaries with image names
    image_names = set()
    for val_dict in all_val_image_names:
        image_names.update(val_dict.keys())

    image_names = sorted(list(image_names))

    # Create lists
    image_names_list = []
    predictions_list = []
    probabilities_list = []

    if ensemble is not None:
        pass
    else:
        # Average of all predictions
        for img_name in image_names:
            probs = []

            for val_dict in all_val_probs:
                if img_name in val_dict:
                    probs.append(val_dict[img_name])

            if probs:
                avg_prob = sum(probs) / len(probs)
                pred = 1 if avg_prob >= 0.5 else 0  # Default threshold 0.5

                image_names_list.append(img_name)
                predictions_list.append(pred)
                probabilities_list.append(avg_prob)

    # Save to CSV
    output_path = os.path.join(config['data']['output_dir'], 'validation_output.csv')
    save_validation_predictions(image_names_list, predictions_list, probabilities_list, output_path)


def main():
    """Main function for model training."""
    args = parse_args()
    config = load_config(args.config)

    # Create output directory
    if args.output_dir:
        config['data']['output_dir'] = args.output_dir
    elif not os.path.exists(config['data']['output_dir']):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config['data']['output_dir'] = os.path.join('./models', f"run_{timestamp}")

    os.makedirs(config['data']['output_dir'], exist_ok=True)

    # Save configuration
    with open(os.path.join(config['data']['output_dir'], 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    # Train model
    if config['kfold'].get('use_kfold', True):
        print("Using k-fold cross validation")
        models, metrics, thresholds, val_probs, val_targets, val_image_names = train_with_kfold(config)

        # Create ensemble
        if config['ensemble'].get('create_ensemble', True):
            ensemble = create_ensemble(config, models, thresholds)
            create_global_validation_output(val_image_names, val_probs, val_targets, config, ensemble)
        else:
            create_global_validation_output(val_image_names, val_probs, val_targets, config)
    else:
        # Train without k-fold
        all_models = []
        all_metrics = []
        all_thresholds = []

        for architecture in config['model']['architectures']:
            model, metrics, val_probs, val_targets, val_image_names, threshold = train_model(config, architecture)
            all_models.append([model])
            all_metrics.append([metrics])
            all_thresholds.append([threshold])

            # Create validation_output.csv
            output_dir = os.path.join(config['data']['output_dir'], architecture)
            os.makedirs(output_dir, exist_ok=True)

            predictions = (np.array(val_probs) >= threshold).astype(int)
            output_path = os.path.join(output_dir, 'validation_output.csv')
            save_validation_predictions(val_image_names, predictions, val_probs, output_path)

        # Create ensemble
        if config['ensemble'].get('create_ensemble', True):
            ensemble = create_ensemble(config, all_models, all_thresholds)

        # Create global validation_output.csv
        all_val_image_names = [{name: name for name in val_image_names} for val_image_names in [val_image_names]]
        all_val_probs = [{name: prob for name, prob in zip(val_image_names, val_probs)}]
        all_val_targets = [{name: target for name, target in zip(val_image_names, val_targets)}]

        create_global_validation_output(all_val_image_names, all_val_probs, all_val_targets, config)

    print(f"Training complete. Results saved to {config['data']['output_dir']}")


if __name__ == "__main__":
    main()