import os
import torch
import numpy as np
import pandas as pd
import yaml
import json
from src.model import get_model
from src.ensemble import EnsembleModel
from src.utils import calculate_metrics, optimize_threshold, save_validation_predictions


def convert_to_python_types(obj):
    """Converts NumPy types to standard Python types for JSON serialization."""

    import numpy as np
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_to_python_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_python_types(value) for key, value in obj.items()}
    else:
        return obj


def create_ensemble_all_folds(config_path, output_dir=None):
    """Loads models from all folds and creates an ensemble model"""

    # Loading config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Defining paths
    if output_dir is None:
        output_dir = config['data']['output_dir']

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # List for saving models and information
    all_models = []
    all_thresholds = []
    models_info = []

    # Loading models from all folds
    n_splits = config['kfold'].get('n_splits', 5)

    for architecture in config['model']['architectures']:
        print(f"Loading {architecture} models from all folds...")
        model_dir = os.path.join(output_dir, architecture)

        for fold in range(n_splits):
            model_path = os.path.join(model_dir, f"best_model_fold{fold}.pt")
            if os.path.exists(model_path):
                # Creating model
                model = get_model(architecture, pretrained=False)
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()
                model = model.to(device)

                # Using default threshold 0.5
                threshold = 0.5

                # Add model to list
                all_models.append(model)
                all_thresholds.append(threshold)
                models_info.append({
                    'architecture': architecture,
                    'fold': fold,
                    'path': model_path
                })
                print(f"  Added {architecture} from fold {fold}")

    # Create ensemble model
    if all_models:
        ensemble_method = config['ensemble'].get('method', 'average')
        print(f"\nCreating ensemble with method: {ensemble_method} from {len(all_models)} models")
        ensemble = EnsembleModel(all_models, method=ensemble_method, thresholds=all_thresholds)

        # Save ensemble model
        ensemble_dir = os.path.join(output_dir, 'ensemble_all_folds')
        os.makedirs(ensemble_dir, exist_ok=True)

        # Save model state
        torch.save(ensemble.state_dict(), os.path.join(ensemble_dir, 'ensemble_model.pt'))

        # Create and save config
        ensemble_config = {
            'method': ensemble_method,
            'architectures': [info['architecture'] for info in models_info],
            'model_paths': [info['path'] for info in models_info],
            'thresholds': all_thresholds,
            'img_size': config['model']['img_size']
        }

        # Converting all values to standard Python types
        ensemble_config = convert_to_python_types(ensemble_config)

        # Save config
        with open(os.path.join(ensemble_dir, 'ensemble_config.json'), 'w') as f:
            json.dump(ensemble_config, f, indent=4)

        print(f"Ensemble saved to {ensemble_dir}")

        # Evaluation on validation set
        print("\nEvaluating ensemble on validation set...")
        metrics, all_preds, all_probs, all_targets, all_image_names = evaluate_ensemble(
            ensemble, config, ensemble_dir, device
        )

        # Print metrics
        print("\n========== ENSEMBLE METRICS (ALL FOLDS) ==========")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"AUC: {metrics.get('auc', 0):.4f}")
        print(f"Balanced Accuracy: {metrics['balanced_acc']:.4f}")
        print(f"Specificity: {metrics['specificity']:.4f}")
        print("================================================\n")

        # Optimizing threshold
        if config['ensemble'].get('optimize_threshold', True):
            target_recall = config['ensemble'].get('target_recall', 0.9)
            min_precision = config['ensemble'].get('min_precision', 0.6)

            print(f"Optimizing threshold for target recall {target_recall} with min precision {min_precision}...")
            optimal_threshold = optimize_threshold(
                all_targets, all_probs,
                target_recall=target_recall,
                min_precision=min_precision
            )

            # New metrics with optimal threshold
            threshold_preds = (np.array(all_probs) >= optimal_threshold).astype(int)
            threshold_metrics = calculate_metrics(all_targets, threshold_preds, all_probs)

            print(f"\nOptimized threshold: {optimal_threshold:.4f}")
            print("\n==== METRICS WITH OPTIMIZED THRESHOLD (ALL FOLDS) ====")
            print(f"Accuracy: {threshold_metrics['accuracy']:.4f}")
            print(f"Precision: {threshold_metrics['precision']:.4f}")
            print(f"Recall: {threshold_metrics['recall']:.4f} (Target: {target_recall:.2f})")
            print(f"F1 Score: {threshold_metrics['f1']:.4f}")
            print(f"Balanced Accuracy: {threshold_metrics['balanced_acc']:.4f}")
            print(f"Specificity: {threshold_metrics['specificity']:.4f}")
            print("======================================================\n")

            # Save optimal threshold
            with open(os.path.join(ensemble_dir, 'optimal_threshold.txt'), 'w') as f:
                f.write(str(optimal_threshold))

            # Save metrics
            metrics_path = os.path.join(ensemble_dir, 'ensemble_metrics.json')
            metrics_data = {
                'default_threshold': {
                    'threshold': 0.5,
                    'metrics': convert_to_python_types(metrics)
                },
                'optimized_threshold': {
                    'threshold': float(optimal_threshold),
                    'metrics': convert_to_python_types(threshold_metrics)
                }
            }

            with open(metrics_path, 'w') as f:
                json.dump(metrics_data, f, indent=4)

            print(f"Metrics saved to {metrics_path}")

        # Additional: Analyzing individual architecture contributions
        print("\nAnalyzing individual architecture contributions...")
        for architecture in config['model']['architectures']:
            # Filter models
            arch_indices = [i for i, info in enumerate(models_info) if info['architecture'] == architecture]
            if arch_indices:
                arch_models = [all_models[i] for i in arch_indices]
                arch_thresholds = [all_thresholds[i] for i in arch_indices]

                # Create ensemble
                arch_ensemble = EnsembleModel(arch_models, method=ensemble_method, thresholds=arch_thresholds)

                # Evaluation
                arch_metrics, _, _, _, _ = evaluate_ensemble(arch_ensemble, config, None, device)

                print(f"\n--- {architecture} (Ensemble of {len(arch_models)} models) ---")
                print(f"Accuracy: {arch_metrics['accuracy']:.4f}")
                print(f"Precision: {arch_metrics['precision']:.4f}")
                print(f"Recall: {arch_metrics['recall']:.4f}")
                print(f"F1 Score: {arch_metrics['f1']:.4f}")
                print(f"AUC: {arch_metrics.get('auc', 0):.4f}")
                print(f"Balanced Accuracy: {arch_metrics['balanced_acc']:.4f}")

        return ensemble, all_thresholds
    else:
        print("No models found!")
        return None, []


def evaluate_ensemble(ensemble, config, output_dir, device):
    """Evaluates ensemble model on validation set and calculates metrics"""
    from src.dataset import load_data

    # Load validation set
    _, val_loader, _ = load_data(config)

    # Predictions
    all_preds = []
    all_probs = []
    all_targets = []
    all_image_names = []

    ensemble.eval()
    with torch.no_grad():
        for batch in val_loader:
            # Load data
            if len(batch) == 3:  # With metadata
                inputs, targets, metadata = batch
                if 'image_name' in metadata:
                    all_image_names.extend(metadata['image_name'])
            else:
                inputs, targets = batch

            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = ensemble(inputs)
            probs = outputs[:, 1] if outputs.shape[1] > 1 else outputs
            preds = (probs >= 0.5).long()

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Calculate metrics
    metrics = calculate_metrics(all_targets, all_preds, all_probs)

    # Save predictions
    if output_dir and all_image_names:
        output_path = os.path.join(output_dir, 'validation_output.csv')

        # All arrays are same length
        min_length = min(len(all_image_names), len(all_preds), len(all_probs))
        all_image_names = all_image_names[:min_length]
        all_preds = all_preds[:min_length]
        all_probs = all_probs[:min_length]

        save_validation_predictions(all_image_names, all_preds, all_probs, output_path)

    return metrics, all_preds, all_probs, all_targets, all_image_names


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Create ensemble from all folds')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory (optional)')
    args = parser.parse_args()

    ensemble, thresholds = create_ensemble_all_folds(args.config, args.output_dir)
    if ensemble:
        print("Ensemble of all folds created successfully!")