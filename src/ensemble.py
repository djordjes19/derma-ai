import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
from src.model import get_model


class EnsembleModel(nn.Module):
    """
    Ensemble model which combines predictions of other models.

    Combination methods:
    - average: average the predictions of all models
    - max: max the predictions of all models(favours recall)
    - vote: vote based on discrete predictions
    """

    def __init__(self, models, method='average', thresholds=None):
        """
        Args:
            models (list): List of pretrained PyTorch models.
            method (str): Combination method ('average', 'max', 'vote')
            thresholds (list, optional): Listof thresholds of every model
        """
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        self.method = method
        self.thresholds = thresholds if thresholds is not None else [0.5] * len(models)

    def forward(self, x):
        """
        Forward pass in ensemble model.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Combined predictions
        """
        # Collecting predictions
        predictions = []
        for model in self.models:
            with torch.no_grad():  # Efficiency for inference
                output = model(x)
                if output.shape[1] > 1:  # For multi-class
                    output = F.softmax(output, dim=1)
                predictions.append(output)

        # Stack all predictions
        all_preds = torch.stack(predictions)

        # Combining predictions based on chosen method
        if self.method == 'average':
            return torch.mean(all_preds, dim=0)

        elif self.method == 'max':
            return torch.max(all_preds, dim=0)[0]

        elif self.method == 'vote':
            discrete_preds = []
            for i, pred in enumerate(all_preds):
                threshold = self.thresholds[i]
                discrete_pred = (pred[:, 1] >= threshold).float().unsqueeze(1)
                discrete_preds.append(torch.cat([1 - discrete_pred, discrete_pred], dim=1))

            discrete_stack = torch.stack(discrete_preds)
            votes = torch.sum(discrete_stack, dim=0)

            return votes / len(self.models)

        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")


class ThresholdOptimizedEnsemble(nn.Module):
    """
    Ensemble model with thresholds for recall optimization.
    """

    def __init__(self, models, thresholds, method='average', ensemble_threshold=0.5):
        """
        Args:
            models (list): List of pretrained PyTorch models.
            thresholds (list): List of optimized thresholds of every model
            method (str): Combining method ('average', 'max', 'vote')
            ensemble_threshold (float): Threshold for final ensemble output
        """
        super(ThresholdOptimizedEnsemble, self).__init__()
        self.models = nn.ModuleList(models)
        self.thresholds = thresholds
        self.method = method
        self.ensemble_threshold = ensemble_threshold

    def forward(self, x):
        """Forward pass with optimized thresholds"""
        # Collecting predictions
        predictions = []
        for model in self.models:
            with torch.no_grad():
                output = model(x)
                if output.shape[1] > 1:
                    output = F.softmax(output, dim=1)
                predictions.append(output)

        # Stack all predictions
        all_preds = torch.stack(predictions)

        # Combining predictions
        if self.method == 'average':
            combined = torch.mean(all_preds, dim=0)
        elif self.method == 'max':
            combined = torch.max(all_preds, dim=0)[0]
        elif self.method == 'vote':

            discrete_preds = []
            for i, pred in enumerate(all_preds):
                threshold = self.thresholds[i]
                discrete_pred = (pred[:, 1] >= threshold).float().unsqueeze(1)
                discrete_preds.append(torch.cat([1 - discrete_pred, discrete_pred], dim=1))

            # Voting
            discrete_stack = torch.stack(discrete_preds)
            votes = torch.sum(discrete_stack, dim=0)
            combined = votes / len(self.models)
        else:
            combined = torch.mean(all_preds, dim=0)

        if self.ensemble_threshold != 0.5:
            pass

        return combined


def save_ensemble(ensemble, config, models_info, thresholds, output_path):
    """
    Saves ensemble model and metadata.

    Args:
        ensemble (nn.Module): Ensemble model
        config (dict): Configuration
        models_info (list): Models info
        thresholds (list): Optimized thresholds
        output_path (str): Output path
    """
    os.makedirs(output_path, exist_ok=True)

    def convert_to_python_types(obj):
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

    # Saving parameters and thresholds
    ensemble_config = {
        'method': config['ensemble']['method'],
        'architectures': [info['architecture'] for info in models_info],
        'model_paths': [info['path'] for info in models_info],
        'thresholds': thresholds,
        'img_size': config['model']['img_size']
    }

    # Saving configuration
    with open(os.path.join(output_path, 'ensemble_config.json'), 'w') as f:
        json.dump(ensemble_config, f, indent=4)

    # Saving model state
    torch.save(ensemble.state_dict(), os.path.join(output_path, 'ensemble_model.pt'))

    # Saving models for ensemble model
    for info in models_info:
        model_name = info['architecture']
        fold = info.get('fold', '')
        model_filename = f"{model_name}_{fold}.pt" if fold else f"{model_name}.pt"
        if not os.path.exists(os.path.join(output_path, model_filename)):
            try:
                import shutil
                shutil.copy(info['path'], os.path.join(output_path, model_filename))
            except Exception as e:
                print(f"Error copying model: {e}")

    print(f"Ensemble saved to {output_path}")


def load_ensemble(config_path, device):
    """
    Loading ensemble modela from configuration.

    Args:
        config_path (str): Path to config file
        device (torch.device): Device to use

    Returns:
        nn.Module: Loaded ensemble model
    """
    # Loading config
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Loading models
    models = []
    model_dir = os.path.dirname(config_path)

    for i, (arch, path) in enumerate(zip(config['architectures'], config['model_paths'])):
        # Creating models
        model = get_model(arch, pretrained=False)

        # Loading model states
        model_path = os.path.join(model_dir, os.path.basename(path))
        if not os.path.exists(model_path):
            model_path = path  # Using absolute path if relative does not exist

        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        models.append(model)

    # Creating ensemble model
    thresholds = config.get('thresholds', [0.5] * len(models))
    method = config.get('method', 'average')

    if 'ensemble_threshold' in config:
        ensemble = ThresholdOptimizedEnsemble(models, thresholds, method, config['ensemble_threshold'])
    else:
        ensemble = EnsembleModel(models, method, thresholds)

    # Loading ensemble model state if possible
    ensemble_state_path = os.path.join(model_dir, 'ensemble_model.pt')
    if os.path.exists(ensemble_state_path):
        ensemble.load_state_dict(torch.load(ensemble_state_path, map_location=device))

    ensemble = ensemble.to(device)
    return ensemble