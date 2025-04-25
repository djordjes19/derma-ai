import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
import yaml
from model import get_model


class EnsembleModel(nn.Module):
    """
    Ensemble model koji kombinuje predikcije više modela.

    Metode kombinovanja:
    - average: jednostavna srednja vrednost predikcija
    - weighted: težinska srednja vrednost predikcija
    - max: maksimalna vrednost predikcija
    - vote: glasanje na osnovu diskretnih predikcija
    """

    def __init__(self, models, method='average', weights=None):
        """
        Args:
            models (list): Lista već istreniranih PyTorch modela
            method (str): Metod kombinovanja ('average', 'weighted', 'max', 'vote')
            weights (list, optional): Lista težina za 'weighted' metod
        """
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        self.method = method

        # Validacija i inicijalizacija težina
        if method == 'weighted':
            if weights is None:
                # Default: jednake težine
                self.weights = torch.ones(len(models)) / len(models)
            else:
                # Normalizacija težina
                weights_sum = sum(weights)
                self.weights = torch.tensor([w / weights_sum for w in weights])
        elif weights is not None:
            print(f"Warning: weights provided but method is '{method}', not 'weighted'. Weights will be ignored.")

    def forward(self, x):
        """
        Forward pass kroz ensemble model.

        Args:
            x (torch.Tensor): Ulazni tensor

        Returns:
            torch.Tensor: Kombinovane predikcije
        """
        # Skupljanje predikcija svih modela
        predictions = []
        for model in self.models:
            with torch.no_grad():  # Efficiency for inference
                output = model(x)
                # Pretvaramo logite u verovatnoće
                if output.shape[1] > 1:  # Za multi-class
                    output = F.softmax(output, dim=1)
                predictions.append(output)

        # Stack all predictions
        all_preds = torch.stack(predictions)

        # Kombinovanje predikcija na osnovu izabrane metode
        if self.method == 'average':
            # Jednostavna srednja vrednost
            return torch.mean(all_preds, dim=0)

        elif self.method == 'weighted':
            # Težinska srednja vrednost
            weights = self.weights.to(all_preds.device)
            weighted_preds = all_preds * weights.view(-1, 1, 1)
            return torch.sum(weighted_preds, dim=0)

        elif self.method == 'max':
            # Maksimalna verovatnoća za svaku klasu
            return torch.max(all_preds, dim=0)[0]

        elif self.method == 'vote':
            # Diskretno glasanje
            class_votes = torch.zeros(x.size(0), all_preds.shape[2]).to(x.device)
            for pred in all_preds:
                pred_classes = torch.argmax(pred, dim=1)
                for i, cls in enumerate(pred_classes):
                    class_votes[i, cls] += 1
            return class_votes / len(self.models)

        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")


class MultiArchitectureEnsemble(nn.Module):
    """
    Ensemble model koji kombinuje više različitih arhitektura.
    Omogućava kreiranje modela iz konfiguracije i kombinovanje njihovih predikcija.
    """

    def __init__(self, architectures, model_paths, method='average', weights=None):
        """
        Args:
            architectures (list): Lista tipova arhitektura (npr. 'resnet50', 'efficientnet_b0')
            model_paths (list): Putanje do sačuvanih modela za svaku arhitekturu
            method (str): Metod kombinovanja ('average', 'weighted', 'max', 'vote')
            weights (list, optional): Lista težina za 'weighted' metod
        """
        super(MultiArchitectureEnsemble, self).__init__()

        if len(architectures) != len(model_paths):
            raise ValueError("Number of architectures must match number of model paths")

        # Učitavanje modela različitih arhitektura
        self.models = nn.ModuleList()
        for arch, path in zip(architectures, model_paths):
            config = {'model_type': arch, 'pretrained': False}
            model = get_model(config)
            model.load_state_dict(torch.load(path, map_location='cpu'))
            model.eval()  # Postavljamo u evaluation mod
            self.models.append(model)

        # Inicijalizacija základnog ensemble-a
        self.ensemble = EnsembleModel(self.models, method=method, weights=weights)

    def forward(self, x):
        """Forward pass kroz ensemble"""
        return self.ensemble(x)


class TestTimeAugmentation:
    """
    Test-time augmentacija za poboljšanje predikcija.
    Primenjuje različite augmentacije na sliku tokom inference-a
    i kombinuje predikcije.
    """

    def __init__(self, model, augmentations, device):
        """
        Args:
            model (nn.Module): Model za predikciju
            augmentations (list): Lista transformacija za augmentaciju
            device (torch.device): Uređaj za predikcije (CPU ili GPU)
        """
        self.model = model
        self.augmentations = augmentations
        self.device = device
        self.model.eval()  # Postavljamo u evaluation mod

    def predict(self, image, combine_method='average'):
        """
        Predikcija sa test-time augmentacijom

        Args:
            image (torch.Tensor): Ulazna slika
            combine_method (str): Metod kombinovanja ('average', 'max', 'vote')

        Returns:
            torch.Tensor: Kombinovane predikcije
        """
        all_preds = []

        # Originalna slika
        with torch.no_grad():
            orig_pred = self.model(image.unsqueeze(0).to(self.device))
            if orig_pred.shape[1] > 1:
                orig_pred = F.softmax(orig_pred, dim=1)
            all_preds.append(orig_pred)

        # Augmentirane slike
        for aug in self.augmentations:
            aug_image = aug(image)
            with torch.no_grad():
                aug_pred = self.model(aug_image.unsqueeze(0).to(self.device))
                if aug_pred.shape[1] > 1:
                    aug_pred = F.softmax(aug_pred, dim=1)
                all_preds.append(aug_pred)

        # Kombinovanje predikcija
        all_preds = torch.cat(all_preds, dim=0)

        if combine_method == 'average':
            return torch.mean(all_preds, dim=0, keepdim=True)
        elif combine_method == 'max':
            return torch.max(all_preds, dim=0, keepdim=True)[0]
        elif combine_method == 'vote':
            pred_classes = torch.argmax(all_preds, dim=1)
            class_votes = torch.bincount(pred_classes, minlength=all_preds.shape[1])
            return F.one_hot(torch.argmax(class_votes), num_classes=all_preds.shape[1]).float().unsqueeze(0)
        else:
            raise ValueError(f"Unknown combine method: {combine_method}")


def create_multi_architecture_ensemble(config_paths, model_paths, output_dir, ensemble_method='average', weights=None):
    """
    Kreiranje i čuvanje multi-architecture ensemble modela

    Args:
        config_paths (list): Lista putanja do konfiguracionih fajlova za svaki model
        model_paths (list): Lista putanja do sačuvanih modela za svaku arhitekturu
        output_dir (str): Direktorijum za čuvanje ensemble modela
        ensemble_method (str): Metod kombinovanja ('average', 'weighted', 'max', 'vote')
        weights (list, optional): Lista težina za 'weighted' metod

    Returns:
        dict: Konfiguracija ensemble modela
    """
    # Čitanje konfiguracionih fajlova za arhitekture
    architectures = []

    for config_path in config_paths:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            architectures.append(config['model_type'])

    # Kreiranje konfiguracije ensemble modela
    ensemble_config = {
        'architectures': architectures,
        'model_paths': [str(path) for path in model_paths],
        'ensemble_method': ensemble_method
    }

    if weights is not None:
        ensemble_config['weights'] = weights

    # Kreiranje direktorijuma za ensemble model
    os.makedirs(output_dir, exist_ok=True)

    # Čuvanje konfiguracije
    config_path = os.path.join(output_dir, 'multi_architecture_ensemble_config.json')
    with open(config_path, 'w') as f:
        json.dump(ensemble_config, f, indent=4)

    print(f"Multi-architecture ensemble configuration saved to {config_path}")
    print(f"Ensemble will use these architectures: {architectures}")

    return ensemble_config


def load_multi_architecture_ensemble(config_path, device):
    """
    Učitavanje multi-architecture ensemble modela iz konfiguracije

    Args:
        config_path (str): Putanja do konfiguracionog fajla
        device (torch.device): Uređaj za model

    Returns:
        MultiArchitectureEnsemble: Ensemble model
    """
    # Učitavanje konfiguracije
    with open(config_path, 'r') as f:
        ensemble_config = json.load(f)

    # Izvlačenje parametara
    architectures = ensemble_config['architectures']
    model_paths = ensemble_config['model_paths']
    ensemble_method = ensemble_config.get('ensemble_method', 'average')
    weights = ensemble_config.get('weights', None)

    # Kreiranje ensemble modela
    ensemble = MultiArchitectureEnsemble(
        architectures,
        model_paths,
        method=ensemble_method,
        weights=weights
    )

    ensemble = ensemble.to(device)
    return ensemble