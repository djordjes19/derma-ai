import torch
import torch.nn as nn
import torchvision.models as models


def get_model(config):
    """Kreiranje modela na osnovu konfiguracije"""
    model_type = config['model_type']

    if model_type == 'resnet50':
        # Učitavanje ResNet-50 modela sa pretreniranim težinama
        model = models.resnet50(pretrained=True)

        # Zamena poslednjeg sloja za binarnu klasifikaciju
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 2)
        )

    elif model_type == 'efficientnet-b0':
        # Učitavanje EfficientNet-B0 modela (potreban dodatni import)
        try:
            from efficientnet_pytorch import EfficientNet
            model = EfficientNet.from_pretrained('efficientnet-b0')

            # Zamena poslednjeg sloja za binarnu klasifikaciju
            num_features = model._fc.in_features
            model._fc = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(num_features, 2)
            )
        except ImportError:
            raise ImportError("Potrebno je instalirati 'efficientnet-pytorch' biblioteku za EfficientNet")

    else:
        raise ValueError(f"Nepodržani tip modela: {model_type}")

    return model