import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import Parameter
import torch.nn.functional as F


class GeM(nn.Module):
    """
    Generalized Mean Pooling sloj.
    Bolji od standardnog average pooling-a za mnoge zadatke klasifikacije slika.

    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)


class CustomClassifier(nn.Module):
    """
    Napredna klasifikaciona glava sa batch normalizacijom i dropoutom
    """

    def __init__(self, in_features, num_classes=2, dropout=0.5):
        super(CustomClassifier, self).__init__()
        self.fc1 = nn.Linear(in_features, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class EnhancedResNet(nn.Module):
    """
    Poboljšana verzija ResNet modela sa GeM poolingom i custom classifierom
    """

    def __init__(self, base_model='resnet50', pretrained=True, num_classes=2, dropout=0.5):
        super(EnhancedResNet, self).__init__()

        # Učitavanje osnovnog modela
        if base_model == 'resnet18':
            base = models.resnet18(pretrained=pretrained)
        elif base_model == 'resnet34':
            base = models.resnet34(pretrained=pretrained)
        elif base_model == 'resnet50':
            base = models.resnet50(pretrained=pretrained)
        elif base_model == 'resnet101':
            base = models.resnet101(pretrained=pretrained)
        elif base_model == 'resnet152':
            base = models.resnet152(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported ResNet model: {base_model}")

        # Preuzimamo sve slojeve osim poslednjeg fully connected sloja
        self.backbone = nn.Sequential(*list(base.children())[:-2])

        # Feature veličina za ResNet modele
        if base_model in ['resnet18', 'resnet34']:
            feature_size = 512
        else:
            feature_size = 2048

        # GeM pooling
        self.gem_pool = GeM()

        # Custom classifier head
        self.classifier = CustomClassifier(feature_size, num_classes, dropout)

    def forward(self, x):
        # Feature extraction
        features = self.backbone(x)

        # GeM pooling
        pooled = self.gem_pool(features).view(features.size(0), -1)

        # Classification
        output = self.classifier(pooled)

        return output
def get_model(config):
    """Kreiranje modela na osnovu konfiguracije"""
    model_type = config['model_type']
    pretrained = config.get('pretrained', True)
    dropout = config.get('dropout', 0.5)

    # Prvo proveriti napredne modele
    try:
        from advanced_models import get_advanced_model
        # Pokušaj kreirati napredni model
        advanced_model = get_advanced_model(config)
        if advanced_model is not None:
            return advanced_model
    except ImportError:
        print("Advanced models module not found, using base models only.")
    except Exception as e:
        print(f"Error loading advanced model: {e}")
        print("Falling back to base models.")

    # Napredna verzija ResNet
    if model_type in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
        model = EnhancedResNet(
            base_model=model_type,
            pretrained=pretrained,
            num_classes=2,
            dropout=dropout
        )
        print(f"Created Enhanced {model_type} with GeM pooling and custom classifier")

    # Ako želimo standardnu verziju ResNet (bez GeM-a i custom glave)
    elif model_type == 'vanilla_resnet50':
        model = models.resnet50(pretrained=pretrained)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 2)
        )
        print("Created standard ResNet50 model")

    # DenseNet
    elif model_type == 'densenet121':
        model = models.densenet121(pretrained=pretrained)
        num_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 2)
        )
        print("Created DenseNet121 model")

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return model