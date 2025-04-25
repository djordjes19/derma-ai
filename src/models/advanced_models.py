import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import Parameter
import torch.nn.functional as F
from model import GeM, CustomClassifier


class EfficientNetModel(nn.Module):
    """
    EfficientNet model sa naprednim pooling i custom classifier-om
    """

    def __init__(self, model_variant='b0', pretrained=True, num_classes=2, dropout=0.5):
        super(EfficientNetModel, self).__init__()

        # Učitavanje osnovnog modela
        if model_variant == 'b0':
            base = models.efficientnet_b0(pretrained=pretrained)
            feature_size = 1280
        elif model_variant == 'b1':
            base = models.efficientnet_b1(pretrained=pretrained)
            feature_size = 1280
        elif model_variant == 'b2':
            base = models.efficientnet_b2(pretrained=pretrained)
            feature_size = 1408
        elif model_variant == 'b3':
            base = models.efficientnet_b3(pretrained=pretrained)
            feature_size = 1536
        elif model_variant == 'b4':
            base = models.efficientnet_b4(pretrained=pretrained)
            feature_size = 1792
        else:
            raise ValueError(f"Unsupported EfficientNet variant: {model_variant}")

        # Uklanjamo classifier
        self.backbone = nn.Sequential(*list(base.children())[:-2])

        # GeM pooling
        self.gem_pool = GeM()

        # Custom classifier
        self.classifier = CustomClassifier(feature_size, num_classes, dropout)

    def forward(self, x):
        # Feature extraction
        features = self.backbone(x)

        # GeM pooling
        pooled = self.gem_pool(features).view(features.size(0), -1)

        # Classification
        output = self.classifier(pooled)

        return output


class ConvNeXtModel(nn.Module):
    """
    ConvNeXt model sa naprednim pooling i custom classifier-om
    """

    def __init__(self, model_variant='tiny', pretrained=True, num_classes=2, dropout=0.5):
        super(ConvNeXtModel, self).__init__()

        # Učitavanje osnovnog modela
        if model_variant == 'tiny':
            base = models.convnext_tiny(pretrained=pretrained)
            feature_size = 768
        elif model_variant == 'small':
            base = models.convnext_small(pretrained=pretrained)
            feature_size = 768
        elif model_variant == 'base':
            base = models.convnext_base(pretrained=pretrained)
            feature_size = 1024
        elif model_variant == 'large':
            base = models.convnext_large(pretrained=pretrained)
            feature_size = 1536
        else:
            raise ValueError(f"Unsupported ConvNeXt variant: {model_variant}")

        # Uklanjamo classifier
        self.backbone = nn.Sequential(*list(base.children())[:-2])

        # GeM pooling
        self.gem_pool = GeM()

        # Custom classifier
        self.classifier = CustomClassifier(feature_size, num_classes, dropout)

    def forward(self, x):
        # Feature extraction
        features = self.backbone(x)

        # GeM pooling
        pooled = self.gem_pool(features).view(features.size(0), -1)

        # Classification
        output = self.classifier(pooled)

        return output


class SwinTransformerModel(nn.Module):
    """
    Swin Transformer model sa custom classifier-om
    """

    def __init__(self, model_variant='tiny', pretrained=True, num_classes=2, dropout=0.5):
        super(SwinTransformerModel, self).__init__()

        # Učitavanje osnovnog modela
        if model_variant == 'tiny':
            base = models.swin_t(pretrained=pretrained)
            feature_size = 768
        elif model_variant == 'small':
            base = models.swin_s(pretrained=pretrained)
            feature_size = 768
        elif model_variant == 'base':
            base = models.swin_b(pretrained=pretrained)
            feature_size = 1024
        else:
            raise ValueError(f"Unsupported Swin Transformer variant: {model_variant}")

        # Uklanjamo head
        self.backbone = nn.Sequential(*list(base.children())[:-1])

        # Custom classifier (bez GeM poolinga jer Swin već ima svoj način poolinga)
        self.classifier = CustomClassifier(feature_size, num_classes, dropout)

    def forward(self, x):
        # Feature extraction
        features = self.backbone(x)

        # Reshape ako je potrebno
        if len(features.shape) > 2:
            features = features.view(features.size(0), -1)

        # Classification
        output = self.classifier(features)

        return output


class MaxVitModel(nn.Module):
    """
    MaxVit model sa custom classifier-om
    """

    def __init__(self, model_variant='t', pretrained=True, num_classes=2, dropout=0.5):
        super(MaxVitModel, self).__init__()

        # Učitavanje osnovnog modela
        if model_variant == 't':
            base = models.maxvit_t(pretrained=pretrained)
            feature_size = 512
        else:
            raise ValueError(f"Unsupported MaxVit variant: {model_variant}")

        # Uklanjamo head
        self.backbone = nn.Sequential(*list(base.children())[:-1])

        # GeM pooling
        self.gem_pool = GeM()

        # Custom classifier
        self.classifier = CustomClassifier(feature_size, num_classes, dropout)

    def forward(self, x):
        # Feature extraction
        features = self.backbone(x)

        # GeM pooling
        pooled = self.gem_pool(features).view(features.size(0), -1)

        # Classification
        output = self.classifier(pooled)

        return output


class MobileNetV3Model(nn.Module):
    """
    MobileNetV3 model sa GeM pooling i custom classifier-om.
    Optimizovan za brzo izvršavanje na mobilnim/laptop uređajima.
    """

    def __init__(self, model_variant='large', pretrained=True, num_classes=2, dropout=0.5):
        super(MobileNetV3Model, self).__init__()

        # Učitavanje osnovnog modela
        if model_variant == 'large':
            base = models.mobilenet_v3_large(pretrained=pretrained)
            feature_size = 960
        elif model_variant == 'small':
            base = models.mobilenet_v3_small(pretrained=pretrained)
            feature_size = 576
        else:
            raise ValueError(f"Unsupported MobileNetV3 variant: {model_variant}")

        # Uklanjamo classifier
        self.backbone = base.features

        # GeM pooling
        self.gem_pool = GeM()

        # Custom classifier
        self.classifier = CustomClassifier(feature_size, num_classes, dropout)

    def forward(self, x):
        # Feature extraction
        features = self.backbone(x)

        # GeM pooling
        pooled = self.gem_pool(features).view(features.size(0), -1)

        # Classification
        output = self.classifier(pooled)

        return output


class DeiTModel(nn.Module):
    """
    DeiT (Data-efficient image Transformer) model
    """

    def __init__(self, model_variant='small', pretrained=True, num_classes=2, dropout=0.5):
        super(DeiTModel, self).__init__()

        # Učitavanje osnovnog modela
        if model_variant == 'tiny':
            base = models.deit_tiny_patch16_224(pretrained=pretrained)
            feature_size = 192
        elif model_variant == 'small':
            base = models.deit_small_patch16_224(pretrained=pretrained)
            feature_size = 384
        elif model_variant == 'base':
            base = models.deit_base_patch16_224(pretrained=pretrained)
            feature_size = 768
        else:
            raise ValueError(f"Unsupported DeiT variant: {model_variant}")

        # Uklanjamo head
        self.backbone = nn.Sequential(*list(base.children())[:-1])

        # Custom classifier
        self.classifier = CustomClassifier(feature_size, num_classes, dropout)

    def forward(self, x):
        # Feature extraction
        features = self.backbone(x)

        # Classification (uzimamo CLS token)
        cls_token = features[:, 0]

        # Classification
        output = self.classifier(cls_token)

        return output


def get_advanced_model(config):
    """
    Kreiranje naprednog modela na osnovu konfiguracije
    """
    model_type = config['model_type']
    pretrained = config.get('pretrained', True)
    dropout = config.get('dropout', 0.5)

    # EfficientNet modeli
    if model_type.startswith('efficientnet_'):
        variant = model_type.split('_')[1]  # npr. 'b0', 'b3'
        model = EfficientNetModel(
            model_variant=variant,
            pretrained=pretrained,
            num_classes=2,
            dropout=dropout
        )
        print(f"Created EfficientNet-{variant.upper()} with GeM pooling and custom classifier")

    # ConvNeXt modeli
    elif model_type.startswith('convnext_'):
        variant = model_type.split('_')[1]  # npr. 'tiny', 'base'
        model = ConvNeXtModel(
            model_variant=variant,
            pretrained=pretrained,
            num_classes=2,
            dropout=dropout
        )
        print(f"Created ConvNeXt-{variant.capitalize()} with GeM pooling and custom classifier")

    # Swin Transformer modeli
    elif model_type.startswith('swin_'):
        variant = model_type.split('_')[1]  # npr. 'tiny', 'base'
        model = SwinTransformerModel(
            model_variant=variant,
            pretrained=pretrained,
            num_classes=2,
            dropout=dropout
        )
        print(f"Created Swin Transformer-{variant.capitalize()} with custom classifier")

    # MaxVit modeli
    elif model_type.startswith('maxvit_'):
        variant = model_type.split('_')[1]  # npr. 't'
        model = MaxVitModel(
            model_variant=variant,
            pretrained=pretrained,
            num_classes=2,
            dropout=dropout
        )
        print(f"Created MaxVit-{variant.upper()} with GeM pooling and custom classifier")

    # MobileNetV3 modeli
    elif model_type.startswith('mobilenetv3_'):
        variant = model_type.split('_')[1]  # npr. 'large', 'small'
        model = MobileNetV3Model(
            model_variant=variant,
            pretrained=pretrained,
            num_classes=2,
            dropout=dropout
        )
        print(f"Created MobileNetV3-{variant.capitalize()} with GeM pooling and custom classifier")

    # DeiT modeli
    elif model_type.startswith('deit_'):
        variant = model_type.split('_')[1]  # npr. 'tiny', 'small', 'base'
        model = DeiTModel(
            model_variant=variant,
            pretrained=pretrained,
            num_classes=2,
            dropout=dropout
        )
        print(f"Created DeiT-{variant.capitalize()} with custom classifier")

    else:
        # Za nepoznate tipove, vraćamo None i oslanjamo se na originalni get_model
        return None

    return model