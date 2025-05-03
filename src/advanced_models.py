import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import Parameter
import torch.nn.functional as F
from src.model import GeM, CustomClassifier


class EfficientNetModel(nn.Module):
    """
    EfficientNet model with advanced pooling and custom classifier.
    """

    def __init__(self, model_variant='b0', pretrained=True, num_classes=2, dropout=0.5):
        super(EfficientNetModel, self).__init__()

        # Loading base model
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

        # Removing classifier
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
    ConvNeXt model with advanced pooling and custom classifier.
    """

    def __init__(self, model_variant='tiny', pretrained=True, num_classes=2, dropout=0.5):
        super(ConvNeXtModel, self).__init__()

        # Loading base model
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

        # Removing classifier
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


class MobileNetV3Model(nn.Module):
    """
    MobileNetV3 model with GeM pooling and custom classifier.
    """

    def __init__(self, model_variant='large', pretrained=True, num_classes=2, dropout=0.5):
        super(MobileNetV3Model, self).__init__()

        # Loading base model
        if model_variant == 'large':
            base = models.mobilenet_v3_large(pretrained=pretrained)
            feature_size = 960
        elif model_variant == 'small':
            base = models.mobilenet_v3_small(pretrained=pretrained)
            feature_size = 576
        else:
            raise ValueError(f"Unsupported MobileNetV3 variant: {model_variant}")

        # Removing classifier
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


def get_advanced_model(config):
    """
    Creating advanced model based on config.

    Args:
        config (dict): Model configuration

    Returns:
        nn.Module: Initialized model or None if model type is not supported.
    """
    model_type = config['model_type']
    pretrained = config.get('pretrained', True)
    dropout = config.get('dropout', 0.5)

    # EfficientNet models
    if model_type.startswith('efficientnet_'):
        variant = model_type.split('_')[1]
        model = EfficientNetModel(
            model_variant=variant,
            pretrained=pretrained,
            num_classes=2,
            dropout=dropout
        )
        print(f"Created EfficientNet-{variant.upper()} with GeM pooling and custom classifier")
        return model

    # ConvNeXt models
    elif model_type.startswith('convnext_'):
        variant = model_type.split('_')[1]
        model = ConvNeXtModel(
            model_variant=variant,
            pretrained=pretrained,
            num_classes=2,
            dropout=dropout
        )
        print(f"Created ConvNeXt-{variant.capitalize()} with GeM pooling and custom classifier")
        return model

    # MobileNetV3 models
    elif model_type.startswith('mobilenetv3_'):
        variant = model_type.split('_')[1]
        model = MobileNetV3Model(
            model_variant=variant,
            pretrained=pretrained,
            num_classes=2,
            dropout=dropout
        )
        print(f"Created MobileNetV3-{variant.capitalize()} with GeM pooling and custom classifier")
        return model

    # For unsupported types, return None
    return None