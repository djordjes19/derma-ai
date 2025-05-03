# Fair Melanoma Detection

An ensemble-based deep learning system for accurate and fair melanoma detection across diverse skin tones.

## Overview

This project addresses the critical challenge of developing an AI system for melanoma classification that maintains consistent performance across different skin tones. Using dermatoscopic images from the ISIC 2020 Challenge dataset, we've built a multi-architecture ensemble approach that prioritizes both accuracy and fairness.

## Features

- **Multi-architecture ensemble model** combining ResNet50, EfficientNet-B0, ConvNeXt-Tiny, and MobileNetV3-Large
- **Fairness-aware design** with performance evaluation across different skin tones
- **Threshold optimization** to balance precision and recall for clinical utility
- **K-fold cross-validation** with stratification by both target class and skin tone
- **Comprehensive metrics** including traditional performance and fairness measures

## Project Structure

```
/
├── data/                  # Dataset directory
│   ├── skin_tones.csv     # Skin tone information
│   ├── ground_truth.csv   # Ground truth labels
│   └── balanced_data.csv  # Balanced dataset file
├── models/                # Trained models
│   ├── resnet50/          # ResNet50 models
│   ├── efficientnet_b0/   # EfficientNet models
│   ├── convnext_tiny/     # ConvNeXt models
│   ├── mobilenetv3_large/ # MobileNet models
│   ├── ensemble/          # Ensemble model
│   └── config.yaml        # Configuration file
├── notebooks/             # Jupyter notebooks
│   ├── preprocessing/     # Data preprocessing notebooks
│   └── exploration/       # Data exploration notebooks
├── scripts/               # Utility scripts
│   ├── augmentation_function.py # Data augmentation utilities
│   └── Augmentacion.py    # Augmentation classes
├── src/                   # Source code
│   ├── model.py           # Model definitions
│   ├── dataset.py         # Dataset handling
│   ├── utils.py           # Utility functions
│   ├── ensemble.py        # Ensemble implementation
│   └── advanced_models.py # Additional model architectures
├── train.py               # Training script
└── predict.py             # Prediction script
```

## Technical Highlights

### Enhanced Model Architectures

All model architectures are enhanced with:
- Generalized Mean (GeM) pooling for improved feature aggregation
- Custom classification heads with batch normalization and dropout
- Pre-trained weights from ImageNet for transfer learning

### Ensemble Approach

The ensemble model combines predictions from multiple architectures and folds using:
- Model-specific optimized thresholds
- Configurable combination methods (average, max, voting)
- Fair performance across demographic groups

### Fairness Evaluation

We measure fairness using:
- Equal opportunity difference
- Precision and recall disparity across skin tones
- Performance ratio between different skin tone groups

## Results

The final ensemble model achieved:
- Balanced accuracy: 0.87
- Recall: 0.93
- Precision: 0.73
- AUC: 0.91

Fairness metrics showed significant improvement:
- Initial recall disparity of 0.15 across skin tone groups reduced to 0.06
- Precision ratio improved from 0.78 to 0.92
- High recall (>0.90) maintained across all skin tone groups

## Usage

### Training

```bash
python train.py --config config/config.yaml
```

### Prediction

```bash
python predict.py <INPUT_FOLDER> <OUTPUT_CSV> [--config CONFIG_PATH] [--threshold THRESHOLD]
```

## Requirements

- Python 3.6+
- PyTorch 1.8.0+
- CUDA-compatible GPU (recommended)
- Dependencies in requirements.txt

