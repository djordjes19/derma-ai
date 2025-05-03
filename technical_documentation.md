 # Fair Melanoma Detection - Technical Documentation

## 1. System Architecture

### 1.1 Overview
The Fair Melanoma Detection system consists of a multi-architecture ensemble model for melanoma classification from dermatoscopic images. The system is designed with fairness in mind, ensuring consistent performance across different skin tones through specialized training, threshold optimization, and ensemble techniques.

### 1.2 Architecture Diagram
```
                   +------------------+
                   |   Input Image    |
                   +--------+---------+
                            |
                            v
              +-------------+-------------+
              |        Preprocessing      |
              | (Resize, Normalize, etc.) |
              +-------------+-------------+
                            |
                            v
+------------------------------------------------------+
|                       Ensemble                       |
|  +----------+  +----------+  +----------+  +-------+ |
|  | ResNet50 |  |EfficientB0|  |ConvNeXt  |  |MobileV3| |
|  +----+-----+  +-----+----+  +-----+----+  +---+---+ |
|       |              |             |            |    |
|       v              v             v            v    |
|  +----+-----+  +-----+----+  +-----+----+  +---+---+ |
|  |Threshold1 |  |Threshold2|  |Threshold3|  |Thresh4| |
|  +----+-----+  +-----+----+  +-----+----+  +---+---+ |
|       |              |             |            |    |
+-------|--------------|-------------|------------|----+
        |              |             |            |
        +---+----------+-------------+------------+
            |
            v
    +-------+---------+
    | Ensemble Output |
    +-------+---------+
            |
            v
    +-------+---------+
    |   Final Result  |
    +-------+---------+
```

## 2. Technical Components

### 2.1 Base Model Architectures
The system incorporates multiple CNN architectures, each with specific technical enhancements:

#### 2.1.1 Enhanced ResNet50
- **Base Architecture**: ResNet50 with residual connections
- **Enhancements**: 
  - Replaced standard average pooling with Generalized Mean (GeM) pooling
  - Added custom classifier with batch normalization and dropout (rate: 0.5)
  - Used ImageNet pre-trained weights for transfer learning

#### 2.1.2 EfficientNet-B0
- **Base Architecture**: EfficientNet-B0 with compound scaling
- **Feature Size**: 1280 dimensions
- **Enhancements**:
  - GeM pooling layer
  - Custom classifier with 512-unit hidden layer
  - Batch normalization and ReLU activation
  - Dropout regularization (rate: 0.5)

#### 2.1.3 ConvNeXt-Tiny
- **Base Architecture**: ConvNeXt-Tiny with modern CNN design
- **Feature Size**: 768 dimensions
- **Enhancements**:
  - GeM pooling for better feature aggregation
  - Custom classification head with batch normalization
  - Dropout regularization (rate: 0.5)

#### 2.1.4 MobileNetV3-Large
- **Base Architecture**: MobileNetV3-Large for efficiency
- **Feature Size**: 960 dimensions
- **Enhancements**:
  - GeM pooling to improve feature quality
  - Custom classifier with regularization
  - Optimized for better performance while maintaining efficiency

### 2.2 Key Technical Components

#### 2.2.1 Generalized Mean (GeM) Pooling
```python
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), 
                          (x.size(-2), x.size(-1))).pow(1. / p)
```
- **Functionality**: Provides a generalized form of pooling that includes both max pooling (p→∞) and average pooling (p=1)
- **Advantage**: Allows the model to learn the optimal pooling strategy during training
- **Implementation**: Parameterized pooling with learnable parameter p

#### 2.2.2 Custom Classifier
```python
class CustomClassifier(nn.Module):
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
```
- **Architecture**: Two-layer neural network with batch normalization
- **Hidden Layer**: 512 units with ReLU activation
- **Regularization**: Dropout with configurable rate (default: 0.5)
- **Output**: Binary classification (2 classes)

### 2.3 Data Processing Pipeline

#### 2.3.1 Dataset Class
```python
class MelanomaDataset(Dataset):
    def __init__(self, img_paths, labels, transform=None, metadata=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform
        self.metadata = metadata

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        if self.metadata is not None:
            metadata = {key: self.metadata.iloc[idx][key] 
                      for key in self.metadata.columns}
            return image, torch.tensor(label, dtype=torch.long), metadata

        return image, torch.tensor(label, dtype=torch.long)
```
- **Functionality**: Custom PyTorch Dataset for loading dermatoscopic images
- **Features**: 
  - Supports metadata inclusion for fairness analysis
  - Applies image transformations
  - Handles RGB conversion

#### 2.3.2 Data Augmentation
```python
train_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, 
                          saturation=0.1, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225]),
])
```
- **Transforms**: Resize, flips, rotation, color jitter
- **Normalization**: ImageNet statistics (mean and std)
- **Color Adjustments**: Carefully limited to preserve diagnostic features

### 2.4 Ensemble Implementation

#### 2.4.1 Ensemble Model
```python
class EnsembleModel(nn.Module):
    def __init__(self, models, method='average', thresholds=None):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        self.method = method
        self.thresholds = thresholds if thresholds is not None else [0.5] * len(models)

    def forward(self, x):
        predictions = []
        for model in self.models:
            with torch.no_grad():
                output = model(x)
                if output.shape[1] > 1:
                    output = F.softmax(output, dim=1)
                predictions.append(output)

        all_preds = torch.stack(predictions)

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
```
- **Methods**: 
  - `average`: Mean of all model outputs
  - `max`: Maximum probability (favors recall)
  - `vote`: Threshold-based voting system
- **Thresholds**: Individual thresholds for each model
- **Implementation**: Uses PyTorch's ModuleList for proper model management

#### 2.4.2 Threshold-Optimized Ensemble
```python
class ThresholdOptimizedEnsemble(nn.Module):
    def __init__(self, models, thresholds, method='average', ensemble_threshold=0.5):
        super(ThresholdOptimizedEnsemble, self).__init__()
        self.models = nn.ModuleList(models)
        self.thresholds = thresholds
        self.method = method
        self.ensemble_threshold = ensemble_threshold
```
- **Enhancement**: Extends base ensemble with individual model thresholds
- **Functionality**: Applies model-specific thresholds before combination
- **Configuration**: Supports setting a final ensemble threshold

### 2.5 Threshold Optimization

#### 2.5.1 Optimization Algorithm
```python
def optimize_threshold(y_true, y_prob, target_recall=0.9, min_precision=0.6):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    
    if len(thresholds) < len(precision):
        precision = precision[:len(thresholds)]
        recall = recall[:len(thresholds)]
    
    optimal_threshold = 0.5
    best_f1 = 0
    
    valid_indices = recall >= target_recall
    
    if np.any(valid_indices):
        valid_precision = precision[valid_indices]
        valid_recall = recall[valid_indices]
        valid_thresholds = thresholds[valid_indices]
        
        precision_indices = valid_precision >= min_precision
        
        if np.any(precision_indices):
            f1_scores = 2 * (valid_precision[precision_indices] * 
                           valid_recall[precision_indices]) / (
                        valid_precision[precision_indices] + 
                        valid_recall[precision_indices] + 1e-10)
            
            best_idx = np.argmax(f1_scores)
            optimal_threshold = valid_thresholds[precision_indices][best_idx]
            best_f1 = f1_scores[best_idx]
        else:
            optimal_threshold = valid_thresholds[np.argmax(valid_precision)]
    else:
        optimal_threshold = thresholds[np.argmax(recall)]
    
    return optimal_threshold
```
- **Objective**: Find threshold that maximizes F1 score while meeting target recall
- **Constraints**: Minimum precision requirement to control false positives
- **Method**: Analyzes precision-recall curve to find optimal operating point
- **Fallback**: Graceful degradation when constraints cannot be satisfied

## 3. Training Process

### 3.1 Training Configuration
The training process is configured using a YAML configuration file:

```yaml
data:
  csv_file: balansirani_podaci.csv
  data_dir: [path_to_training_images]
  output_dir: ./models
  valid_size: 0.2
model:
  architectures:
  - resnet50
  - efficientnet_b0
  - convnext_tiny
  - mobilenetv3_large
  dropout: 0.5
  img_size: 256
  pretrained: true
training:
  batch_size: 32
  early_stopping_patience: 7
  epochs: 25
  grad_clip_value: 1.0
  learning_rate: 0.0002
  scheduler: cosine
  use_class_weights: true
  weight_decay: 0.0001
kfold:
  n_splits: 5
  random_state: 42
  shuffle: true
  use_kfold: true
ensemble:
  create_ensemble: true
  method: average
  min_precision: 0.7
  target_recall: 0.925
```

### 3.2 Training Algorithm
The main training function implements the following process:

```python
def train_model(config, architecture, fold=None, train_loader=None, val_loader=None, class_weights=None):
    # Device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = get_model(
        model_type=architecture,
        pretrained=config['model']['pretrained'],
        dropout=config['model']['dropout']
    )
    model = model.to(device)
    
    # Loss function with class weights if specified
    if config['training'].get('use_class_weights', False) and class_weights is not None:
        class_weights = class_weights.to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler_type = config['training'].get('scheduler', None)
    if scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(...)
    elif scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(...)
    
    # Training loop with early stopping
    best_val_metric = float('inf') if config['training'].get('monitor_metric', 'loss') == 'loss' else 0
    patience_counter = 0
    
    for epoch in range(config['training']['epochs']):
        # Train epoch
        train_metrics = train_epoch(...)
        
        # Validation
        val_metrics, val_probs, val_targets, val_image_names, group_metrics = validate(...)
        
        # Early stopping check
        monitor_metric = config['training'].get('monitor_metric', 'loss')
        current_metric = val_metrics['loss'] if monitor_metric == 'loss' else val_metrics[monitor_metric]
        
        if (monitor_metric == 'loss' and current_metric < best_val_metric) or \
                (monitor_metric != 'loss' and current_metric > best_val_metric):
            # Save best model
            # Reset patience
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['training'].get('early_stopping_patience', 7):
            break
    
    # Load best model
    # Optimize threshold
    # Return model and metrics
```

### 3.3 K-Fold Cross-Validation
The system implements k-fold cross-validation to improve robustness:

```python
def train_with_kfold(config):
    # Load data
    df = pd.read_csv(config['data']['csv_file'])
    
    # Create stratification key
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
    
    # Train models for each architecture and fold
    for architecture in config['model']['architectures']:
        for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['stratify_key'])):
            # Create data loaders
            train_loader, val_loader, class_weights = load_data(
                config, fold_indices=(train_idx, val_idx)
            )
            
            # Train model
            model, metrics, val_probs, val_targets, val_image_names, threshold = train_model(...)
            
            # Save results
```

## 4. Evaluation and Metrics

### 4.1 Evaluation Metrics
The system uses the following metrics for evaluation:

```python
def calculate_metrics(y_true, y_pred, y_prob=None):
    metrics = {}
    
    # Base metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['true_negative'] = tn
    metrics['false_positive'] = fp
    metrics['false_negative'] = fn
    metrics['true_positive'] = tp
    
    # Specificity and balanced accuracy
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['balanced_acc'] = (metrics['recall'] + metrics['specificity']) / 2
    
    # AUC if probabilities are given
    if y_prob is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_prob)
        except:
            metrics['auc'] = 0.5
    
    return metrics
```

### 4.2 Fairness Metrics
To evaluate fairness, the system implements specialized metrics across groups:

```python
def calculate_fairness_metrics(metrics_by_group):
    fairness = {}
    
    # Metrics to evaluate
    metrics_to_check = ['recall', 'precision', 'specificity', 'balanced_acc', 'f1']
    
    for metric in metrics_to_check:
        values = [group_metrics[metric] for group_metrics in metrics_by_group.values()]
        fairness[f'{metric}_min'] = min(values)
        fairness[f'{metric}_max'] = max(values)
        fairness[f'{metric}_disparity'] = max(values) - min(values)
        fairness[f'{metric}_ratio'] = min(values) / max(values) if max(values) > 0 else 1.0
    
    # Equal opportunity difference
    recalls = [group_metrics['recall'] for group_metrics in metrics_by_group.values()]
    fairness['equal_opportunity_difference'] = max(recalls) - min(recalls)
    
    # Disparate impact
    precision_ratio = fairness['precision_ratio']
    fairness['disparate_impact'] = 1.0 - precision_ratio
    
    return fairness
```

## 5. Inference System

### 5.1 Prediction Pipeline
The system provides a streamlined prediction pipeline for new images:

```python
def predict(model, test_loader, device, threshold=0.5):
    model.eval()
    all_probs = []
    
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            if outputs.shape[1] == 2:
                probs = outputs[:, 1]
            else:
                probs = torch.sigmoid(outputs)
            
            all_probs.extend(probs.cpu().numpy())
    
    # Convert to predictions
    all_preds = (torch.tensor(all_probs) >= threshold).int().tolist()
    
    return all_preds, all_probs
```

### 5.2 Ensemble Loading
The prediction system can load saved ensemble models:

```python
def load_ensemble(config_path, device):
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load individual models
    models = []
    model_dir = os.path.dirname(config_path)
    
    for i, (arch, path) in enumerate(zip(config['architectures'], config['model_paths'])):
        # Create model
        model = get_model(arch, pretrained=False)
        
        # Load weights
        model_path = os.path.join(model_dir, os.path.basename(path))
        if not os.path.exists(model_path):
            model_path = path
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        models.append(model)
    
    # Create ensemble
    thresholds = config.get('thresholds', [0.5] * len(models))
    method = config.get('method', 'average')
    
    if 'ensemble_threshold' in config:
        ensemble = ThresholdOptimizedEnsemble(models, thresholds, method, config['ensemble_threshold'])
    else:
        ensemble = EnsembleModel(models, method, thresholds)
    
    # Load ensemble state if available
    ensemble_state_path = os.path.join(model_dir, 'ensemble_model.pt')
    if os.path.exists(ensemble_state_path):
        ensemble.load_state_dict(torch.load(ensemble_state_path, map_location=device))
    
    ensemble = ensemble.to(device)
    return ensemble
```

### 5.3 Prediction Command-Line Interface
The system provides a simple command-line interface for predictions:

```python
def main():
    parser = argparse.ArgumentParser(description='Fair Melanoma Detection Prediction')
    parser.add_argument('input_folder', type=str, help='Directory with input images')
    parser.add_argument('output_csv', type=str, help='Path for output CSV file')
    parser.add_argument('--config', type=str, default='./models/ensemble/ensemble_config.json',
                        help='Path to ensemble configuration')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Classification threshold')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Use provided threshold or average from config
    if args.threshold is not None:
        threshold = args.threshold
    elif 'thresholds' in config:
        threshold = sum(config['thresholds']) / len(config['thresholds'])
    else:
        threshold = 0.5
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load test data
    test_loader, image_names = get_test_loader(
        args.input_folder,
        config['img_size'],
        batch_size=args.batch_size
    )
    
    # Load model
    model = load_ensemble(args.config, device)
    
    # Make predictions
    predictions, probabilities = predict(model, test_loader, device, threshold)
    
    # Save results
    results = pd.DataFrame({
        'image_name': image_names,
        'target': predictions
    })
    results.to_csv(args.output_csv, index=False)
```

## 6. Technical Requirements and Dependencies

### 6.1 Hardware Requirements
- **Recommended**: CUDA-compatible GPU with 8+ GB memory
- **Minimum**: 16 GB RAM, multicore CPU
- **Storage**: 10+ GB free space for models and data

### 6.2 Software Dependencies
```
torch>=1.8.0
torchvision>=0.9.0
numpy>=1.19.2
pandas>=1.1.5
scikit-learn>=0.24.0
pillow>=8.1.0
matplotlib>=3.3.4
pyyaml>=5.4.1
tqdm>=4.56.0
```

### 6.3 Directory Structure
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

## 7. Technical Deployment

### 7.1 Saved Model Format
The system saves models in the following formats:

- **Individual Models**: PyTorch state dictionaries (.pt files)
- **Ensemble Configuration**: JSON file with architecture paths and thresholds
- **Ensemble State**: Optional state dictionary for the ensemble model

### 7.2 Inference Pipeline
The inference pipeline follows these steps:

1. **Image Loading**: Load and preprocess input images
2. **Model Loading**: Load ensemble configuration and individual models
3. **Inference**: Generate predictions using the ensemble model
4. **Threshold Application**: Apply optimized thresholds to probabilities
5. **Output Generation**: Save predictions to CSV file


```

