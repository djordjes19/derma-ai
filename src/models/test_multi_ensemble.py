import os
import argparse
import torch
import json
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report

from dataset import get_transforms
from ensemble import load_multi_architecture_ensemble, TestTimeAugmentation


def parse_args():
    parser = argparse.ArgumentParser(description='Test Multi-Architecture Ensemble modela')
    parser.add_argument('--ensemble_config', type=str, required=True,
                        help='Putanja do konfiguracije multi-architecture ensemble modela')
    parser.add_argument('--test_dir', type=str, required=True,
                        help='Direktorijum sa test slikama')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Direktorijum za čuvanje rezultata (ako je None, koristi se direktorijum ensemble konfiguracije)')
    parser.add_argument('--use_tta', action='store_true',
                        help='Da li koristiti test-time augmentaciju')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Veličina slike za inference')
    parser.add_argument('--csv_file', type=str, default=None,
                        help='CSV fajl sa oznakama za test slike (opciono)')
    return parser.parse_args()


def load_model(config_path, device):
    """Učitavanje multi-architecture ensemble modela"""
    ensemble = load_multi_architecture_ensemble(config_path, device)
    return ensemble


def create_test_transforms(img_size):
    """Kreiranje transformacija za testiranje"""
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Transformacije za test-time augmentaciju
    tta_transforms = [
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(degrees=(90, 90)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(degrees=(180, 180)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
    ]

    return test_transform, tta_transforms


def load_images_from_directory(dir_path, transform):
    """Učitavanje slika iz direktorijuma"""
    images = []
    filenames = []

    # Podrška za različite formate slika
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

    # Pretraga svih fajlova u direktorijumu
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)

        # Provera da li je fajl i ima ispravnu ekstenziju
        if os.path.isfile(file_path) and any(filename.lower().endswith(ext) for ext in valid_extensions):
            try:
                # Učitavanje i transformacija slike
                img = Image.open(file_path).convert('RGB')
                img_tensor = transform(img)

                # Čuvanje slike i imena fajla
                images.append(img_tensor)
                filenames.append(filename)
            except Exception as e:
                print(f"Error loading image {filename}: {e}")

    return images, filenames


def predict_with_ensemble(ensemble, images, device, use_tta=False, tta_transforms=None):
    """Predikcija sa ensemble modelom"""
    ensemble.eval()
    all_probs = []
    all_preds = []

    if use_tta and tta_transforms:
        # Kreiranje TTA objekta
        tta = TestTimeAugmentation(ensemble, tta_transforms, device)

        # Predikcija za svaku sliku sa TTA
        for img in tqdm(images, desc="Predicting with TTA"):
            pred = tta.predict(img, combine_method='average')
            prob = pred[0, 1].item()  # Verovatnoća za pozitivnu klasu
            pred_class = 1 if prob >= 0.5 else 0

            all_probs.append(prob)
            all_preds.append(pred_class)
    else:
        # Standardna predikcija bez TTA
        with torch.no_grad():
            for img in tqdm(images, desc="Predicting"):
                img_tensor = img.unsqueeze(0).to(device)
                output = ensemble(img_tensor)

                # Verovatnoća za pozitivnu klasu
                prob = torch.softmax(output, dim=1)[0, 1].item()
                pred_class = 1 if prob >= 0.5 else 0

                all_probs.append(prob)
                all_preds.append(pred_class)

    return all_preds, all_probs


def evaluate_predictions(predictions, true_labels):
    """Evaluacija predikcija u odnosu na stvarne oznake"""
    preds = np.array(predictions)
    labels = np.array(true_labels)

    # Standardne metrike
    accuracy = np.mean(preds == labels)

    # Konfuziona matrica
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    # Osnovne metrike
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    balanced_acc = (sensitivity + specificity) / 2

    # Kreiranje izveštaja
    metrics = {
        'accuracy': accuracy,
        'sensitivity': sensitivity,  # Recall za pozitivnu klasu
        'specificity': specificity,  # Specifičnost (TNR)
        'precision': precision,
        'f1_score': f1,
        'balanced_accuracy': balanced_acc,
        'true_positive': int(tp),
        'true_negative': int(tn),
        'false_positive': int(fp),
        'false_negative': int(fn)
    }

    return metrics


def create_results_visualizations(filenames, predictions, probabilities, true_labels=None, output_dir=None):
    """Kreiranje vizualizacija rezultata"""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Histogram verovatnoća
    plt.figure(figsize=(10, 6))
    plt.hist(probabilities, bins=20, alpha=0.7, color='skyblue')
    plt.axvline(x=0.5, color='red', linestyle='--', label='Threshold (0.5)')
    plt.title('Distribution of Melanoma Probabilities')
    plt.xlabel('Probability of Melanoma')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if output_dir:
        plt.savefig(os.path.join(output_dir, 'probability_distribution.png'))
    plt.close()

    # Ako imamo stvarne oznake, kreiramo dodatne vizualizacije
    if true_labels is not None:
        # Konfuziona matrica
        cm = confusion_matrix(true_labels, predictions)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        classes = ['Benign', 'Malignant']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        # Dodavanje vrednosti u ćelije
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        if output_dir:
            plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        plt.close()

        # ROC kriva
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(true_labels, probabilities)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        if output_dir:
            plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
        plt.close()


def main():
    args = parse_args()

    # Postavljanje izlaznog direktorijuma
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.ensemble_config)

    os.makedirs(args.output_dir, exist_ok=True)

    # Postavljanje uređaja
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Učitavanje ensemble modela
    print(f"Loading ensemble model from {args.ensemble_config}...")
    ensemble = load_model(args.ensemble_config, device)

    # Kreiranje transformacija
    test_transform, tta_transforms = create_test_transforms(args.img_size)

    # Učitavanje slika
    print(f"Loading images from {args.test_dir}...")
    images, filenames = load_images_from_directory(args.test_dir, test_transform)
    print(f"Loaded {len(images)} images")

    # Učitavanje stvarnih oznaka (ako postoje)
    true_labels = None
    if args.csv_file and os.path.exists(args.csv_file):
        try:
            df = pd.read_csv(args.csv_file)
            # Pretpostavljamo da CSV ima kolone 'image_name' i 'target'
            label_dict = dict(zip(df['image_name'], df['target']))
            true_labels = [label_dict.get(filename, None) for filename in filenames]
            if None in true_labels:
                print("Warning: Some images don't have corresponding labels in CSV")
                true_labels = None
            else:
                print(f"Loaded {len(true_labels)} labels from CSV")
        except Exception as e:
            print(f"Error loading labels from CSV: {e}")
            true_labels = None

    # Predikcija
    print("Running predictions...")
    predictions, probabilities = predict_with_ensemble(
        ensemble, images, device,
        use_tta=args.use_tta,
        tta_transforms=tta_transforms if args.use_tta else None
    )

    # Kreiranje DataFrame sa rezultatima
    results_df = pd.DataFrame({
        'image_name': filenames,
        'prediction': predictions,
        'probability': probabilities
    })

    # Dodavanje stvarnih oznaka ako postoje
    if true_labels is not None:
        results_df['true_label'] = true_labels

    # Čuvanje rezultata
    results_path = os.path.join(args.output_dir, 'predictions.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Predictions saved to {results_path}")

    # Evaluacija i vizualizacija
    if true_labels is not None:
        # Računanje metrika
        metrics = evaluate_predictions(predictions, true_labels)

        # Čuvanje metrika
        metrics_path = os.path.join(args.output_dir, 'test_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)

        # Ispis metrika
        print("\nTest Results:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Sensitivity (Recall): {metrics['sensitivity']:.4f}")
        print(f"Specificity: {metrics['specificity']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")

        # Kreiranje vizualizacija
        create_results_visualizations(filenames, predictions, probabilities, true_labels, args.output_dir)
    else:
        # Samo osnovne vizualizacije bez evaluacije
        create_results_visualizations(filenames, predictions, probabilities, output_dir=args.output_dir)


if __name__ == "__main__":
    main()