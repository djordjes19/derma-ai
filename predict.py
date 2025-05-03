"""
Fair Melanoma Detection - Prediction

Script used for prediction on new dermatoscopic images.
Uses ensemble model za melanoma classification.

How to use:
python predict.py <INPUT_FOLDER> <OUTPUT_CSV>
"""

import os
import sys
import torch
import pandas as pd
import yaml
import json
import argparse
from tqdm import tqdm
from src.dataset import get_test_loader
from src.ensemble import load_ensemble


def parse_args():
    """Parses arguments from command line."""
    parser = argparse.ArgumentParser(description='Fair Melanoma Detection Prediction')
    parser.add_argument('input_folder', type=str, help='Direktorijum sa ulaznim slikama')
    parser.add_argument('output_csv', type=str, help='Putanja za izlazni CSV fajl')
    parser.add_argument('--config', type=str, default='./models/ensemble/ensemble_config.json',
                        help='Putanja do ensemble konfiguracije')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Threshold za klasifikaciju (ako nije navedeno, koristi se iz konfiguracije)')
    parser.add_argument('--batch_size', type=int, default=16, help='Veličina batch-a')
    parser.add_argument('--use_tta', action='store_true', help='Koristi test-time augmentaciju')
    return parser.parse_args()


def predict(model, test_loader, device, threshold=0.5):
    """
    Makes predictions on test set.

    Args:
        model (nn.Module): Prediction model
        test_loader (DataLoader): DataLoader for test set
        device (torch.device): Prediction device
        threshold (float): Classification threshold

    Returns:
        tuple: (predictions, probabilities)
    """
    model.eval()
    all_probs = []

    with torch.no_grad():
        for inputs in tqdm(test_loader, desc="Predicting"):
            inputs = inputs.to(device)
            outputs = model(inputs)

            if outputs.shape[1] == 2:
                probs = outputs[:, 1]
            else:
                probs = torch.sigmoid(outputs)

            all_probs.extend(probs.cpu().numpy())

    # Conversion to predictions
    all_preds = (torch.tensor(all_probs) >= threshold).int().tolist()

    return all_preds, all_probs


def main():
    """Main function for prediction."""
    args = parse_args()

    # Check input parameters
    if not os.path.exists(args.input_folder):
        print(f"Error: Input folder '{args.input_folder}' does not exist.")
        sys.exit(1)

    if not os.path.exists(args.config):
        print(f"Error: Config file '{args.config}' does not exist.")
        sys.exit(1)

    # Creating output directory
    output_dir = os.path.dirname(args.output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load config
    if args.config.endswith('.json'):
        with open(args.config, 'r') as f:
            config = json.load(f)
            # Extract img_size from config
            img_size = config.get('img_size', 256)
    else:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            # Extract img_size from config
            img_size = config.get('model', {}).get('img_size', 256)

    #  Thresholds
    if args.threshold is not None:
        threshold = args.threshold
    elif 'thresholds' in config:
        # Za ensemble koristimo prosečan threshold
        threshold = sum(config['thresholds']) / len(config['thresholds'])
    else:
        threshold = 0.5

    print(f"Using classification threshold: {threshold}")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Test loader
    test_loader, image_names = get_test_loader(
        args.input_folder,
        img_size,
        batch_size=args.batch_size
    )

    # Load model
    print(f"Loading model from {args.config}...")
    model = load_ensemble(args.config, device)

    # Prediction
    predictions, probabilities = predict(model, test_loader, device, threshold)

    # Create DataFrame-a with results
    results = pd.DataFrame({
        'image_name': image_names,
        'target': predictions
    })

    # Save results
    results.to_csv(args.output_csv, index=False)
    print(f"Predictions saved to {args.output_csv}")

    # Save detailed results with probabilities (optional)
    detailed_output = args.output_csv.replace('.csv', '_detailed.csv')
    detailed_results = pd.DataFrame({
        'image_name': image_names,
        'target': predictions,
        'probability': probabilities
    })
    detailed_results.to_csv(detailed_output, index=False)
    print(f"Detailed predictions saved to {detailed_output}")


if __name__ == "__main__":
    main()