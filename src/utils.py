import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, precision_recall_curve
)


def calculate_metrics(y_true, y_pred, y_prob=None):
    """
    Calculating metrics for model evaluation.

    Args:
        y_true (array): Real labels
        y_pred (array): Predicted labels
        y_prob (array, optional): Predicted probabilities

    Returns:
        dict: Metrics dictionary
    """
    metrics = {}

    # Base metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)

    # Confuzion matrix
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


def optimize_threshold(y_true, y_prob, target_recall=0.9, min_precision=0.6):
    """
    Optimizes threshold for maximization of F1 score with target recall.
    """
    # Calculation of precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    # ISPRAVKA: Provera da li su dimenzije usklađene
    # thresholds je uvek za 1 kraći od precision/recall
    if len(thresholds) < len(precision):
        precision = precision[:len(thresholds)]
        recall = recall[:len(thresholds)]

    # ALTERNATIVNO REŠENJE:
    # Dodajemo threshold=1.0 za recall=0 da izjednačimo dimenzije
    # thresholds = np.append(thresholds, 1.0)

    # Inicijalni threshold
    optimal_threshold = 0.5
    best_f1 = 0

    # Finding threshold which meets our target recall
    valid_indices = recall >= target_recall

    if np.any(valid_indices):
        # Pick valid threshold which maximizes precision
        valid_precision = precision[valid_indices]
        valid_recall = recall[valid_indices]
        valid_thresholds = thresholds[valid_indices]

        precision_indices = valid_precision >= min_precision

        if np.any(precision_indices):
            # Calculating F1 score
            f1_scores = 2 * (valid_precision[precision_indices] * valid_recall[precision_indices]) / (
                        valid_precision[precision_indices] + valid_recall[precision_indices] + 1e-10)

            # Find best F1 score which meets min_precision criteria
            best_idx = np.argmax(f1_scores)
            optimal_threshold = valid_thresholds[precision_indices][best_idx]
            best_f1 = f1_scores[best_idx]
        else:
            # If none of the F1 scores meets the criteria
            optimal_threshold = valid_thresholds[np.argmax(valid_precision)]
    else:
        # If we can not achieve target_recall
        optimal_threshold = thresholds[np.argmax(recall[:-1])] if len(recall) > len(thresholds) else thresholds[
            np.argmax(recall)]

    return optimal_threshold


def calculate_fairness_metrics(metrics_by_group):
    """
    Calculates fairness metrics for different model groups.

    Args:
        metrics_by_group (dict): Metrics for each model group

    Returns:
        dict: Fairness metrics
    """
    fairness = {}

    # Calculating lowest and highest value for every metric
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


def plot_metrics_by_group(metrics_by_group, output_path):
    """
    Plots metrics by group.

    Args:
        metrics_by_group (dict): Metrics for each model group
        output_path (str): Output path
    """
    metrics_to_plot = ['recall', 'precision', 'specificity', 'balanced_acc', 'f1']
    groups = list(metrics_by_group.keys())

    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(metrics_to_plot))
    width = 0.8 / len(groups)

    for i, group in enumerate(groups):
        values = [metrics_by_group[group][metric] for metric in metrics_to_plot]
        ax.bar(x + i * width - width * len(groups) / 2, values, width, label=group)

    ax.set_ylabel('Score')
    ax.set_title('Metrics by Group')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_validation_predictions(image_names, predictions, probabilities, output_path):
    """
    Saves validation predictions in a CSV file.

    Args:
        image_names (list): List of image names
        predictions (list): List of predictions
        probabilities (list): List of probabilities
        output_path (str): Output path
    """

    min_length = min(len(image_names), len(predictions), len(probabilities))

    image_names = image_names[:min_length]
    predictions = predictions[:min_length]
    probabilities = probabilities[:min_length]
    df = pd.DataFrame({
        'image_name': image_names,
        'target': predictions,
        'probability': probabilities
    })

    df[['image_name', 'target']].to_csv(output_path, index=False)
    print(f"Validation predictions saved to {output_path}")

    # Detailed CSV with probabilities
    detailed_path = output_path.replace('.csv', '_detailed.csv')
    df.to_csv(detailed_path, index=False)
    print(f"Detailed validation predictions with probabilities saved to {detailed_path}")


def plot_precision_recall_curve(y_true, y_prob, output_path, optimal_threshold=None):
    """
    Plots precision-recall curve and optimal threshold point.

    Args:
        y_true (array): Real labels
        y_prob (array): Predicted labels
        output_path (str): Output path
        optimal_threshold (float, optional): Optimal threshold
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, 'b-', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')

    if optimal_threshold is not None:
        idx = np.argmin(np.abs(thresholds - optimal_threshold)) if len(thresholds) > 0 else 0
        if idx < len(precision) and idx < len(recall):
            plt.plot(recall[idx], precision[idx], 'ro', markersize=8,
                     label=f'Threshold: {optimal_threshold:.2f}, Precision: {precision[idx]:.2f}, Recall: {recall[idx]:.2f}')
            plt.legend()

    plt.grid(True, alpha=0.3)
    plt.savefig(output_path)
    plt.close()