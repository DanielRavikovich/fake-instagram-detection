"""
src/evaluation.py
-----------------
Metric display, confusion matrices, ROC curves, and feature importance plots.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             RocCurveDisplay)


def results_to_dataframe(results):
    """Convert the results dict to a rounded DataFrame """
    results_df = pd.DataFrame(results).T.round(4)
    return results_df


def save_results(results_df, path='results/metrics.csv'):
    """Save the metrics DataFrame to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    results_df.to_csv(path)
    print(f'Saved {path}')


def plot_confusion_matrices(models, predictions, figures_dir='figures'):
    """
    Plot confusion matrices for all models side by side.
    """
    os.makedirs(figures_dir, exist_ok=True)

    from src.models import SCALED_MODELS

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4))
    if len(models) == 1:
        axes = [axes]

    for idx, (name, model) in enumerate(models.items()):
        y_pred = predictions[name]['y_pred']
        y_test = predictions[name]['y_test']

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=['Real', 'Fake'])
        disp.plot(ax=axes[idx], colorbar=False, cmap='Blues')
        axes[idx].set_title(name)

    plt.tight_layout()
    out = os.path.join(figures_dir, 'confusion_matrices.png')
    plt.savefig(out, dpi=150)
    plt.show()
    print(f'Saved {out}')


def plot_roc_curves(models, predictions, figures_dir='figures'):
    """
    Plot ROC curves for all models.
    """
    os.makedirs(figures_dir, exist_ok=True)

    from sklearn.metrics import roc_curve, roc_auc_score

    fig, ax = plt.subplots(figsize=(8, 6))

    for name in models:
        y_test = predictions[name]['y_test']
        y_prob = predictions[name]['y_prob']
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', label='Random (AUC=0.5)')
    ax.set_title('ROC Curve Comparison')
    ax.legend()
    plt.tight_layout()
    out = os.path.join(figures_dir, 'roc_curves.png')
    plt.savefig(out, dpi=150)
    plt.show()
    print(f'Saved {out}')


def plot_feature_importance(rf_model, feature_names, figures_dir='figures'):
    """
    Plot Random Forest feature importance.
    """
    os.makedirs(figures_dir, exist_ok=True)

    importances = pd.Series(
        rf_model.feature_importances_,
        index=feature_names
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    importances.plot(kind='barh', ax=ax, color='steelblue')
    ax.set_title('Random Forest - Feature Importance')
    ax.set_xlabel('Importance Score')
    plt.tight_layout()
    out = os.path.join(figures_dir, 'feature_importance.png')
    plt.savefig(out, dpi=150)
    plt.show()
    print(importances.sort_values(ascending=False))
    print(f'Saved {out}')
