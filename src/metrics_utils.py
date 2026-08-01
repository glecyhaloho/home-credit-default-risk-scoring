"""Shared evaluation helpers: KS statistic, ROC/confusion plots, metrics persistence."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
from config import FIGURES_DIR, MODELS_DIR


def ks_statistic(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    ks = np.max(tpr - fpr)
    ks_threshold = thresholds[np.argmax(tpr - fpr)]
    return ks, ks_threshold, fpr, tpr, thresholds


def evaluate_model(name, y_true, y_prob, threshold=None):
    auc = roc_auc_score(y_true, y_prob)
    ks, ks_thresh, fpr, tpr, thresholds = ks_statistic(y_true, y_prob)
    thr = threshold if threshold is not None else ks_thresh
    y_pred = (y_prob >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    metrics = {
        "model": name,
        "auc": float(auc),
        "ks": float(ks),
        "chosen_threshold": float(thr),
        "precision_at_threshold": float(precision),
        "recall_at_threshold": float(recall),
        "f1_at_threshold": float(f1),
        "confusion_matrix": cm.tolist(),  # [[TN, FP], [FN, TP]]
    }
    print(json.dumps(metrics, indent=2))
    return metrics, fpr, tpr


def save_metrics(all_metrics, filename="metrics.json"):
    path = MODELS_DIR / filename
    with open(path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print("Saved metrics:", path)


def plot_roc_comparison(curves, filename="roc_curve_comparison.png"):
    plt.figure(figsize=(6, 6))
    for name, fpr, tpr, auc in curves:
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.tight_layout()
    out = FIGURES_DIR / filename
    plt.savefig(out, dpi=150)
    plt.close()
    print("Saved:", out)


def plot_confusion_matrix(cm, name, filename):
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
    plt.xticks([0, 1], ["Pred: Repay", "Pred: Default"])
    plt.yticks([0, 1], ["Actual: Repay", "Actual: Default"])
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    out = FIGURES_DIR / filename
    plt.savefig(out, dpi=150)
    plt.close()
    print("Saved:", out)
