import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    confusion_matrix,
    roc_curve,
)


def compute_binary_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    return {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "accuracy": float(acc),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ppv": float(ppv),
        "npv": float(npv),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
        },
        "n": len(y_true),
        "threshold": threshold,
    }


def metrics_to_text(metrics):
    lines = []
    lines.append(f"Sample size: {metrics['n']}")
    lines.append(f"Threshold: {metrics['threshold']:.3f}")
    lines.append("")
    lines.append(f"AUROC: {metrics['auroc']:.4f}")
    lines.append(f"AUPRC: {metrics['auprc']:.4f}")
    lines.append(f"Accuracy: {metrics['accuracy']:.4f}")
    lines.append("")
    lines.append(f"Sensitivity (Recall): {metrics['sensitivity']:.4f}")
    lines.append(f"Specificity: {metrics['specificity']:.4f}")
    lines.append(f"PPV (Precision): {metrics['ppv']:.4f}")
    lines.append(f"NPV: {metrics['npv']:.4f}")
    lines.append("")
    cm = metrics["confusion_matrix"]
    lines.append("Confusion Matrix:")
    lines.append(f"  TN: {cm['tn']}  FP: {cm['fp']}")
    lines.append(f"  FN: {cm['fn']}  TP: {cm['tp']}")
    return "\n".join(lines)
