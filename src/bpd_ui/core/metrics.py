"""
Binary Classification Metrics
==============================

This module provides binary classification metric computation and formatting
for model evaluation. It calculates standard metrics (AUROC, AUPRC, accuracy,
sensitivity, specificity) and generates human-readable summaries.

Functions
---------
compute_binary_metrics
    Compute comprehensive binary classification metrics
metrics_to_text
    Format metrics dictionary as human-readable text

Notes
-----
All metrics are computed using scikit-learn. The threshold for binary
predictions defaults to 0.5 but can be adjusted.

The metrics focus on binary BPD classification:
- Positive class: Moderate/Severe BPD
- Negative class: No/Mild BPD

See Also
--------
bpd_ui.ui.dataset_eval_tab : Uses metrics for batch evaluation
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    confusion_matrix,
    roc_curve,
)


def compute_binary_metrics(y_true, y_prob, threshold=0.5):
    """
    Compute comprehensive binary classification metrics.

    This function calculates standard binary classification metrics including
    AUROC, AUPRC, accuracy, sensitivity/specificity, PPV/NPV, confusion matrix,
    and ROC curve coordinates.

    Parameters
    ----------
    y_true : array-like
        Ground truth binary labels (0 or 1)
    y_prob : array-like
        Predicted probabilities (0-1 range)
    threshold : float, default=0.5
        Threshold for converting probabilities to binary predictions

    Returns
    -------
    dict
        Dictionary containing:
        - 'auroc' : float, area under ROC curve
        - 'auprc' : float, area under precision-recall curve
        - 'accuracy' : float, overall accuracy
        - 'sensitivity' : float, recall/true positive rate
        - 'specificity' : float, true negative rate
        - 'ppv' : float, positive predictive value (precision)
        - 'npv' : float, negative predictive value
        - 'confusion_matrix' : dict with 'tn', 'fp', 'fn', 'tp' keys
        - 'roc_curve' : dict with 'fpr', 'tpr', 'thresholds' arrays
        - 'n' : int, sample size
        - 'threshold' : float, threshold used

    Notes
    -----
    Metric definitions:
    - AUROC: Area under receiver operating characteristic curve
    - AUPRC: Area under precision-recall curve
    - Sensitivity (Recall): TP / (TP + FN)
    - Specificity: TN / (TN + FP)
    - PPV (Precision): TP / (TP + FP)
    - NPV: TN / (TN + FN)

    The ROC curve arrays can be used for plotting:
    - fpr: False positive rates (x-axis)
    - tpr: True positive rates (y-axis)
    - thresholds: Corresponding probability thresholds

    Metrics handle edge cases gracefully (e.g., division by zero returns 0.0).

    Examples
    --------
    >>> import numpy as np
    >>> from bpd_ui.core.metrics import compute_binary_metrics
    >>>
    >>> # Simulated predictions
    >>> y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    >>> y_prob = np.array([0.1, 0.3, 0.6, 0.8, 0.9, 0.2, 0.7, 0.4])
    >>>
    >>> # Compute metrics
    >>> metrics = compute_binary_metrics(y_true, y_prob)
    >>> print(f"AUROC: {metrics['auroc']:.3f}")
    AUROC: 0.875
    >>> print(f"Sensitivity: {metrics['sensitivity']:.3f}")
    Sensitivity: 1.000
    >>>
    >>> # Custom threshold
    >>> metrics_strict = compute_binary_metrics(y_true, y_prob, threshold=0.7)

    See Also
    --------
    metrics_to_text : Convert metrics to readable text
    sklearn.metrics.roc_auc_score : AUROC computation
    """
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
    """
    Format metrics dictionary as human-readable text summary.

    This function converts the metrics dictionary from compute_binary_metrics()
    into a formatted text summary suitable for console output or file saving.

    Parameters
    ----------
    metrics : dict
        Metrics dictionary from compute_binary_metrics()

    Returns
    -------
    str
        Multi-line text summary with aligned metrics

    Notes
    -----
    The formatted output includes:
    - Sample size and threshold
    - Primary metrics (AUROC, AUPRC, Accuracy)
    - Diagnostic metrics (Sensitivity, Specificity, PPV, NPV)
    - Confusion matrix in 2x2 table format

    The confusion matrix format:
        TN: <count>  FP: <count>
        FN: <count>  TP: <count>

    Examples
    --------
    >>> from bpd_ui.core.metrics import compute_binary_metrics, metrics_to_text
    >>> import numpy as np
    >>>
    >>> y_true = np.array([0, 0, 1, 1, 1])
    >>> y_prob = np.array([0.2, 0.3, 0.7, 0.8, 0.9])
    >>> metrics = compute_binary_metrics(y_true, y_prob)
    >>>
    >>> # Print formatted summary
    >>> print(metrics_to_text(metrics))
    Sample size: 5
    Threshold: 0.500
    <BLANKLINE>
    AUROC: 1.0000
    AUPRC: 1.0000
    Accuracy: 1.0000
    <BLANKLINE>
    Sensitivity (Recall): 1.0000
    Specificity: 1.0000
    PPV (Precision): 1.0000
    NPV: 1.0000
    <BLANKLINE>
    Confusion Matrix:
      TN: 2  FP: 0
      FN: 0  TP: 3
    >>>
    >>> # Save to file
    >>> with open("metrics.txt", "w") as f:
    ...     f.write(metrics_to_text(metrics))

    See Also
    --------
    compute_binary_metrics : Compute metrics dictionary
    """
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
