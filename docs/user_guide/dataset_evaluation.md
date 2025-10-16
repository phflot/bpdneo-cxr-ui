# Dataset Evaluation Workflow

This guide covers batch evaluation of preprocessed datasets with binary classification metrics and visualizations.

## Overview

The dataset evaluation workflow enables you to:

1. Load a preprocessed manifest with labeled images
2. Select a dataset split (train/val/test)
3. Run batch inference with a pretrained model
4. Compute binary classification metrics
5. Generate reports and visualizations

**Output**: Predictions CSV, metrics JSON/TXT, ROC curve, confusion matrix.

## Prerequisites

- Preprocessed dataset with manifest ({doc}`preprocessing`)
- Ground truth labels in `grade_label` column
- (Optional) `eval_split` column for filtering

## Step-by-Step Guide

### 1. Launch the GUI

```bash
python apps/gui_app.py
```

Navigate to the **Dataset Evaluation** tab.

### 2. Load Manifest

1. Click **"Load Manifest..."**
2. Select `manifest.xlsx` from your prepared directory
3. The application displays:
   - Total number of entries
   - Available splits (if `eval_split` column exists)
   - Ground truth label distribution

### 3. Select Evaluation Split

**Option A: All Data**
- Select **"all"** from the dropdown
- Evaluates all entries in the manifest

**Option B: Specific Split**
- Select `train`, `val`, or `test` from the dropdown
- Only entries with matching `eval_split` value are evaluated

The status bar shows: "Loaded 150 images from split 'test'"

### 4. Select Model

1. Choose model from the dropdown:
   - `bpd_xrv_progfreeze_lp_cutmix` (default, AUROC 0.783)
   - `bpd_xrv_progfreeze` (AUROC 0.775)
   - `bpd_xrv_fullft` (AUROC 0.761)
   - `bpd_rgb_progfreeze` (AUROC 0.717)

2. Model info displays:
   - Architecture (ResNet50 + XRV or ImageNet)
   - Validation AUROC
   - Preprocessing pipeline

### 5. Run Evaluation

1. Click **"Run Evaluation"**
2. Progress bar updates during batch inference
3. Status messages show current progress:
   - "Downloading model weights..." (first time only)
   - "Loading model..."
   - "Processing image 45/150..."

**Note**: First-time model loading downloads ~100 MB weights from Google Drive. Subsequent runs use cached weights.

### 6. View Results

After completion, results are displayed in the results panel:

#### Binary Classification Metrics

```
Binary Classification Metrics
==============================

AUROC: 0.7654
AUPRC: 0.8123

Confusion Matrix (Threshold=0.5):
                 Predicted Neg  Predicted Pos
Actual Neg              45              12
Actual Pos               8              85

Accuracy:      0.8667
Sensitivity:   0.9140
Specificity:   0.7895
PPV:           0.8763
NPV:           0.8491
```

#### Saved Files

Results are saved to `<root>/prepared/eval/<timestamp>/`:

**Files created**:
- `predictions.csv` - Per-image predictions with columns:
  - `patient_id`, `image_relpath`, `ground_truth`, `predicted_proba`, `predicted_label`
- `metrics.json` - Structured metrics (for programmatic use)
- `metrics.txt` - Human-readable report (copy of console output)
- `plots/roc_curve.png` - ROC curve with AUC annotation
- `plots/confusion_matrix.png` - Confusion matrix heatmap

#### ROC Curve

The ROC curve plot shows:
- True Positive Rate vs False Positive Rate
- AUC value annotated
- Optimal threshold point marked
- Diagonal reference line

#### Confusion Matrix

The confusion matrix heatmap shows:
- Rows: Actual labels (No/Mild, Moderate/Severe)
- Columns: Predicted labels
- Cell values: Count of predictions
- Color intensity: Relative frequency

## Label Mapping

The evaluation workflow converts 4-level operator grades to binary labels:

**Binary mapping**:
- `no_bpd` → **No/Mild** (negative class, 0)
- `mild` → **No/Mild** (negative class, 0)
- `moderate` → **Moderate/Severe** (positive class, 1)
- `severe` → **Moderate/Severe** (positive class, 1)

This matches the training setup for all pretrained models.

## Programmatic Evaluation

Use the Python API directly for scripting or automation:

```python
from bpd_ui.core import read_manifest, ModelManager, compute_binary_metrics
import numpy as np

# Load manifest
df = read_manifest("prepared/manifest.xlsx")
test_df = df[df["eval_split"] == "test"]

# Initialize model manager
mm = ModelManager("cpu")

# Run inference
predictions = []
for idx, row in test_df.iterrows():
    img = load_image_for_model(row["image_relpath"])
    prob, logit = mm.predict("bpd_xrv_progfreeze_lp_cutmix", img)
    predictions.append(prob)

# Compute metrics
y_true = (test_df["grade_label"].isin(["moderate", "severe"])).astype(int).values
y_proba = np.array(predictions)
metrics = compute_binary_metrics(y_true, y_proba)

print(f"Test AUROC: {metrics['auroc']:.4f}")
```

## Understanding Metrics

### AUROC (Area Under ROC Curve)

- Range: 0.5 (random) to 1.0 (perfect)
- Interpretation: Probability that the model ranks a random positive higher than a random negative
- **Good**: > 0.7
- **Excellent**: > 0.8

### AUPRC (Area Under Precision-Recall Curve)

- Range: 0.0 to 1.0
- More informative than AUROC for imbalanced datasets
- **Good**: > 0.6 (depends on class balance)

### Sensitivity (Recall, True Positive Rate)

- Formula: TP / (TP + FN)
- Interpretation: Proportion of actual positives correctly identified
- **High sensitivity**: Few false negatives (missed cases)

### Specificity (True Negative Rate)

- Formula: TN / (TN + FP)
- Interpretation: Proportion of actual negatives correctly identified
- **High specificity**: Few false positives (false alarms)

### PPV (Positive Predictive Value, Precision)

- Formula: TP / (TP + FP)
- Interpretation: Proportion of positive predictions that are correct
- Depends on prevalence (class balance)

### NPV (Negative Predictive Value)

- Formula: TN / (TN + FN)
- Interpretation: Proportion of negative predictions that are correct

## Comparing Models

To compare multiple models on the same dataset:

1. Run evaluation for each model
2. Save results to separate output directories
3. Compare AUROC values

**Example using Python API**:
```python
from bpd_ui.core import read_manifest, ModelManager, compute_binary_metrics
import numpy as np
import json
from pathlib import Path

models_to_compare = [
    "bpd_xrv_progfreeze_lp_cutmix",
    "bpd_xrv_fullft",
    "bpd_rgb_progfreeze"
]

# Load test data
df = read_manifest("prepared/manifest.xlsx")
test_df = df[df["eval_split"] == "test"]
mm = ModelManager("cpu")

# Evaluate each model
for model_name in models_to_compare:
    predictions = []
    for idx, row in test_df.iterrows():
        img = load_image_for_model(row["image_relpath"])
        prob, _ = mm.predict(model_name, img)
        predictions.append(prob)

    # Compute metrics
    y_true = (test_df["grade_label"].isin(["moderate", "severe"])).astype(int).values
    y_proba = np.array(predictions)
    metrics = compute_binary_metrics(y_true, y_proba)

    # Save results
    output_dir = Path(f"eval/{model_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"{model_name}: AUROC {metrics['auroc']:.4f}")
```

## Troubleshooting

### Issue: "No images found for split 'test'"

**Solutions**:
- Check that `eval_split` column exists in manifest
- Verify split values are lowercase (`test`, not `Test`)
- Use `all` split to evaluate entire dataset

### Issue: Low AUROC on training set

**Cause**: Likely overfitting (model memorized training data)

**Solutions**:
- Always evaluate on held-out test set
- Do NOT use training AUROC for model selection

### Issue: Evaluation is slow

**Solutions**:
- Ensure images are preprocessed (ROI masks extracted)
- Use smaller image sizes if possible
- Evaluation runs on CPU; GPU support not required

### Issue: Manifest has missing `grade_label` values

**Solutions**:
- Filter DataFrame to remove rows with `pd.isna(grade_label)`
- Complete preprocessing workflow for all images
- Check that grades were saved correctly

## Next Steps

- {doc}`single_image` - Test individual images interactively
- {doc}`../api/core` - ModelManager API reference
- {doc}`../api/models` - Available models documentation
