# Data Format Specification

This document specifies the data formats used by BPDneo-CXR for datasets, manifests, and outputs.

## Folder Structure

BPDneo-CXR uses a standardized folder structure for preprocessed datasets:

```
<root>/
  prepared/
    <patient_id>/
      images/              # Original or standardized images
        img_001.png
        img_002.png
        ...
      masks/               # Binary ROI masks
        roi_001.png
        roi_002.png
        ...
    <patient_id>/
      images/
      masks/
    ...
    manifest.xlsx          # Central Excel manifest
  prepared/eval/
    <timestamp>/
      predictions.csv
      metrics.json
      metrics.txt
      plots/
        roc_curve.png
        confusion_matrix.png
```

## Manifest Format

The manifest is an Excel file (`manifest.xlsx`) with the following schema:

### Required Columns

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `patient_id` | str | Patient identifier | `PT001` |
| `image_relpath` | str | Path to image relative to `prepared/` | `PT001/images/img_001.png` |
| `roi_relpath` | str | Path to ROI mask relative to `prepared/` | `PT001/masks/roi_001.png` |
| `grade_label` | str | Operator annotation (4-level) | `moderate` |
| `timestamp` | datetime | Last modification timestamp | `2025-01-15 14:32:18` |

### Optional Columns

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `random_id` | str | Anonymized/randomized patient ID | `R12345` |
| `eval_split` | str | Dataset split | `test` |
| `preproc_json` | str | JSON-encoded preprocessing metadata | `{"method": "grabcut", ...}` |

### Column Details

#### `grade_label`

**Valid values**:
- `no_bpd` - No bronchopulmonary dysplasia
- `mild` - Mild BPD
- `moderate` - Moderate BPD
- `severe` - Severe BPD

**Binary mapping for model evaluation**:
- `{no_bpd, mild}` → Class 0 (negative, "No/Mild BPD")
- `{moderate, severe}` → Class 1 (positive, "Moderate/Severe BPD")

#### `eval_split`

**Valid values**:
- `train` - Training set (not used for evaluation)
- `val` - Validation set (hyperparameter tuning, model selection)
- `test` - Test set (final evaluation, reporting)

**Note**: If `eval_split` is not specified, all rows are treated as one dataset.

#### `preproc_json`

JSON-encoded string containing preprocessing metadata:

```json
{
  "method": "grabcut",
  "n_seeds": 5,
  "border_vertices": 12,
  "auto_deskew": true,
  "timestamp": "2025-01-15T14:32:18"
}
```

**Purpose**: Track preprocessing provenance for reproducibility.

### Example Manifest

| patient_id | image_relpath | roi_relpath | grade_label | eval_split | timestamp |
|------------|---------------|-------------|-------------|------------|-----------|
| PT001 | PT001/images/img_001.png | PT001/masks/roi_001.png | moderate | train | 2025-01-15 10:00:00 |
| PT001 | PT001/images/img_002.png | PT001/masks/roi_002.png | severe | train | 2025-01-15 10:05:00 |
| PT002 | PT002/images/img_001.png | PT002/masks/roi_001.png | mild | test | 2025-01-15 11:00:00 |

## Image Formats

### Supported Input Formats

- **PNG** (`.png`) - Lossless, recommended for processed images
- **JPEG** (`.jpg`, `.jpeg`) - Lossy compression, common for scans
- **DICOM** (`.dcm`) - Medical imaging standard

### Image Requirements

- **Modality**: Chest X-ray (frontal view)
- **Bit depth**: 8-bit or 16-bit grayscale (RGB supported but converted)
- **Resolution**: No strict requirement (resized to 512×512 during preprocessing)
- **Aspect ratio**: No strict requirement (preserves aspect, crops/pads to square)

### DICOM Requirements

Required DICOM tags:
- `PixelData` (7FE0,0010) - Image pixel array
- `Rows` (0028,0010) - Image height
- `Columns` (0028,0011) - Image width

Optional but recommended:
- `PhotometricInterpretation` (0028,0004) - Pixel interpretation (`MONOCHROME1` or `MONOCHROME2`)
- `RescaleSlope` (0028,1053) - Pixel value scaling
- `RescaleIntercept` (0028,1052) - Pixel value offset

**Handling**:
- `MONOCHROME1` (inverted): Automatically inverted to `MONOCHROME2`
- Rescaling: Applied if `RescaleSlope` and `RescaleIntercept` are present

## ROI Mask Format

### Format Specification

- **File format**: PNG (8-bit grayscale)
- **Values**:
  - `0` - Background (excluded from ROI)
  - `255` - Foreground (included in ROI)
- **Size**: Must match corresponding image size

### Example

```python
from PIL import Image
import numpy as np

# Load mask
mask = np.array(Image.open("roi_001.png"))

# Verify binary
assert set(np.unique(mask)).issubset({0, 255}), "Mask must be binary"

# Create mask programmatically
h, w = 512, 512
new_mask = np.zeros((h, w), dtype=np.uint8)
new_mask[100:400, 100:400] = 255  # ROI region
Image.fromarray(new_mask).save("roi_new.png")
```

## Evaluation Output Formats

### Predictions CSV

**File**: `<root>/prepared/eval/<timestamp>/predictions.csv`

**Schema**:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `patient_id` | str | Patient identifier | `PT001` |
| `image_relpath` | str | Image path | `PT001/images/img_001.png` |
| `ground_truth` | int | Binary ground truth (0 or 1) | `1` |
| `predicted_proba` | float | Predicted probability P(class 1) | `0.7823` |
| `predicted_label` | int | Binary prediction (threshold=0.5) | `1` |

**Example**:

```csv
patient_id,image_relpath,ground_truth,predicted_proba,predicted_label
PT001,PT001/images/img_001.png,1,0.7823,1
PT001,PT001/images/img_002.png,1,0.8654,1
PT002,PT002/images/img_001.png,0,0.2341,0
```

### Metrics JSON

**File**: `<root>/prepared/eval/<timestamp>/metrics.json`

**Schema**:

```json
{
  "auroc": 0.7654,
  "auprc": 0.8123,
  "accuracy": 0.8667,
  "sensitivity": 0.9140,
  "specificity": 0.7895,
  "ppv": 0.8763,
  "npv": 0.8491,
  "confusion_matrix": [[45, 12], [8, 85]],
  "threshold": 0.5,
  "n_samples": 150,
  "n_positive": 93,
  "n_negative": 57,
  "model": "bpd_xrv_progfreeze_lp_cutmix",
  "timestamp": "2025-01-15T14:32:18"
}
```

**Field descriptions**:

- `auroc`: Area under ROC curve (range: 0.5-1.0)
- `auprc`: Area under precision-recall curve (range: 0.0-1.0)
- `accuracy`: (TP + TN) / (TP + TN + FP + FN)
- `sensitivity`: TP / (TP + FN), aka recall or true positive rate
- `specificity`: TN / (TN + FP), aka true negative rate
- `ppv`: TP / (TP + FP), aka positive predictive value or precision
- `npv`: TN / (TN + FN), aka negative predictive value
- `confusion_matrix`: [[TN, FP], [FN, TP]]

### Metrics TXT

**File**: `<root>/prepared/eval/<timestamp>/metrics.txt`

Human-readable text report:

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

Dataset Statistics:
  Total samples: 150
  Positive samples: 93 (62.0%)
  Negative samples: 57 (38.0%)

Model: bpd_xrv_progfreeze_lp_cutmix
Timestamp: 2025-01-15 14:32:18
```

## Programmatic Access

### Reading Manifest

```python
from bpd_ui.core import read_manifest

df = read_manifest("prepared/manifest.xlsx")

# Filter by split
test_df = df[df["eval_split"] == "test"]

# Filter by grade
moderate_severe_df = df[df["grade_label"].isin(["moderate", "severe"])]
```

### Writing Manifest

```python
from bpd_ui.core import upsert_manifest

upsert_manifest(
    manifest_path="prepared/manifest.xlsx",
    patient_id="PT001",
    image_relpath="PT001/images/img_001.png",
    roi_relpath="PT001/masks/roi_001.png",
    grade_label="moderate",
    eval_split="train"
)
```

### Loading Predictions

```python
import pandas as pd

preds = pd.read_csv("prepared/eval/20250115-143218/predictions.csv")

# Get predictions for specific patient
pt001_preds = preds[preds["patient_id"] == "PT001"]

# Compute custom metrics
from sklearn.metrics import roc_auc_score

auroc = roc_auc_score(preds["ground_truth"], preds["predicted_proba"])
print(f"AUROC: {auroc:.4f}")
```

### Loading Metrics

```python
import json

with open("prepared/eval/20250115-143218/metrics.json") as f:
    metrics = json.load(f)

print(f"Model: {metrics['model']}")
print(f"AUROC: {metrics['auroc']:.4f}")
print(f"Accuracy: {metrics['accuracy']:.4f}")
```

## Validation

### Manifest Validation

Ensure manifest integrity:

```python
from bpd_ui.core import read_manifest
from pathlib import Path

df = read_manifest("prepared/manifest.xlsx")

# Check required columns
required_cols = ["patient_id", "image_relpath", "roi_relpath", "grade_label"]
assert all(col in df.columns for col in required_cols), "Missing required columns"

# Check file existence
root = Path("prepared")
for idx, row in df.iterrows():
    img_path = root / row["image_relpath"]
    roi_path = root / row["roi_relpath"]
    assert img_path.exists(), f"Image not found: {img_path}"
    assert roi_path.exists(), f"ROI not found: {roi_path}"

# Check grade labels
valid_grades = {"no_bpd", "mild", "moderate", "severe"}
invalid = df[~df["grade_label"].isin(valid_grades)]
assert len(invalid) == 0, f"Invalid grades: {invalid['grade_label'].unique()}"

print("Manifest validation passed!")
```

### ROI Mask Validation

Ensure ROI masks are binary:

```python
from PIL import Image
import numpy as np
from pathlib import Path

def validate_roi_mask(mask_path):
    mask = np.array(Image.open(mask_path))
    unique_vals = np.unique(mask)
    assert set(unique_vals).issubset({0, 255}), f"Non-binary values: {unique_vals}"
    return True

# Validate all ROI masks
root = Path("prepared")
df = read_manifest("prepared/manifest.xlsx")

for idx, row in df.iterrows():
    roi_path = root / row["roi_relpath"]
    validate_roi_mask(roi_path)

print("All ROI masks are valid!")
```

## Next Steps

- {doc}`preprocessing` - Build a preprocessed dataset
- {doc}`dataset_evaluation` - Run batch evaluation
- {doc}`../api/core` - Core module API for data operations
