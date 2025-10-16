# Core Module

The `bpd_ui.core` module provides application logic for data management, image processing, ROI extraction, and inference orchestration.

## Overview

This module handles:

- Excel-based manifest management
- Image loading (PNG, JPEG, DICOM)
- ROI extraction and refinement
- Model caching and batch inference
- Binary classification metrics
- Asynchronous task execution

## Data Management

### Manifest Operations

```{eval-rst}
.. autofunction:: bpd_ui.core.read_manifest
```

```{eval-rst}
.. autofunction:: bpd_ui.core.upsert_manifest
```

```{eval-rst}
.. autofunction:: bpd_ui.core.manifest_path
```

## Image I/O

### Loading Images

```{eval-rst}
.. autofunction:: bpd_ui.core.load_image_for_display
```

```{eval-rst}
.. autofunction:: bpd_ui.core.load_image_for_model
```

## Model Management

```{eval-rst}
.. autoclass:: bpd_ui.core.ModelManager
   :members:
   :undoc-members:
   :show-inheritance:
```

## Task System

```{eval-rst}
.. autofunction:: bpd_ui.core.submit
```

## Submodules

For advanced usage, import from specific submodules:

### ROI Service

```{eval-rst}
.. autoclass:: bpd_ui.core.roi_service.SegCache
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autofunction:: bpd_ui.core.roi_service.auto_chest_mask
```

```{eval-rst}
.. autofunction:: bpd_ui.core.roi_service.bbox_from_mask
```

```{eval-rst}
.. autofunction:: bpd_ui.core.roi_service.refine_with_scribbles
```

### Metrics

```{eval-rst}
.. autofunction:: bpd_ui.core.metrics.compute_binary_metrics
```

```{eval-rst}
.. autofunction:: bpd_ui.core.metrics.metrics_to_text
```

### Tasks

```{eval-rst}
.. autoclass:: bpd_ui.core.tasks.TaskSignals
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: bpd_ui.core.tasks.Task
   :members:
   :undoc-members:
   :show-inheritance:
```

### Deskewing

```{eval-rst}
.. autofunction:: bpd_ui.core.deskew.rectify_if_scan
```

## Usage Examples

### Working with Manifests

```python
from bpd_ui.core import read_manifest, upsert_manifest

# Read existing manifest
df = read_manifest("prepared/manifest.xlsx")
print(f"Found {len(df)} entries")

# Add new preprocessing result
upsert_manifest(
    manifest_path="prepared/manifest.xlsx",
    patient_id="PT001",
    image_relpath="PT001/images/img_001.png",
    roi_relpath="PT001/masks/roi_001.png",
    grade_label="moderate",
    eval_split="train"
)
```

### Loading Images

```python
from bpd_ui.core import load_image_for_display, load_image_for_model

# For GUI display (maintains aspect ratio)
qimage = load_image_for_display("chest_xray.dcm")

# For model inference (returns PIL Image)
pil_img = load_image_for_model("chest_xray.dcm")
```

### Using ModelManager

```python
from bpd_ui.core import ModelManager
from PIL import Image

# Create manager (caches loaded models)
mm = ModelManager("cpu")

# Run inference
img = Image.open("chest_xray.jpg")
prob, logit = mm.predict("bpd_xrv_progfreeze_lp_cutmix", img)

print(f"P(Moderate/Severe BPD): {prob:.4f}")
```

### Computing Metrics

```python
from bpd_ui.core.metrics import compute_binary_metrics, metrics_to_text
import numpy as np

# Ground truth and predictions
y_true = np.array([1, 0, 1, 1, 0])
y_proba = np.array([0.8, 0.2, 0.9, 0.6, 0.3])

# Compute metrics
metrics = compute_binary_metrics(y_true, y_proba)

# Format as text
report = metrics_to_text(metrics)
print(report)
```

### ROI Extraction

```python
from bpd_ui.core.roi_service import auto_chest_mask, refine_with_scribbles
import numpy as np

# Automatic chest mask
img_array = np.array(pil_image)
rough_mask = auto_chest_mask(img_array)

# Refine with user scribbles
seed_points = [(100, 150), (200, 180)]  # Foreground seeds
refined_mask = refine_with_scribbles(img_array, rough_mask, seed_points)
```

### Asynchronous Tasks

```python
from bpd_ui.core import submit
from bpd_ui.core.tasks import Task
from PySide6.QtCore import QObject

class MyWorker(QObject):
    def long_running_task(self):
        # Define task function
        def work(progress_callback):
            for i in range(10):
                progress_callback.emit(i * 10)
                # ... do work ...
            return "Result"

        # Submit to thread pool
        task = submit(work, on_success=self.handle_result)

    def handle_result(self, result):
        print(f"Task completed: {result}")
```

## Data Format

### Manifest Schema

The Excel manifest (`prepared/manifest.xlsx`) has the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `patient_id` | str | Patient identifier |
| `random_id` | str | Randomized/anonymized ID |
| `image_relpath` | str | Relative path to image (from `<root>/prepared/`) |
| `roi_relpath` | str | Relative path to ROI mask |
| `grade_label` | str | Operator annotation: `{no_bpd, mild, moderate, severe}` |
| `preproc_json` | str | JSON metadata (preprocessing parameters) |
| `eval_split` | str | Dataset split: `{train, val, test}` |
| `timestamp` | datetime | Last modification timestamp |

**Note**: While `grade_label` captures 4-level operator grading, model inference outputs binary classification (Moderate/Severe vs No/Mild).

### Folder Structure

```
<root>/
  prepared/
    <patient_id>/
      images/          # Original or standardized images
      masks/           # Binary ROI masks (.png)
    manifest.xlsx      # Central manifest
  prepared/eval/
    <YYYYMMDD-HHMMSS>/
      predictions.csv
      metrics.json
      metrics.txt
      plots/
        roc_curve.png
        confusion_matrix.png
```
