# Single Image Evaluation

This guide covers end-to-end prediction on individual chest X-ray images without dataset setup.

## Overview

The single image evaluation workflow enables you to:

1. Load any chest X-ray image (PNG, JPEG, DICOM)
2. Optionally extract ROI
3. Run inference with a pretrained model
4. View prediction probabilities instantly

**Use cases**:
- Quick testing of new images
- Exploratory analysis
- Demonstration or validation
- No dataset preprocessing required

## Step-by-Step Guide

### 1. Launch the GUI

```bash
python apps/gui_app.py
```

Navigate to the **Single Image Evaluation** tab.

### 2. Select Image

1. Click **"Select Image..."**
2. Browse to your chest X-ray file
3. Supported formats:
   - PNG (`.png`)
   - JPEG (`.jpg`, `.jpeg`)
   - DICOM (`.dcm`)

The image displays in the viewer with zoom/pan controls.

### 3. (Optional) Extract ROI

**If you want to restrict inference to a specific region**:

1. Click **"Enable ROI Extraction"** checkbox
2. Select **"Seed Mode"** from toolbar
3. Click on the lung region to place 3-5 seed points
4. (Optional) Switch to **"Border Mode"** and draw boundary
5. Click **"Extract ROI"**
6. Mask overlays on the image

**If you skip this step**: Full image is used for inference (after automatic preprocessing).

### 4. Select Model

Choose model from the dropdown:

- `bpd_xrv_progfreeze_lp_cutmix` (default, AUROC 0.783)
- `bpd_xrv_progfreeze` (AUROC 0.775)
- `bpd_xrv_fullft` (AUROC 0.761)
- `bpd_rgb_progfreeze` (AUROC 0.717)

### 5. Run Inference

1. Click **"Run Inference"**
2. Status bar shows: "Running inference..."
3. Model automatically:
   - Downloads weights (first time only)
   - Applies preprocessing (resize, normalize)
   - Runs forward pass
   - Computes probabilities

**Typical runtime**: < 1 second on CPU (after model is cached)

### 6. View Results

Results display in the results panel:

```
Prediction Results
==================

Binary Label: Moderate/Severe BPD

Probabilities:
  P(Moderate/Severe BPD): 0.7823
  P(No/Mild BPD):         0.2177

Model: bpd_xrv_progfreeze_lp_cutmix
Input: chest_xray_001.dcm
Timestamp: 2025-01-15 14:32:18
```

**Interpretation**:
- **Binary Label**: Predicted class (threshold = 0.5)
- **Probabilities**: Confidence for each class
- Higher probability = higher confidence

## Programmatic Usage

For batch processing or scripting, use the Python API:

### Basic Inference

```python
from bpd_ui.models import load_pretrained_model, get_preprocessing_transforms
from PIL import Image
import torch

# Load model and transforms
model = load_pretrained_model("bpd_xrv_progfreeze_lp_cutmix")
transform = get_preprocessing_transforms("bpd_xrv_progfreeze_lp_cutmix")

# Load image
img = Image.open("chest_xray.jpg")

# Preprocess
tensor = transform(img).unsqueeze(0)  # Add batch dimension

# Predict
model.eval()
with torch.no_grad():
    logits = model(tensor)
    prob = torch.sigmoid(logits).item()

# Interpret
if prob >= 0.5:
    prediction = "Moderate/Severe BPD"
else:
    prediction = "No/Mild BPD"

print(f"Prediction: {prediction}")
print(f"P(Moderate/Severe): {prob:.4f}")
print(f"P(No/Mild): {1-prob:.4f}")
```

### Using ModelManager (Recommended)

```python
from bpd_ui.core import ModelManager
from PIL import Image

# Create manager (caches loaded models)
mm = ModelManager("cpu")

# Load image
img = Image.open("chest_xray.jpg")

# Predict (handles preprocessing automatically)
prob, logit = mm.predict("bpd_xrv_progfreeze_lp_cutmix", img)

# Interpret
label = "Moderate/Severe BPD" if prob >= 0.5 else "No/Mild BPD"
print(f"Prediction: {label} (p={prob:.4f})")
```

## Working with DICOM Files

DICOM files are automatically detected and processed:

```python
from bpd_ui.core import load_image_for_model

# Load DICOM (handles pixel_array extraction)
img = load_image_for_model("chest_xray.dcm")

# Now use with ModelManager
mm = ModelManager("cpu")
prob, logit = mm.predict("bpd_xrv_progfreeze_lp_cutmix", img)
```

**DICOM metadata used**:
- `PixelData`: Extracted as NumPy array
- `PhotometricInterpretation`: Auto-inverts if `MONOCHROME1`
- `RescaleSlope`, `RescaleIntercept`: Applied if present

## Understanding Predictions

### Binary Classification

All models output binary predictions:

- **Class 0 (Negative)**: No/Mild BPD
  - Includes operator grades: `no_bpd`, `mild`
- **Class 1 (Positive)**: Moderate/Severe BPD
  - Includes operator grades: `moderate`, `severe`

### Probability Threshold

Default threshold: **0.5**

- `P(Moderate/Severe) >= 0.5` → Predicted as "Moderate/Severe"
- `P(Moderate/Severe) < 0.5` → Predicted as "No/Mild"

**Adjusting threshold**:
```python
# More conservative (fewer false positives)
threshold = 0.7
if prob >= threshold:
    prediction = "Moderate/Severe BPD"
else:
    prediction = "No/Mild BPD"
```

**Trade-off**:
- Higher threshold → Fewer false positives, more false negatives
- Lower threshold → Fewer false negatives, more false positives

### Confidence Levels

Rough interpretation:

- `p > 0.9`: Very confident
- `0.7 < p < 0.9`: Confident
- `0.5 < p < 0.7`: Moderate confidence
- `0.4 < p < 0.6`: Uncertain (close to decision boundary)

**Note**: Probabilities are NOT calibrated. Do not interpret `p = 0.8` as "80% chance of Moderate/Severe BPD" without calibration.

## Model Comparison

To compare predictions across models:

```python
from bpd_ui.core import ModelManager
from PIL import Image

mm = ModelManager("cpu")
img = Image.open("chest_xray.jpg")

# Get predictions from all models
models = [
    "bpd_xrv_progfreeze_lp_cutmix",
    "bpd_xrv_progfreeze",
    "bpd_xrv_fullft",
    "bpd_rgb_progfreeze",
]

for model_name in models:
    prob, _ = mm.predict(model_name, img)
    label = "Moderate/Severe" if prob >= 0.5 else "No/Mild"
    print(f"{model_name}: {label} (p={prob:.4f})")
```

**Expected output**:
```
bpd_xrv_progfreeze_lp_cutmix: Moderate/Severe (p=0.7823)
bpd_xrv_progfreeze: Moderate/Severe (p=0.7654)
bpd_xrv_fullft: Moderate/Severe (p=0.7432)
bpd_rgb_progfreeze: No/Mild (p=0.4876)
```

**Interpretation**:
- XRV models generally agree (similar probabilities)
- RGB model may disagree (different pretraining)
- Consensus prediction: Use majority vote or ensemble

## Troubleshooting

### Issue: Prediction seems incorrect

**Debugging steps**:
1. Verify image quality (resolution, contrast)
2. Check if image is chest X-ray (models are specialized)
3. Try different models (compare outputs)
4. Extract ROI manually to exclude artifacts
5. Check DICOM metadata (ensure correct pixel interpretation)

### Issue: Probabilities are always close to 0.5

**Possible causes**:
- Low-quality or corrupted image
- Wrong image type (not chest X-ray)
- Extreme brightness/contrast
- Model uncertainty (genuine ambiguous case)

**Solutions**:
- Preprocess image (adjust contrast, resize)
- Verify image format and metadata
- Compare with known ground truth examples

### Issue: DICOM image appears inverted

**Solution**:
- Check `PhotometricInterpretation` tag
- Should be automatically handled by `load_image_for_model()`
- If issue persists, manually invert:

```python
from bpd_ui.core import load_image_for_model
import numpy as np
from PIL import Image

img_array = np.array(load_image_for_model("chest_xray.dcm"))
img_inverted = Image.fromarray(255 - img_array)
```

### Issue: Model download fails

**Solutions**:
- Check internet connection
- Verify Google Drive link is accessible
- Manually download weights and place in `~/.cache/bpdneo/models/`
- See {doc}`../api/models` for download instructions

## Next Steps

- {doc}`dataset_evaluation` - Evaluate multiple images with metrics
- {doc}`preprocessing` - Build a preprocessed dataset
- {doc}`../api/models` - Model API reference
- {doc}`data_format` - Data format specification
