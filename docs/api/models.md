# Models Module

The `bpd_ui.models` module provides the model backend for BPDneo-CXR, including pretrained model loading, preprocessing transforms, and inference utilities.

## Overview

This module handles:

- Loading pretrained binary classification models
- Downloading model weights from Google Drive
- Providing preprocessing transforms matched to each model
- Managing model registry and configurations

## Key Functions

### Model Loading

```{eval-rst}
.. autofunction:: bpd_ui.models.load_pretrained_model
```

### Preprocessing

```{eval-rst}
.. autofunction:: bpd_ui.models.get_preprocessing_transforms
```

### Model Registry

```{eval-rst}
.. autofunction:: bpd_ui.models.list_available_models
```

### Weight Management

```{eval-rst}
.. autofunction:: bpd_ui.models.download_model_weights
```

## Classes

### BPDModel

```{eval-rst}
.. autoclass:: bpd_ui.models.BPDModel
   :members:
   :undoc-members:
   :show-inheritance:
```

### ModelDownloader

```{eval-rst}
.. autoclass:: bpd_ui.models.model_util.ModelDownloader
   :members:
   :undoc-members:
   :show-inheritance:
```

## Available Models

BPDneo-CXR includes four pretrained models developed for early BPD prediction from day-1 chest radiographs in extremely preterm infants {cite}`goedicke2025site`:

| Model ID | AUROC | Architecture | Preprocessing |
|----------|-------|--------------|---------------|
| `bpd_xrv_progfreeze_lp_cutmix` | 0.783 | ResNet50 + XRV | XRV normalization |
| `bpd_xrv_progfreeze` | 0.775 | ResNet50 + XRV | XRV normalization |
| `bpd_xrv_fullft` | 0.761 | ResNet50 + XRV | XRV normalization |
| `bpd_rgb_progfreeze` | 0.717 | ResNet50 + ImageNet | ImageNet normalization |

All models:
- Output binary classification (Moderate/Severe vs No/Mild BPD)
- Accept 512×512 images
- Run on CPU (no GPU required)
- Validated on multi-site data from extremely preterm infants (gestational age < 28 weeks)

## Usage Examples

### Basic Inference

```python
from bpd_ui.models import load_pretrained_model, get_preprocessing_transforms
from PIL import Image
import torch

# Load model and transforms
model = load_pretrained_model("bpd_xrv_progfreeze_lp_cutmix")
transform = get_preprocessing_transforms("bpd_xrv_progfreeze_lp_cutmix")

# Load and preprocess image
img = Image.open("chest_xray.jpg")
tensor = transform(img).unsqueeze(0)

# Run inference
model.eval()
with torch.no_grad():
    logits = model(tensor)
    prob = torch.sigmoid(logits).item()

# Interpret result
prediction = "Moderate/Severe BPD" if prob >= 0.5 else "No/Mild BPD"
print(f"Prediction: {prediction} (p={prob:.4f})")
```

### Listing Available Models

```python
from bpd_ui.models import list_available_models

models = list_available_models()
for name, info in models.items():
    print(f"{name}: AUROC {info['auroc']:.3f}")
```

### Pre-downloading Weights

```python
from bpd_ui.models import download_model_weights

# Download all model weights (useful for bundling)
download_model_weights()

# Download specific model
download_model_weights("bpd_xrv_progfreeze_lp_cutmix")
```

## Implementation Details

### Preprocessing Pipeline

**XRV Models** (xrv_progfreeze*, xrv_fullft):
1. Convert to grayscale
2. Normalize: `xrv.datasets.normalize(arr, maxval=255)`
3. Convert to tensor
4. Resize to 512×512

**ImageNet Models** (rgb_progfreeze):
1. Resize to 512×512
2. Convert to RGB
3. Convert to tensor
4. Normalize with ImageNet statistics

### Model Architecture

All models use ResNet50 backbone with:
- **XRV models**: Pretrained on CheXpert/MIMIC-CXR via TorchXRayVision
- **ImageNet models**: Pretrained on ImageNet-1k
- **Head**: Binary classification layer (1 output unit)
- **Loss**: BCEWithLogitsLoss during training
- **Inference**: Sigmoid activation for probability

### Progressive Freezing

The training strategy (used in `*_progfreeze` models) implements the progressive layer freezing approach described in {cite}`goedicke2025site`:
1. Freeze backbone, train head only (early epochs)
2. Unfreeze backbone progressively (later epochs)
3. Use different learning rates for backbone/head

This approach achieves superior generalization across multiple clinical sites while maintaining computational efficiency. See {meth}`bpd_ui.models.BPDModel.compile` for implementation details.
