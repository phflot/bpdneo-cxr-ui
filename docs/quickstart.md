# Quick Start Guide

Get up and running with BPDneo-CXR in minutes.

## GUI Application

### Launch the GUI

```bash
python apps/gui_app.py
```

The main window provides three tabs:

1. **Preprocessing** - Prepare images and extract ROIs
2. **Dataset Evaluation** - Batch inference on prepared datasets
3. **Single Image Evaluation** - Predict on individual images

### Single Image Prediction (Fastest Start)

1. Click the **"Single Image Evaluation"** tab
2. Click **"Select Image..."** and choose a chest X-ray (PNG, JPEG, or DICOM)
3. Select a model from the dropdown (default: `bpd_xrv_progfreeze_lp_cutmix`)
4. Click **"Run Inference"**

The application will:
- Download the model weights automatically (first time only)
- Preprocess the image
- Display prediction probabilities

**Example output**:
```
Binary Label: Moderate/Severe BPD
P(Moderate/Severe BPD): 0.7823
P(No/Mild BPD): 0.2177
```

## Programmatic Usage

### Load a Pretrained Model

```python
from bpd_ui.models import load_pretrained_model, get_preprocessing_transforms
from PIL import Image
import torch

# Load the best-performing model
model = load_pretrained_model("bpd_xrv_progfreeze_lp_cutmix")

# Get preprocessing transforms
transform = get_preprocessing_transforms("bpd_xrv_progfreeze_lp_cutmix")
```

### Run Inference on an Image

```python
# Load and preprocess image
img = Image.open("chest_xray.jpg")
tensor = transform(img).unsqueeze(0)  # Add batch dimension

# Predict
model.eval()
with torch.no_grad():
    logits = model(tensor)
    prob = torch.sigmoid(logits).item()

# Interpret result
if prob >= 0.5:
    prediction = "Moderate/Severe BPD"
else:
    prediction = "No/Mild BPD"

print(f"Prediction: {prediction}")
print(f"Probability: {prob:.4f}")
```

### Using ModelManager (Recommended)

The `ModelManager` provides a simpler interface with automatic caching:

```python
from bpd_ui.core import ModelManager
from PIL import Image

# Create manager
mm = ModelManager("cpu")

# Load and predict
img = Image.open("chest_xray.jpg")
prob, logit = mm.predict("bpd_xrv_progfreeze_lp_cutmix", img)

print(f"P(Moderate/Severe BPD): {prob:.4f}")
```

## Available Models

BPDneo-CXR includes four pretrained models developed using site-level fine-tuning with progressive layer freezing {cite}`goedicke2025site`:

| Model | AUROC | Description |
|-------|-------|-------------|
| `bpd_xrv_progfreeze_lp_cutmix` | 0.783 | **Best model** - XRV pretraining + progressive freezing + CutMix |
| `bpd_xrv_progfreeze` | 0.775 | Baseline without augmentation |
| `bpd_xrv_fullft` | 0.761 | Full fine-tuning variant |
| `bpd_rgb_progfreeze` | 0.717 | ImageNet baseline |

All models:
- Accept 512×512 images after preprocessing
- Output binary predictions (Moderate/Severe vs No/Mild)
- Run on CPU (no GPU required)
- Trained on day-1 chest radiographs from extremely preterm infants

## Next Steps

- {doc}`user_guide/preprocessing` - Prepare your own dataset
- {doc}`user_guide/dataset_evaluation` - Evaluate on multiple images
- {doc}`api/models` - Detailed model API documentation
