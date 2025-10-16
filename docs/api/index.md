# API Reference

Complete API documentation for BPDneo-CXR.

## Organization

The BPDneo-CXR API is organized into three main modules:

- {doc}`models` - Model loading, inference, and preprocessing
- {doc}`core` - Application logic and data management
- {doc}`ui` - GUI components (PyQt6)

## Quick Reference

### Loading a Pretrained Model

```python
from bpd_ui.models import load_pretrained_model

model = load_pretrained_model("bpd_xrv_progfreeze_lp_cutmix")
```

### Running Inference

```python
from bpd_ui.core import ModelManager
from PIL import Image

mm = ModelManager("cpu")
img = Image.open("chest_xray.jpg")
prob, logit = mm.predict("bpd_xrv_progfreeze_lp_cutmix", img)
```

### Working with Datasets

```python
from bpd_ui.core import read_manifest, upsert_manifest

# Read existing manifest
df = read_manifest("prepared/manifest.xlsx")

# Update manifest with new entry
upsert_manifest(
    manifest_path="prepared/manifest.xlsx",
    patient_id="PT001",
    image_relpath="PT001/images/img_001.png",
    roi_relpath="PT001/masks/roi_001.png",
    grade_label="moderate"
)
```

## Module Documentation

```{toctree}
:maxdepth: 2

models
core
ui
```

## Indices

- {ref}`genindex`
- {ref}`modindex`
