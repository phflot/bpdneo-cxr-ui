"""
Model Backend for BPD Prediction
==================================

This module provides the core model infrastructure for BPD prediction from
chest X-rays, including:

- **Pretrained model loading**: Automated download and loading of pretrained
  model checkpoints from cloud storage
- **Unified model wrapper**: BPDModel class that handles both TorchXRayVision
  and torchvision ResNet backbones with a consistent interface
- **Preprocessing transforms**: Training-consistent image preprocessing pipelines
  for XRV (grayscale) and ImageNet (RGB) models
- **Model registry**: Configuration database with performance metrics (AUROC)
  for all available pretrained models

Available Models
----------------
- ``bpd_xrv_progfreeze_lp_cutmix`` : Best model (AUROC 0.783)
- ``bpd_xrv_progfreeze`` : Baseline without augmentation (AUROC 0.775)
- ``bpd_xrv_fullft`` : Full fine-tuning variant (AUROC 0.761)
- ``bpd_rgb_progfreeze`` : ImageNet baseline (AUROC 0.717)

Critical Usage Notes
--------------------
1. **Binary classification only**: All models output binary predictions
   (Moderate/Severe vs No/Mild), not 4-way classification
2. **Exact preprocessing required**: Always use ``get_preprocessing_transforms()``
   to ensure consistency with training
3. **CPU inference**: All models run on CPU by default for deployment
4. **Input size**: All models expect 512×512 images after preprocessing

Examples
--------
>>> from bpd_ui.models import load_pretrained_model, get_preprocessing_transforms
>>> from PIL import Image
>>> import torch
>>>
>>> # Load model and transforms
>>> model = load_pretrained_model("bpd_xrv_progfreeze_lp_cutmix")
>>> transform = get_preprocessing_transforms("bpd_xrv_progfreeze_lp_cutmix")
>>>
>>> # Preprocess image
>>> img = Image.open("xray.jpg")
>>> tensor = transform(img).unsqueeze(0)
>>>
>>> # Inference
>>> model.eval()
>>> with torch.no_grad():
...     logits = model(tensor)
...     prob = torch.sigmoid(logits).item()
>>>
>>> # Binary classification
>>> label = "Moderate/Severe BPD" if prob >= 0.5 else "No/Mild BPD"

Classes
-------
BPDModel
    Unified wrapper for XRV and torchvision ResNet models
ModelDownloader
    Handles downloading pretrained weights from cloud storage
ModelConfig
    Dataclass for model configuration (used in checkpoint serialization)

Functions
---------
load_pretrained_model
    Load a pretrained model by name (downloads if needed)
get_preprocessing_transforms
    Get training-consistent preprocessing transforms for a model
download_model_weights
    Download model weights to local cache
list_available_models
    List all available models with their metadata
init_weights
    Xavier uniform initialization for linear layers

See Also
--------
bpd_ui.core.model_manager : High-level model management for GUI
"""

from .model import BPDModel, init_weights
from .model_util import (
    ModelDownloader,
    download_model_weights,
    load_pretrained_model,
    get_preprocessing_transforms,
    list_available_models,
    MODEL_CONFIGS,
    ModelConfig,
)

__all__ = [
    "BPDModel",
    "init_weights",
    "ModelDownloader",
    "download_model_weights",
    "load_pretrained_model",
    "get_preprocessing_transforms",
    "list_available_models",
    "MODEL_CONFIGS",
    "ModelConfig",
]
