"""
Model Utilities for BPD Prediction
===================================

This module provides utilities for downloading, loading, and preprocessing
pretrained BPD prediction models. It handles:

- **Model registry**: Configuration database with AUROC metrics for all models
- **Download management**: Automatic download from cloud storage with progress bars
- **Checkpoint loading**: Robust deserialization with backwards compatibility
- **Preprocessing transforms**: Training-consistent image preprocessing pipelines
- **Model construction**: Automatic backbone selection and wrapper initialization

Available Models
----------------
The MODEL_CONFIGS dictionary contains metadata for all available models:

- bpd_xrv_progfreeze_lp_cutmix : Best model (AUROC 0.783)
- bpd_xrv_progfreeze : Baseline (AUROC 0.775)
- bpd_xrv_fullft : Full fine-tuning (AUROC 0.761)
- bpd_rgb_progfreeze : ImageNet baseline (AUROC 0.717)

Critical Implementation Details
--------------------------------
**Preprocessing consistency**: The get_preprocessing_transforms() function returns
transforms that exactly match those used during training. Never use custom transforms.

**XRV preprocessing**:
1. Convert to grayscale
2. Apply xrv.datasets.normalize(arr, 255)
3. Convert to tensor
4. Resize to 512×512

**ImageNet preprocessing**:
1. Resize to 512×512
2. Convert to RGB
3. ToTensor
4. ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

**Checkpoint compatibility**: The _load_checkpoint() function handles legacy
checkpoints that reference __main__.ModelConfig, ensuring compatibility with
older saved models.

Examples
--------
>>> from bpd_ui.models.model_util import load_pretrained_model, get_preprocessing_transforms
>>> from PIL import Image
>>> import torch
>>>
>>> # List available models
>>> from bpd_ui.models.model_util import list_available_models
>>> models = list_available_models()
>>> for name, info in models.items():
...     print(f"{name}: AUROC {info['auroc']}")
>>>
>>> # Load model and transforms
>>> model = load_pretrained_model("bpd_xrv_progfreeze_lp_cutmix")
>>> transform = get_preprocessing_transforms("bpd_xrv_progfreeze_lp_cutmix")
>>>
>>> # Preprocess and predict
>>> img = Image.open("xray.jpg")
>>> tensor = transform(img).unsqueeze(0)
>>> model.eval()
>>> with torch.no_grad():
...     logits = model(tensor)
...     prob = torch.sigmoid(logits).item()
>>>
>>> print(f"P(Moderate/Severe BPD) = {prob:.4f}")
"""

import torch
import os
import sys
import types
import requests
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Dict, Any
import json
import numpy as np
from dataclasses import dataclass


def _load_checkpoint(model_path, device):
    """
    Load model checkpoint with backwards compatibility for legacy references.

    This function handles checkpoints that may contain references to
    __main__.ModelConfig, which occurs when models were saved from scripts
    rather than modules. It attempts multiple loading strategies to ensure
    maximum compatibility.

    Parameters
    ----------
    model_path : str or Path
        Path to the checkpoint file (.pth)
    device : torch.device
        Device to map tensors to during loading

    Returns
    -------
    dict or torch.nn.Module
        Loaded checkpoint (usually a dict with 'model_state_dict' key)

    Notes
    -----
    Loading strategy (in order of attempt):
    1. Try weights_only=True (safest, PyTorch 2.x+)
    2. Try weights_only=True with __main__.ModelConfig stub registered
    3. Fall back to weights_only=False (less safe but compatible)

    The ModelConfig stub is a minimal class that allows unpickling without
    requiring the actual implementation, since we only need tensor data.

    Examples
    --------
    >>> checkpoint = _load_checkpoint("model.pth", torch.device("cpu"))
    >>> state_dict = checkpoint['model_state_dict']
    """
    # 1) Try safest path first
    try:
        return torch.load(model_path, map_location=device, weights_only=True)
    except Exception:
        pass

    # 2) Allow-list the legacy __main__.ModelConfig for safe weights-only load
    m = sys.modules.get("__main__")
    if m is None:
        m = types.ModuleType("__main__")
        sys.modules["__main__"] = m
    if not hasattr(m, "ModelConfig"):

        class ModelConfig:  # stub; structure not needed to read tensors
            pass

        m.ModelConfig = ModelConfig

    try:
        torch.serialization.add_safe_globals([m.ModelConfig])
        return torch.load(model_path, map_location=device, weights_only=True)
    except Exception:
        pass

    return torch.load(model_path, map_location=device, weights_only=False)


@dataclass
class ModelConfig:
    """Model configuration dataclass for checkpoint serialization."""

    name: str
    display_name: str
    backbone: str
    freezing: bool
    mixup: bool
    epochs: int
    probe_epochs: int
    probe_mixup: bool
    expected_auroc: float
    unfreeze_layer4: float = 0.10
    unfreeze_layer3: float = 0.30
    unfreeze_layer2: float = 0.60
    init_mixup_prob: float = 0.5
    mixup_alpha: float = 1.0
    probe_lr: float = 1e-3


MODEL_CONFIGS = {
    "bpd_xrv_progfreeze_lp_cutmix": {
        "file_id": "nLYMSE8jRSg3j8j",
        "description": "Best performing model with XRV pretraining, progressive freezing, linear probing, and CutMix",
        "auroc": 0.783,
        "preprocessing": "xrv",
        "input_size": 512,
        "num_classes": 1,
        "architecture": "resnet50",
        "backbone": "xrv",
        "frozen_layers": ["layer1", "layer2", "layer3"],
    },
    "bpd_xrv_progfreeze": {
        "file_id": "SRxGJzLSpEMMAD4",
        "description": "Baseline with XRV pretraining and progressive freezing (no augmentation)",
        "auroc": 0.775,
        "preprocessing": "xrv",
        "input_size": 512,
        "num_classes": 1,
        "architecture": "resnet50",
        "backbone": "xrv",
        "frozen_layers": ["layer1", "layer2", "layer3"],
    },
    "bpd_rgb_progfreeze": {
        "file_id": "W7EmnFDSFwoFSBL",
        "description": "ImageNet baseline with progressive freezing (for comparison)",
        "auroc": 0.717,
        "preprocessing": "imagenet",
        "input_size": 512,
        "num_classes": 1,
        "architecture": "resnet50",
        "backbone": "torchvision",
        "frozen_layers": ["layer1", "layer2", "layer3"],
    },
    "bpd_xrv_fullft": {
        "file_id": "w2czAo4oYxFaAGi",
        "description": "XRV pretraining with full fine-tuning (no freezing)",
        "auroc": 0.761,
        "preprocessing": "xrv",
        "input_size": 512,
        "num_classes": 1,
        "architecture": "resnet50",
        "backbone": "xrv",
        "frozen_layers": [],
    },
}


class ModelDownloader:
    """
    Downloads pre-trained BPD prediction models from cloud storage.

    This class handles automatic downloading of pretrained model weights from
    the HIZ Saarland cloud storage. It manages download progress, caching,
    and configuration file persistence.

    Parameters
    ----------
    model_name : str
        Name of the model to download (must be a key in MODEL_CONFIGS)
    save_dir : str, default="~/.bpdneo/models"
        Directory path where models will be saved (supports ~ expansion)

    Attributes
    ----------
    model_name : str
        Name of the requested model
    config : dict
        Model configuration from MODEL_CONFIGS
    save_dir : Path
        Expanded absolute path to save directory
    model_path : Path
        Full path to the model weights file (.pth)
    config_path : Path
        Full path to the configuration JSON file
    model_url : str
        Download URL constructed from config file_id

    Raises
    ------
    ValueError
        If model_name is not found in MODEL_CONFIGS

    Examples
    --------
    >>> from bpd_ui.models.model_util import ModelDownloader
    >>> downloader = ModelDownloader("bpd_xrv_progfreeze_lp_cutmix")
    >>> model_path = downloader.download_model()
    >>> print(f"Model downloaded to: {model_path}")

    See Also
    --------
    download_model_weights : Convenience function wrapper
    load_pretrained_model : Load model after download

    Notes
    -----
    The downloader automatically:
    - Creates the save directory if it doesn't exist
    - Saves a JSON config file alongside the weights
    - Skips download if the model already exists
    - Shows progress bar during download
    - Removes partial downloads on failure
    """

    def __init__(self, model_name: str, save_dir: str = "~/.bpdneo/models"):
        """
        Initialize the model downloader.

        Parameters
        ----------
        model_name : str
            Name of the model to download (see MODEL_CONFIGS keys)
        save_dir : str, default="~/.bpdneo/models"
            Directory to save downloaded models
        """
        if model_name not in MODEL_CONFIGS:
            available_models = ", ".join(MODEL_CONFIGS.keys())
            raise ValueError(
                f"Model '{model_name}' not found. Available models: {available_models}"
            )

        self.model_name = model_name
        self.config = MODEL_CONFIGS[model_name]
        self.save_dir = Path(os.path.expanduser(save_dir))
        self.model_path = self.save_dir / f"{model_name}.pth"
        self.config_path = self.save_dir / f"{model_name}_config.json"

        # Download URL
        file_id = self.config["file_id"]
        self.model_url = f"https://cloud.hiz-saarland.de/public.php/dav/files/{file_id}"

    def download_model(self) -> Path:
        """
        Download the model weights if not already present.

        This method downloads the model weights from cloud storage, displays
        download progress, and saves both the weights and configuration file.
        If the model already exists locally, it skips the download.

        Returns
        -------
        Path
            Path to the model weights file (.pth)

        Raises
        ------
        RuntimeError
            If the download fails due to network errors or server issues

        Notes
        -----
        Download process:
        1. Create save directory if needed
        2. Save configuration JSON
        3. Check if model already exists (skip if yes)
        4. Stream download with progress bar
        5. Remove partial download on failure

        The download uses streaming to handle large files efficiently and
        displays a tqdm progress bar showing download speed and ETA.

        Examples
        --------
        >>> downloader = ModelDownloader("bpd_xrv_progfreeze_lp_cutmix")
        >>> path = downloader.download_model()
        Model 'bpd_xrv_progfreeze_lp_cutmix' already exists at ...
        >>> print(path)
        """
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Save config file
        if not self.config_path.exists():
            with open(self.config_path, "w") as f:
                json.dump(self.config, f, indent=2)

        # Check if model already exists
        if self.model_path.exists():
            print(f"Model '{self.model_name}' already exists at {self.model_path}")
            return self.model_path

        print(f"Downloading {self.model_name}...")
        print(f"Description: {self.config['description']}")
        print(f"Expected AUROC: {self.config['auroc']}")

        try:
            response = requests.get(self.model_url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))

            with open(self.model_path, "wb") as f:
                with tqdm(
                    desc=f"Downloading {self.model_name}",
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            print(f"Model downloaded successfully to {self.model_path}")
            return self.model_path

        except requests.exceptions.RequestException as e:
            if self.model_path.exists():
                self.model_path.unlink()  # Remove partial download
            raise RuntimeError(f"Failed to download model: {e}")


def download_model_weights(model_name: str, save_dir: str = "~/.bpdneo/models") -> Path:
    """
    Download pre-trained model weights (convenience wrapper).

    This is a convenience function that creates a ModelDownloader instance
    and downloads the specified model. It's the recommended way to download
    models if you don't need to access downloader internals.

    Parameters
    ----------
    model_name : str
        Name of the model to download (must be in MODEL_CONFIGS)
    save_dir : str, default="~/.bpdneo/models"
        Directory to save the model (supports ~ expansion)

    Returns
    -------
    Path
        Path to the downloaded model weights file

    Examples
    --------
    >>> from bpd_ui.models.model_util import download_model_weights
    >>> path = download_model_weights("bpd_xrv_progfreeze_lp_cutmix")
    >>> print(f"Model saved to: {path}")

    See Also
    --------
    ModelDownloader : Full downloader class with more control
    load_pretrained_model : Download and load model in one call
    """
    downloader = ModelDownloader(model_name, save_dir)
    return downloader.download_model()


def _build_model_from_config(config: Dict[str, Any]) -> torch.nn.Module:
    """
    Build model architecture from configuration, matching training setup.

    This internal function constructs a BPDModel with the appropriate backbone
    (XRV or torchvision ResNet50) based on the model configuration. It ensures
    the architecture exactly matches the one used during training.

    Parameters
    ----------
    config : dict
        Model configuration dictionary containing at least:
        - 'backbone' : str, either 'xrv' or 'torchvision'

    Returns
    -------
    torch.nn.Module
        BPDModel instance with appropriate backbone

    Raises
    ------
    ValueError
        If config['backbone'] is not 'xrv' or 'torchvision'

    Notes
    -----
    Backbone details:
    - 'xrv': TorchXRayVision ResNet50 pretrained on ChestX-ray14 (512x512)
    - 'torchvision': Standard ImageNet ResNet50 (adapted for grayscale)

    The BPDModel wrapper replaces the final fully-connected layer with a
    binary classification head (single output unit).

    Examples
    --------
    >>> config = MODEL_CONFIGS['bpd_xrv_progfreeze_lp_cutmix']
    >>> model = _build_model_from_config(config)
    >>> print(type(model).__name__)
    BPDModel

    See Also
    --------
    BPDModel : Model wrapper class
    load_pretrained_model : Public API for loading models
    """
    import torchxrayvision as xrv
    from torchvision import models as tvm
    from bpd_ui.models.model import BPDModel

    if config["backbone"] == "xrv":
        base = xrv.models.ResNet(weights="resnet50-res512-all")
    elif config["backbone"] == "torchvision":
        base = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
    else:
        raise ValueError(f"Unknown backbone: {config['backbone']}")

    return BPDModel(base)


def load_pretrained_model(
    model_name: str,
    device: Optional[torch.device] = None,
    save_dir: str = "~/.bpdneo/models",
) -> torch.nn.Module:
    """
    Load a pre-trained BPD prediction model ready for inference.

    This is the main entry point for loading pretrained models. It handles
    downloading (if needed), architecture construction, checkpoint loading,
    and device placement. The returned model is in eval mode and ready for
    immediate inference.

    Parameters
    ----------
    model_name : str
        Name of the model to load (must be a key in MODEL_CONFIGS)
    device : torch.device, optional
        Device to load the model on. If None, uses CUDA if available,
        otherwise CPU.
    save_dir : str, default="~/.bpdneo/models"
        Directory where models are cached (supports ~ expansion)

    Returns
    -------
    torch.nn.Module
        Loaded BPDModel in eval mode, moved to specified device

    Raises
    ------
    ValueError
        If model_name is not in MODEL_CONFIGS
    RuntimeError
        If checkpoint loading fails

    Notes
    -----
    This function performs the following steps:
    1. Auto-detect device if not specified
    2. Download model weights if not cached
    3. Build model architecture from config
    4. Load checkpoint with backwards compatibility handling
    5. Extract state dict robustly (handles multiple formats)
    6. Load state dict with strict=True
    7. Set model to eval mode
    8. Print confirmation with preprocessing info

    The checkpoint loading uses _load_checkpoint() which handles legacy
    checkpoints that reference __main__.ModelConfig.

    Examples
    --------
    >>> from bpd_ui.models.model_util import load_pretrained_model
    >>> import torch
    >>>
    >>> # Load best model on CPU
    >>> model = load_pretrained_model("bpd_xrv_progfreeze_lp_cutmix")
    Loaded model 'bpd_xrv_progfreeze_lp_cutmix' on cpu
    Preprocessing: xrv
    Input size: 512x512
    >>>
    >>> # Load on GPU if available
    >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    >>> model = load_pretrained_model("bpd_xrv_fullft", device=device)
    >>>
    >>> # Inference
    >>> model.eval()
    >>> with torch.no_grad():
    ...     logits = model(input_tensor)
    ...     prob = torch.sigmoid(logits)

    See Also
    --------
    download_model_weights : Download weights without loading
    get_preprocessing_transforms : Get matching preprocessing
    list_available_models : List all available models
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Download model if needed
    model_path = download_model_weights(model_name, save_dir)

    # Load configuration
    config = MODEL_CONFIGS[model_name]

    # Build model with correct backbone and wrapper
    model = _build_model_from_config(config).to(device)

    # Load checkpoint using our helper function
    checkpoint = _load_checkpoint(model_path, device)

    # Extract state dict robustly
    if isinstance(checkpoint, dict):
        state = (
            checkpoint.get("model_state_dict")
            or checkpoint.get("state_dict")
            or checkpoint
        )
    else:
        state = checkpoint

    model.load_state_dict(state, strict=True)

    model.eval()

    print(f"Loaded model '{model_name}' on {device}")
    print(f"Preprocessing: {config['preprocessing']}")
    print(f"Input size: {config['input_size']}x{config['input_size']}")

    return model


def get_preprocessing_transforms(model_name: str):
    """
    Get the appropriate preprocessing transforms for a model.

    This function returns the exact preprocessing pipeline used during training
    for the specified model. It is CRITICAL to use these transforms rather than
    custom preprocessing to ensure consistent results.

    Parameters
    ----------
    model_name : str
        Name of the model (must be a key in MODEL_CONFIGS)

    Returns
    -------
    torchvision.transforms.Compose or callable
        Transform function/composition that converts PIL Image to tensor.
        For XRV models: grayscale → XRV normalize → tensor → resize(512)
        For ImageNet models: resize(512) → RGB → tensor → ImageNet normalize

    Raises
    ------
    ValueError
        If config['preprocessing'] is not 'xrv' or 'imagenet'
    KeyError
        If model_name is not in MODEL_CONFIGS

    Notes
    -----
    **XRV Preprocessing Pipeline**:
    1. Convert PIL Image to grayscale numpy array
    2. Apply xrv.datasets.normalize(arr, 255) - scales to [0, 1] range
    3. Convert to tensor (adds channel dimension)
    4. Resize to 512×512 with antialiasing

    **ImageNet Preprocessing Pipeline**:
    1. Resize to 512×512 with antialiasing
    2. Convert to RGB (if not already)
    3. Convert to tensor (scales to [0, 1])
    4. Normalize with ImageNet statistics:
       - mean=[0.485, 0.456, 0.406]
       - std=[0.229, 0.224, 0.225]

    **CRITICAL**: Never create custom preprocessing. Always use this function
    to ensure consistency with training. Even minor deviations (e.g., different
    resize interpolation) can significantly impact model performance.

    Examples
    --------
    >>> from bpd_ui.models.model_util import get_preprocessing_transforms
    >>> from PIL import Image
    >>> import torch
    >>>
    >>> # Get transforms for best model
    >>> transform = get_preprocessing_transforms("bpd_xrv_progfreeze_lp_cutmix")
    >>>
    >>> # Apply to image
    >>> img = Image.open("xray.jpg")
    >>> tensor = transform(img)
    >>> print(tensor.shape)
    torch.Size([1, 512, 512])
    >>>
    >>> # Batch inference
    >>> batch = torch.stack([transform(img) for img in images])
    >>> logits = model(batch)

    See Also
    --------
    load_pretrained_model : Load model (use with these transforms)
    MODEL_CONFIGS : Model configuration database
    """
    from torchvision import transforms as T
    import torchxrayvision as xrv

    config = MODEL_CONFIGS[model_name]
    input_size = config["input_size"]

    if config["preprocessing"] == "xrv":
        # XRV preprocessing as used in training:
        # Load grayscale -> xrv.datasets.normalize -> tensor -> resize(512)
        def _xrv_transform(pil_img):
            """Apply XRV preprocessing matching training."""
            # Convert to grayscale numpy array
            arr = np.array(pil_img.convert("L"))
            # Apply XRV normalization (expects values 0-255)
            arr = xrv.datasets.normalize(arr, 255)
            # Convert to tensor (this handles the conversion to float and adds channel dim)
            tensor = T.ToTensor()(arr)
            # Resize to target size
            resize = T.Resize((input_size, input_size), antialias=True)
            return resize(tensor)

        transform = T.Lambda(_xrv_transform)

    elif config["preprocessing"] == "imagenet":
        # ImageNet preprocessing for RGB models
        transform = T.Compose(
            [
                T.Resize((input_size, input_size), antialias=True),
                T.Lambda(lambda x: x.convert("RGB") if x.mode != "RGB" else x),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        raise ValueError(f"Unknown preprocessing type: {config['preprocessing']}")

    return transform


def list_available_models() -> Dict[str, Any]:
    """
    List all available pre-trained models with their metadata.

    This function returns a simplified view of MODEL_CONFIGS containing only
    the most relevant information for model selection: description, AUROC,
    architecture, and preprocessing type.

    Returns
    -------
    dict
        Dictionary mapping model names to metadata dictionaries.
        Each metadata dict contains:
        - 'description' : str, model description
        - 'auroc' : float, test set AUROC performance
        - 'architecture' : str, backbone architecture (always resnet50)
        - 'preprocessing' : str, either 'xrv' or 'imagenet'

    Examples
    --------
    >>> from bpd_ui.models.model_util import list_available_models
    >>>
    >>> models = list_available_models()
    >>> for name, info in models.items():
    ...     print(f"{name}: AUROC {info['auroc']:.3f}")
    bpd_xrv_progfreeze_lp_cutmix: AUROC 0.783
    bpd_xrv_progfreeze: AUROC 0.775
    bpd_xrv_fullft: AUROC 0.761
    bpd_rgb_progfreeze: AUROC 0.717
    >>>
    >>> # Find best XRV model
    >>> xrv_models = {k: v for k, v in models.items()
    ...               if v['preprocessing'] == 'xrv'}
    >>> best = max(xrv_models.items(), key=lambda x: x[1]['auroc'])
    >>> print(f"Best XRV model: {best[0]}")

    See Also
    --------
    MODEL_CONFIGS : Full model configuration database
    load_pretrained_model : Load a specific model
    """
    models_info = {}
    for name, config in MODEL_CONFIGS.items():
        models_info[name] = {
            "description": config["description"],
            "auroc": config["auroc"],
            "architecture": config["architecture"],
            "preprocessing": config["preprocessing"],
        }
    return models_info


if __name__ == "__main__":
    # Example usage
    print("Available BPD prediction models:")
    print("-" * 50)

    for name, info in list_available_models().items():
        print(f"\n{name}:")
        print(f"  Description: {info['description']}")
        print(f"  AUROC: {info['auroc']}")
        print(f"  Architecture: {info['architecture']}")
        print(f"  Preprocessing: {info['preprocessing']}")

    print("\n" + "-" * 50)
    print("\nTo download and use a model:")
    print(">>> from bpd_ui.models.model_util import load_pretrained_model")
    print(">>> model = load_pretrained_model('bpd_xrv_progfreeze_lp_cutmix')")
