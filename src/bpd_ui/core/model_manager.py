"""
Model Manager
=============

This module provides a high-level model management class that handles model
loading, caching, and inference with automatic preprocessing. It maintains
a cache of loaded models to avoid repeated downloads and initialization.

Classes
-------
ModelManager
    Manages model instances and provides inference API

Notes
-----
The ModelManager caches both models and their preprocessing transforms to
minimize overhead when running multiple inferences with the same model.

See Also
--------
bpd_ui.models.model_util : Low-level model loading functions
bpd_ui.ui.single_eval_tab : Uses ModelManager for inference
bpd_ui.ui.dataset_eval_tab : Uses ModelManager for batch evaluation
"""

import torch
from bpd_ui.models.model_util import load_pretrained_model, get_preprocessing_transforms


class ModelManager:
    """
    High-level model manager with caching and inference API.

    This class provides a simplified interface for model loading and inference.
    It automatically manages model caching, preprocessing transforms, and
    device placement. Models are loaded on first use and cached for subsequent
    inferences.

    Parameters
    ----------
    device : str, default="cpu"
        Device to run models on ("cpu" or "cuda")

    Attributes
    ----------
    device : torch.device
        PyTorch device object
    _models : dict
        Cache of loaded models (model_name -> nn.Module)
    _transforms : dict
        Cache of preprocessing transforms (model_name -> callable)

    Examples
    --------
    >>> from bpd_ui.core.model_manager import ModelManager
    >>> from PIL import Image
    >>>
    >>> # Create manager
    >>> mm = ModelManager("cpu")
    >>>
    >>> # Load and run inference
    >>> img = Image.open("xray.jpg")
    >>> prob, logit = mm.predict("bpd_xrv_progfreeze_lp_cutmix", img)
    >>> print(f"P(Moderate/Severe BPD) = {prob:.4f}")
    >>>
    >>> # Subsequent predictions use cached model
    >>> img2 = Image.open("xray2.jpg")
    >>> prob2, logit2 = mm.predict("bpd_xrv_progfreeze_lp_cutmix", img2)

    See Also
    --------
    bpd_ui.models.model_util.load_pretrained_model : Model loading
    bpd_ui.models.model_util.get_preprocessing_transforms : Preprocessing
    """

    def __init__(self, device: str = "cpu"):
        """
        Initialize the model manager.

        Parameters
        ----------
        device : str, default="cpu"
            Device to run models on ("cpu" or "cuda")
        """
        self.device = torch.device(device)
        self._models: dict[str, torch.nn.Module] = {}
        self._transforms: dict[str, object] = {}

    def get(self, model_name: str):
        """
        Get model and transform, loading if not cached.

        This method retrieves a model and its preprocessing transform from
        the cache, loading them on first access. Both are cached for
        subsequent calls.

        Parameters
        ----------
        model_name : str
            Name of the model (must be in MODEL_CONFIGS)

        Returns
        -------
        model : torch.nn.Module
            Loaded model in eval mode on specified device
        transform : callable
            Preprocessing transform for the model

        Notes
        -----
        On first call with a given model_name:
        1. Downloads weights if needed
        2. Loads model architecture and weights
        3. Moves model to specified device
        4. Sets model to eval mode
        5. Loads preprocessing transform
        6. Caches both for future use

        Subsequent calls return cached instances immediately.

        Examples
        --------
        >>> mm = ModelManager("cpu")
        >>> model, transform = mm.get("bpd_xrv_progfreeze_lp_cutmix")
        >>> print(type(model).__name__)
        BPDModel
        >>>
        >>> # Apply transform
        >>> from PIL import Image
        >>> img = Image.open("xray.jpg")
        >>> tensor = transform(img)
        >>> print(tensor.shape)
        torch.Size([1, 512, 512])

        See Also
        --------
        predict : High-level inference method
        """
        if model_name not in self._models:
            m = load_pretrained_model(model_name, device=self.device)
            self._models[model_name] = m
        if model_name not in self._transforms:
            self._transforms[model_name] = get_preprocessing_transforms(model_name)
        return self._models[model_name], self._transforms[model_name]

    @torch.inference_mode()
    def predict(self, model_name: str, image):
        """
        Run inference on an image with automatic preprocessing.

        This is the main inference method. It handles preprocessing, batching,
        model execution, and post-processing in a single call.

        Parameters
        ----------
        model_name : str
            Name of the model to use (must be in MODEL_CONFIGS)
        image : PIL.Image
            Input image (any mode, will be preprocessed appropriately)

        Returns
        -------
        prob : float
            Predicted probability of Moderate/Severe BPD (0-1 range)
        logit : float
            Raw model output logit (unbounded)

        Notes
        -----
        Inference pipeline:
        1. Get cached model and transform (or load if first use)
        2. Apply preprocessing transform
        3. Add batch dimension
        4. Move to device
        5. Forward pass with inference_mode
        6. Apply sigmoid to get probability
        7. Return both probability and raw logit

        The @torch.inference_mode() decorator disables gradient computation
        and reduces memory usage compared to torch.no_grad().

        Examples
        --------
        >>> from bpd_ui.core.model_manager import ModelManager
        >>> from PIL import Image
        >>>
        >>> mm = ModelManager("cpu")
        >>> img = Image.open("xray.jpg")
        >>> prob, logit = mm.predict("bpd_xrv_progfreeze_lp_cutmix", img)
        >>>
        >>> # Interpret results
        >>> if prob >= 0.5:
        ...     label = "Moderate/Severe BPD"
        ... else:
        ...     label = "No/Mild BPD"
        >>> print(f"Prediction: {label} (p={prob:.4f})")
        >>>
        >>> # Batch processing
        >>> images = [Image.open(f"xray_{i}.jpg") for i in range(10)]
        >>> for img in images:
        ...     prob, logit = mm.predict("bpd_xrv_progfreeze_lp_cutmix", img)
        ...     print(f"P(Moderate/Severe) = {prob:.4f}")

        See Also
        --------
        get : Get model and transform
        """
        model, transform = self.get(model_name)
        x = transform(image).unsqueeze(0).to(self.device)
        logits = model(x)
        prob = torch.sigmoid(logits).flatten()[0].item()
        return prob, float(logits.flatten()[0].item())
