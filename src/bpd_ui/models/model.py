# model.py  – revised
# unified model wrapper with configurable compile()

from __future__ import annotations

import torch
from torch import nn, optim
from torch.nn import init
from typing import List


class BPDModel(nn.Module):
    """
    Unified wrapper for BPD prediction models with binary classification head.

    This class wraps either a TorchXRayVision ResNet or a torchvision ResNet
    and provides a unified interface for training and inference. It automatically
    detects the backend type and exposes the underlying torchvision model for
    uniform layer access.

    The wrapper replaces the original fully-connected layer with a binary
    classification head (single output unit) and provides configurable optimizer
    setup with support for progressive unfreezing.

    Parameters
    ----------
    base_model : nn.Module
        Base ResNet model, either from torchxrayvision.models or torchvision.models.
        The final classification layer will be replaced.

    Attributes
    ----------
    base_model : nn.Module
        The original model passed during initialization
    inner_model : nn.Module
        The actual torchvision ResNet (unwrapped from XRV if needed)
    is_xrv : bool
        True if base_model is from TorchXRayVision, False for torchvision
    dropout : nn.Dropout
        Dropout layer (p=0.1) applied before final classification
    fc : nn.Linear
        Binary classification head (features → 1 logit)

    Examples
    --------
    >>> import torchxrayvision as xrv
    >>> from bpd_ui.models import BPDModel
    >>>
    >>> # Create model with XRV backbone
    >>> xrv_base = xrv.models.ResNet(weights="resnet50-res512-all")
    >>> model = BPDModel(xrv_base)
    >>>
    >>> # Configure optimizer with frozen backbone
    >>> model.compile(freeze_backbone=True, lr_head=5e-4)
    >>>
    >>> # Inference
    >>> import torch
    >>> x = torch.randn(1, 1, 512, 512)
    >>> logits = model(x)
    >>> prob = torch.sigmoid(logits)

    See Also
    --------
    bpd_ui.models.model_util.load_pretrained_model : Load pretrained weights
    """

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model

        # ── pick the true torchvision model regardless of wrapper ─────────
        if hasattr(self.base_model, "model"):
            self.inner_model = self.base_model.model  # XRV wrapper
            self.is_xrv = True
        else:
            self.inner_model = self.base_model  # torchvision model
            self.is_xrv = False

        # ── replace FC with identity – we'll add our own head ─────────────
        in_features = self.inner_model.fc.in_features
        self.inner_model.fc = nn.Identity()
        self.dropout = nn.Dropout(p=0.1)
        self.fc = nn.Linear(in_features, 1)
        self.fc.apply(init_weights)

        self._optimizer: optim.Optimizer | None = None

    # ------------------------------------------------------------------
    # forward / utils
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self.is_xrv:
            feats = self.base_model.features(x)
        else:
            feats = self.inner_model(x)
        feats = self.dropout(feats)
        return self.fc(feats)

    def get_resolution(self) -> int:
        """
        Get the default training resolution for this model.

        Returns
        -------
        int
            Training resolution in pixels (always 512 for BPD models)

        Notes
        -----
        All BPD models were trained with 512×512 input images. This method
        is used by data augmentation pipelines to ensure consistent preprocessing.
        """
        return 512

    # ------------------------------------------------------------------
    # Optimiser / freezing logic – CONFIGURABLE
    # ------------------------------------------------------------------
    def compile(
        self,
        *,
        freeze_backbone: bool = True,
        lr_backbone: float = 1e-4,
        lr_head: float = 5e-4,
        wd_backbone: float = 1e-4,
        wd_head: float = 1e-5,
    ) -> None:
        """
        Build an AdamW optimizer with configurable backbone freezing.

        This method creates an optimizer with separate parameter groups for the
        classification head and backbone layers. When freeze_backbone is True,
        backbone parameters are frozen but remain in the optimizer, enabling
        later unfreezing without recreating the optimizer or learning rate scheduler.

        Parameters
        ----------
        freeze_backbone : bool, default=True
            If True, freeze all backbone parameters (only head learns).
            If False, full fine-tuning of all layers.
        lr_backbone : float, default=1e-4
            Learning rate for backbone layers (layer3, layer4)
        lr_head : float, default=5e-4
            Learning rate for the classification head
        wd_backbone : float, default=1e-4
            Weight decay for backbone parameters
        wd_head : float, default=1e-5
            Weight decay for head parameters

        Notes
        -----
        When freeze_backbone=True, this method:
        1. Sets requires_grad=False for all backbone parameters
        2. Creates parameter groups for layer3, layer4, and head
        3. Frozen layers remain in optimizer for later unfreezing

        Progressive unfreezing workflow:
        1. Call compile(freeze_backbone=True)
        2. Train for N epochs
        3. Call model.unfreeze_layers(model.inner_model.layer4)
        4. Continue training (optimizer already has layer4 params)

        Examples
        --------
        >>> model = BPDModel(base_model)
        >>>
        >>> # Progressive freezing training
        >>> model.compile(freeze_backbone=True, lr_head=5e-4)
        >>> optimizer = model.get_optimizer()
        >>>
        >>> # After linear probing, unfreeze layer4
        >>> model.unfreeze_layers(model.inner_model.layer4)
        >>>
        >>> # Continue training with same optimizer
        >>> # (layer4 params already present with lr_backbone rate)

        See Also
        --------
        unfreeze_layers : Unfreeze specific layers during training
        get_optimizer : Retrieve the configured optimizer
        """
        # ── freeze / unfreeze tensors ------------------------------------
        for p in self.inner_model.parameters():
            p.requires_grad = not freeze_backbone

        # ── build parameter groups ---------------------------------------
        param_groups: List[dict] = [
            {"params": self.fc.parameters(), "lr": lr_head, "weight_decay": wd_head},
        ]

        if freeze_backbone:
            # keep layer3/4 frozen for now but add them so scheduler knows them
            for blk in (self.inner_model.layer3, self.inner_model.layer4):
                for p in blk.parameters():
                    p.requires_grad = False
            param_groups += [
                {
                    "params": self.inner_model.layer3.parameters(),
                    "lr": lr_backbone,
                    "weight_decay": wd_backbone,
                },
                {
                    "params": self.inner_model.layer4.parameters(),
                    "lr": lr_backbone,
                    "weight_decay": wd_backbone,
                },
            ]
        else:
            # full‑finetune: one group for the entire backbone
            param_groups.append(
                {
                    "params": self.inner_model.parameters(),
                    "lr": lr_backbone,
                    "weight_decay": wd_backbone,
                }
            )

        self._optimizer = optim.AdamW(param_groups)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def load_model(self, filepath: str) -> None:
        """
        Load model weights and optimizer state from a checkpoint file.

        Parameters
        ----------
        filepath : str
            Path to the checkpoint file (.pth)

        Notes
        -----
        The checkpoint file is expected to contain a dictionary with:
        - 'model_state_dict': Model parameters
        - 'optimizer_state_dict' (optional): Optimizer state

        If an optimizer has been configured via compile() and the checkpoint
        contains optimizer state, it will be restored.

        Examples
        --------
        >>> model = BPDModel(base_model)
        >>> model.compile(freeze_backbone=True)
        >>> model.load_model("checkpoint_epoch_10.pth")
        """
        ckpt = torch.load(filepath, map_location=torch.device("cpu"))
        self.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt and self._optimizer is not None:
            self._optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    def get_optimizer(self) -> optim.Optimizer:
        """
        Retrieve the configured optimizer.

        Returns
        -------
        optim.Optimizer
            The AdamW optimizer created by compile()

        Raises
        ------
        RuntimeError
            If compile() has not been called yet

        Examples
        --------
        >>> model = BPDModel(base_model)
        >>> model.compile(freeze_backbone=True)
        >>> optimizer = model.get_optimizer()
        >>> for epoch in range(10):
        ...     optimizer.zero_grad()
        ...     # training loop
        """
        if self._optimizer is None:
            raise RuntimeError("compile() must be called before get_optimizer()")
        return self._optimizer

    # Optionally expose an easy unfreeze utility -------------------------
    def unfreeze_layers(self, *layers: nn.Sequential) -> None:
        """
        Unfreeze specified layers by setting requires_grad=True.

        This method enables progressive unfreezing during training. Layers must
        already be present in the optimizer (via compile() parameter groups) for
        gradients to be applied.

        Parameters
        ----------
        *layers : nn.Sequential
            One or more layer modules to unfreeze (e.g., model.inner_model.layer4)

        Examples
        --------
        >>> model = BPDModel(base_model)
        >>> model.compile(freeze_backbone=True)
        >>>
        >>> # Train with frozen backbone for 5 epochs
        >>> # ...
        >>>
        >>> # Unfreeze layer4 for fine-tuning
        >>> model.unfreeze_layers(model.inner_model.layer4)
        >>>
        >>> # Continue training (layer4 grads now active)
        >>> # ...
        >>>
        >>> # Optionally unfreeze layer3 later
        >>> model.unfreeze_layers(model.inner_model.layer3)

        See Also
        --------
        compile : Configure optimizer with parameter groups
        """
        for layer in layers:
            for p in layer.parameters():
                p.requires_grad = True


# ----------------------------------------------------------------------
# utils
# ----------------------------------------------------------------------


def init_weights(m: nn.Module) -> None:
    """
    Initialize weights for linear layers using Xavier uniform initialization.

    This function is applied to the binary classification head (fc layer) during
    BPDModel initialization to ensure proper weight initialization.

    Parameters
    ----------
    m : nn.Module
        Module to initialize (only linear layers are affected)

    Notes
    -----
    - Linear layer weights: Xavier uniform initialization
    - Linear layer biases: Constant fill with 0.01
    - Other module types: No initialization applied

    Xavier (Glorot) uniform initialization draws weights from a uniform
    distribution with bounds calculated to maintain variance across layers.

    Examples
    --------
    >>> import torch.nn as nn
    >>> fc = nn.Linear(2048, 1)
    >>> init_weights(fc)
    """
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
