"""
3D Convolutional Neural Network architectures for vertebral CT bone segmentation.
"""

from typing import Tuple, Sequence, Optional
import torch
import torch.nn as nn
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from .config import NUM_CLASSES


def build_orthosegnet_model(
    in_channels: int = 1,
    out_channels: int = NUM_CLASSES,
    channels: Sequence[int] = (16, 32, 64, 128, 256),
    strides: Sequence[int] = (2, 2, 2, 2),
    num_res_units: int = 2,
    norm: str = "batch",
    dropout: float = 0.0,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Constructs a 5-level 3D Residual U-Net for multi-class bone segmentation.

    Architectural properties:
    - 3D spatial convolutions capture inter-slice volumetric connectivity across Z-axis.
    - Residual units in each encoder/decoder block prevent gradient degradation.
    - Strided convolutions for learned downsampling instead of fixed max-pooling.
    - Skip connections concatenate high-resolution fine-boundary features with deep semantic context.
    """
    norm_type = Norm.BATCH if norm.lower() == "batch" else Norm.INSTANCE

    model = UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=strides,
        num_res_units=num_res_units,
        norm=norm_type,
        dropout=dropout,
    )

    if device is not None:
        model = model.to(device)

    return model


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Returns (total_params, trainable_params).
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_model_summary(model: nn.Module) -> None:
    """Prints architectural details and parameter count."""
    total, trainable = count_parameters(model)
    print("=" * 65)
    print("           OrthoSegNet 3D Residual U-Net Architecture")
    print("=" * 65)
    print(f"Total Parameters:     {total:,}")
    print(f"Trainable Parameters: {trainable:,}")
    print(f"Model Memory Footprint (FP32): {total * 4 / (1024 ** 2):.2f} MB")
    print(f"Model Memory Footprint (FP16): {total * 2 / (1024 ** 2):.2f} MB")
    print("=" * 65)

