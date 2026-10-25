"""
Compound loss functions for multi-class 3D CT bone segmentation.
"""

import torch
import torch.nn as nn
from monai.losses import DiceCELoss, DiceFocalLoss


def build_segmentation_loss(
    loss_type: str = "dice_ce",
    to_onehot_y: bool = True,
    softmax: bool = True,
    lambda_dice: float = 1.0,
    lambda_ce: float = 1.0,
    include_background: bool = True,
) -> nn.Module:
    """
    Constructs a compound loss function balancing region overlap with voxel-level classification.

    Why Dice + Cross-Entropy?
    - Class Imbalance: Vertebral structures comprise <5% of total CT volume, with >95% background.
      Standard Cross-Entropy alone tends to collapse towards the background class.
    - Dice Loss provides scale-invariant gradient signals based on volumetric intersection:
        L_Dice = 1 - (2 * |X ∩ Y| + ε) / (|X| + |Y| + ε)
    - Cross-Entropy Loss provides smooth convex gradients early in training to avoid local minima:
        L_CE = - Σ y_c log(p_c)
    - Combined Loss:
        L_Total = λ_dice * L_Dice + λ_ce * L_CE
    """
    if loss_type.lower() == "dice_ce":
        return DiceCELoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            softmax=softmax,
            lambda_dice=lambda_dice,
            lambda_ce=lambda_ce,
        )
    elif loss_type.lower() == "dice_focal":
        return DiceFocalLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            softmax=softmax,
            lambda_dice=lambda_dice,
            lambda_focal=lambda_ce,
        )
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}. Choose 'dice_ce' or 'dice_focal'.")

