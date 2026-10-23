"""
OrthoSegNet: 3D Vertebral CT Bone Segmentation Pipeline
=======================================================
A deep learning pipeline for multi-vertebra CT segmentation using MONAI and PyTorch.
Outputs volumetric masks directly compatible with 3D mesh reconstruction (BoneForge3D).
"""

__version__ = "1.0.0"
__author__ = "Aadarsh Pandita"

from .config import (
    VERTEBRA_LABELS,
    VERTEBRA_REGIONS,
    DEFAULT_CONFIG,
    BONE_HU_WINDOW,
    NUM_CLASSES,
)
from .dataset import (
    VerSeDatasetManager,
    create_verse_dataloaders,
    generate_synthetic_spine_dataset,
)
from .transforms import (
    get_train_transforms,
    get_val_transforms,
    get_infer_transforms,
)
from .model import build_orthosegnet_model
from .loss import build_segmentation_loss
from .metrics import VertebraEvaluator
from .visualize import (
    plot_orthogonal_slices,
    plot_prediction_comparison,
    plot_training_curves,
    plot_per_class_dice,
)
from .infer import OrthoSegInferer

__all__ = [
    "VERTEBRA_LABELS",
    "VERTEBRA_REGIONS",
    "DEFAULT_CONFIG",
    "BONE_HU_WINDOW",
    "NUM_CLASSES",
    "VerSeDatasetManager",
    "create_verse_dataloaders",
    "generate_synthetic_spine_dataset",
    "get_train_transforms",
    "get_val_transforms",
    "get_infer_transforms",
    "build_orthosegnet_model",
    "build_segmentation_loss",
    "VertebraEvaluator",
    "plot_orthogonal_slices",
    "plot_prediction_comparison",
    "plot_training_curves",
    "plot_per_class_dice",
    "OrthoSegInferer",
]

