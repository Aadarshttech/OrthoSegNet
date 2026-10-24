"""
MONAI volumetric transformation pipelines for CT bone segmentation.
"""

from typing import Tuple, Sequence
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandAffined,
    RandGaussianNoised,
    EnsureTyped,
    AsDiscrete,
)
from .config import BONE_HU_WINDOW, TARGET_PIXDIM, DEFAULT_PATCH_SIZE


def get_train_transforms(
    patch_size: Tuple[int, int, int] = DEFAULT_PATCH_SIZE,
    pixdim: Tuple[float, float, float] = TARGET_PIXDIM,
    bone_hu_window: Tuple[float, float] = BONE_HU_WINDOW,
    samples_per_volume: int = 4,
    pos_ratio: int = 1,
    neg_ratio: int = 1,
) -> Compose:
    """
    Constructs data augmentation and preprocessing pipeline for 3D model training.

    Pipeline stages:
    1. LoadImaged: Read 3D NIfTI CT volume and multi-label segmentation mask.
    2. EnsureChannelFirstd: Ensure shape is (C, H, W, D) with C=1.
    3. Orientationd: Reorient anatomical coordinate system to canonical RAS.
    4. Spacingd: Resample anisotropic voxel spacing to target uniform physical dimensions.
    5. ScaleIntensityRanged: Window Hounsfield Units [-200, 1800] HU to normalized [0.0, 1.0].
    6. CropForegroundd: Discard non-body empty background air to save compute and memory.
    7. RandCropByPosNegLabeld: Balanced patch-based sampling centered on bone vs background.
    8. Spatial & Intensity Augmentations: Random 3D flips, 90-degree rotations, affine shearing,
       and mild Gaussian sensor noise to prevent overfitting.
    9. EnsureTyped: Convert arrays into torch tensors.
    """
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=pixdim,
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=bone_hu_window[0],
            a_max=bone_hu_window[1],
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image", margin=5),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=patch_size,
            pos=pos_ratio,
            neg=neg_ratio,
            num_samples=samples_per_volume,
            image_key="image",
            image_threshold=0.0,
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3, spatial_axes=(0, 1)),
        RandAffined(
            keys=["image", "label"],
            prob=0.2,
            rotate_range=(0.05, 0.05, 0.05),
            scale_range=(0.1, 0.1, 0.1),
            mode=("bilinear", "nearest"),
        ),
        RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.05),
        EnsureTyped(keys=["image", "label"]),
    ])


def get_val_transforms(
    pixdim: Tuple[float, float, float] = TARGET_PIXDIM,
    bone_hu_window: Tuple[float, float] = BONE_HU_WINDOW,
) -> Compose:
    """
    Constructs deterministic preprocessing pipeline for full-volume validation.
    No patch cropping or random augmentations applied here; evaluation is conducted
    via sliding window inference on the full resampled volume.
    """
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=pixdim,
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=bone_hu_window[0],
            a_max=bone_hu_window[1],
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image", margin=5),
        EnsureTyped(keys=["image", "label"]),
    ])


def get_infer_transforms(
    pixdim: Tuple[float, float, float] = TARGET_PIXDIM,
    bone_hu_window: Tuple[float, float] = BONE_HU_WINDOW,
) -> Compose:
    """
    Constructs inference preprocessing pipeline for unannotated test CT scans.
    """
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(
            keys=["image"],
            pixdim=pixdim,
            mode="bilinear",
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=bone_hu_window[0],
            a_max=bone_hu_window[1],
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image"], source_key="image", margin=5),
        EnsureTyped(keys=["image"]),
    ])


def get_post_transforms(num_classes: int) -> Tuple[AsDiscrete, AsDiscrete]:
    """
    Post-processing transformations to discretize predictions and ground-truth
    into one-hot encoded tensors for multi-class Dice evaluation.
    """
    post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)
    post_label = AsDiscrete(to_onehot=num_classes)
    return post_pred, post_label

