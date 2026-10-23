"""
Configuration parameters and anatomical label definitions for OrthoSegNet.
"""

from typing import Dict, Tuple

# Number of segmentation classes: 0 = Background, 1..25 = Vertebrae (C1..L6)
NUM_CLASSES = 26

# VerSe2020 anatomical label mapping
VERTEBRA_LABELS: Dict[int, str] = {
    0: "Background",
    # Cervical spine (C1 - C7)
    1: "C1 (Atlas)",
    2: "C2 (Axis)",
    3: "C3",
    4: "C4",
    5: "C5",
    6: "C6",
    7: "C7",
    # Thoracic spine (T1 - T12)
    8: "T1",
    9: "T2",
    10: "T3",
    11: "T4",
    12: "T5",
    13: "T6",
    14: "T7",
    15: "T8",
    16: "T9",
    17: "T10",
    18: "T11",
    19: "T12",
    # Lumbar spine (L1 - L6)
    20: "L1",
    21: "L2",
    22: "L3",
    23: "L4",
    24: "L5",
    25: "L6 (Transitional)",
}

# Regional groupings for clinical evaluation
VERTEBRA_REGIONS: Dict[str, Tuple[int, ...]] = {
    "Cervical": tuple(range(1, 8)),    # 1 to 7
    "Thoracic": tuple(range(8, 20)),   # 8 to 19
    "Lumbar": tuple(range(20, 26)),    # 20 to 25
}

# Clinical CT Bone Window (in Hounsfield Units)
# Bone attenuates between ~300 to 2000 HU. We retain [-200, 1800] HU to preserve
# soft-tissue context (muscle, discs) while sharpening high-density trabecular and cortical bone.
BONE_HU_WINDOW = (-200.0, 1800.0)

# Resampling target voxel resolution in millimeters (X, Y, Z)
TARGET_PIXDIM = (1.5, 1.5, 2.0)

# 3D Patch dimensions sampled during training
DEFAULT_PATCH_SIZE = (96, 96, 96)

# Default model and training hyperparameters
DEFAULT_CONFIG = {
    "num_classes": NUM_CLASSES,
    "patch_size": DEFAULT_PATCH_SIZE,
    "pixdim": TARGET_PIXDIM,
    "bone_hu_window": BONE_HU_WINDOW,
    "batch_size": 2,
    "samples_per_volume": 4,
    "pos_sample_ratio": 1,
    "neg_sample_ratio": 1,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "max_epochs": 50,
    "val_interval": 5,
    "sliding_window_overlap": 0.5,
    "model_channels": (16, 32, 64, 128, 256),
    "model_strides": (2, 2, 2, 2),
    "model_res_units": 2,
    "seed": 42,
}

