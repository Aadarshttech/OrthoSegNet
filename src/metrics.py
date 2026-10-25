"""
Evaluation metrics for multi-vertebra CT bone segmentation.
Computes per-class Dice scores, regional summaries (Cervical, Thoracic, Lumbar),
and exports metric tables.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import torch
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from .config import VERTEBRA_LABELS, VERTEBRA_REGIONS, NUM_CLASSES


class VertebraEvaluator:
    """
    Computes per-vertebra and regional segmentation metrics across validation cases.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        self.num_classes = num_classes
        # Monai DiceMetric with include_background=False for foreground vertebrae
        self.dice_metric = DiceMetric(
            include_background=False,
            reduction="none",  # Retain per-class scores
            get_not_nans=False,
        )

    def reset(self) -> None:
        self.dice_metric.reset()

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        """
        Takes one-hot encoded prediction and ground truth tensors:
        y_pred: (B, C, H, W, D) or list of decollated items
        y_true: (B, C, H, W, D) or list of decollated items
        """
        self.dice_metric(y_pred=y_pred, y=y_true)

    def compute_summary(self) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Aggregates accumulated batch metrics and generates per-class
        and regional performance summaries.

        Returns:
            df_per_class: pd.DataFrame with columns ['Class ID', 'Vertebra', 'Region', 'Dice']
            regional_summary: dict of {'Cervical': float, 'Thoracic': float, 'Lumbar': float, 'Overall': float}
        """
        raw_dice = self.dice_metric.aggregate()  # shape: (N_batches, C-1)
        self.reset()

        if isinstance(raw_dice, torch.Tensor):
            dice_scores = raw_dice.cpu().numpy()
        else:
            dice_scores = np.asarray(raw_dice)

        # Average across batches / cases
        if dice_scores.ndim > 1:
            mean_per_class = np.nanmean(dice_scores, axis=0)
        else:
            mean_per_class = dice_scores

        rows = []
        # Classes 1 to NUM_CLASSES - 1 (background excluded)
        for cls_id in range(1, self.num_classes):
            idx = cls_id - 1
            score = float(mean_per_class[idx]) if idx < len(mean_per_class) else np.nan

            # Determine anatomical region
            if 1 <= cls_id <= 7:
                region = "Cervical"
            elif 8 <= cls_id <= 19:
                region = "Thoracic"
            elif 20 <= cls_id <= 25:
                region = "Lumbar"
            else:
                region = "Other"

            name = VERTEBRA_LABELS.get(cls_id, f"Class {cls_id}")
            rows.append({
                "Class_ID": cls_id,
                "Vertebra": name,
                "Region": region,
                "Dice": score,
            })

        df = pd.DataFrame(rows)

        # Calculate regional means (ignoring NaN where a vertebra was not present in the scan)
        regional_summary: Dict[str, float] = {}
        for region_name in ["Cervical", "Thoracic", "Lumbar"]:
            sub_df = df[df["Region"] == region_name]
            valid_scores = sub_df["Dice"].dropna()
            regional_summary[region_name] = float(valid_scores.mean()) if len(valid_scores) > 0 else 0.0

        valid_all = df["Dice"].dropna()
        regional_summary["Overall"] = float(valid_all.mean()) if len(valid_all) > 0 else 0.0

        return df, regional_summary


def compute_binary_dice(pred_mask: np.ndarray, true_mask: np.ndarray, smooth: float = 1e-6) -> float:
    """Computes binary Dice coefficient between two boolean or binary arrays."""
    p = pred_mask.astype(bool)
    t = true_mask.astype(bool)
    intersection = np.logical_and(p, t).sum()
    total = p.sum() + t.sum()
    if total == 0:
        return 1.0  # Perfect agreement on empty structure
    return float((2.0 * intersection + smooth) / (total + smooth))

