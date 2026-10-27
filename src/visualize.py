"""
Visualization utilities for medical CT scans, multi-vertebra segmentation masks,
training dynamics, and per-class performance comparisons.
"""

from typing import Optional, Tuple, Sequence, Dict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd


def get_discrete_vertebra_cmap(num_classes: int = 26) -> mcolors.ListedColormap:
    """
    Constructs a high-contrast discrete colormap where 0 is transparent/black,
    and each vertebra has a distinct distinguishable color.
    """
    colors = plt.cm.turbo(np.linspace(0.05, 0.95, num_classes))
    colors[0] = [0.0, 0.0, 0.0, 0.0]  # Background is transparent
    return mcolors.ListedColormap(colors)


def plot_orthogonal_slices(
    ct_volume: np.ndarray,
    mask_volume: Optional[np.ndarray] = None,
    slice_indices: Optional[Tuple[int, int, int]] = None,
    save_path: Optional[str] = None,
    dpi: int = 150,
    title: str = "Multiplanar CT Orthogonal Cross-Sections",
) -> plt.Figure:
    """
    Visualizes Axial (XY), Coronal (XZ), and Sagittal (YZ) cross-sectional views.
    Optionally overlays segmentation masks with alpha transparency.
    """
    H, W, D = ct_volume.shape
    if slice_indices is None:
        idx_axial = D // 2
        idx_coronal = H // 2
        idx_sagittal = W // 2
    else:
        idx_axial, idx_coronal, idx_sagittal = slice_indices

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    cmap_mask = get_discrete_vertebra_cmap()

    # Axial: ct_volume[:, :, idx_axial]
    ax = axes[0]
    ax.imshow(np.rot90(ct_volume[:, :, idx_axial]), cmap="bone")
    if mask_volume is not None:
        ax.imshow(np.rot90(mask_volume[:, :, idx_axial]), cmap=cmap_mask, alpha=0.55, interpolation="nearest")
    ax.set_title(f"Axial (Z = {idx_axial})", fontsize=14, fontweight="bold")
    ax.axis("off")

    # Coronal: ct_volume[idx_coronal, :, :]
    ax = axes[1]
    ax.imshow(np.rot90(ct_volume[idx_coronal, :, :]), cmap="bone")
    if mask_volume is not None:
        ax.imshow(np.rot90(mask_volume[idx_coronal, :, :]), cmap=cmap_mask, alpha=0.55, interpolation="nearest")
    ax.set_title(f"Coronal (Y = {idx_coronal})", fontsize=14, fontweight="bold")
    ax.axis("off")

    # Sagittal: ct_volume[:, idx_sagittal, :]
    ax = axes[2]
    ax.imshow(np.rot90(ct_volume[:, idx_sagittal, :]), cmap="bone")
    if mask_volume is not None:
        ax.imshow(np.rot90(mask_volume[:, idx_sagittal, :]), cmap=cmap_mask, alpha=0.55, interpolation="nearest")
    ax.set_title(f"Sagittal (X = {idx_sagittal})", fontsize=14, fontweight="bold")
    ax.axis("off")

    plt.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_prediction_comparison(
    ct_slice: np.ndarray,
    gt_slice: np.ndarray,
    pred_slice: np.ndarray,
    view_name: str = "Axial View",
    save_path: Optional[str] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Renders 4-panel diagnostic comparison:
    1. Input CT Slice
    2. Ground Truth Annotation
    3. OrthoSegNet Prediction
    4. Difference Map (TP = Green, FP = Red, FN = Blue)
    """
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
    cmap_mask = get_discrete_vertebra_cmap()

    # 1. CT
    axes[0].imshow(np.rot90(ct_slice), cmap="bone")
    axes[0].set_title(f"1. Input CT ({view_name})", fontsize=13, fontweight="bold")
    axes[0].axis("off")

    # 2. Ground Truth
    axes[1].imshow(np.rot90(ct_slice), cmap="gray", alpha=0.6)
    axes[1].imshow(np.rot90(gt_slice), cmap=cmap_mask, alpha=0.7, interpolation="nearest")
    axes[1].set_title("2. Ground Truth Mask", fontsize=13, fontweight="bold")
    axes[1].axis("off")

    # 3. Model Prediction
    axes[2].imshow(np.rot90(ct_slice), cmap="gray", alpha=0.6)
    axes[2].imshow(np.rot90(pred_slice), cmap=cmap_mask, alpha=0.7, interpolation="nearest")
    axes[2].set_title("3. OrthoSegNet Prediction", fontsize=13, fontweight="bold")
    axes[2].axis("off")

    # 4. Error / Discrepancy Map
    # TP (both > 0): Green, FP (pred > 0, gt == 0): Red, FN (gt > 0, pred == 0): Blue
    H, W = ct_slice.shape
    diff_rgb = np.zeros((H, W, 3), dtype=np.float32)
    diff_rgb[:, :] = 0.1  # dark background

    tp = (pred_slice > 0) & (gt_slice > 0)
    fp = (pred_slice > 0) & (gt_slice == 0)
    fn = (pred_slice == 0) & (gt_slice > 0)

    diff_rgb[tp] = [0.18, 0.80, 0.44]  # Green: True Positive
    diff_rgb[fp] = [0.91, 0.30, 0.24]  # Red: False Positive
    diff_rgb[fn] = [0.20, 0.60, 0.86]  # Blue: False Negative

    axes[3].imshow(np.rot90(diff_rgb))
    axes[3].set_title("4. Error Map (TP=Green, FP=Red, FN=Blue)", fontsize=13, fontweight="bold")
    axes[3].axis("off")

    plt.suptitle(f"Vertebral Bone Segmentation Diagnostics — {view_name}", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_training_curves(
    train_losses: Sequence[float],
    val_metrics: Sequence[float],
    val_interval: int = 5,
    save_path: Optional[str] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Renders dual-axis publication-style training curve figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    epochs = list(range(1, len(train_losses) + 1))
    ax1.plot(epochs, train_losses, color="#e74c3c", linewidth=2.2, label="Train Loss (DiceCE)")
    ax1.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax1.set_ylabel("DiceCE Loss", fontsize=12, fontweight="bold")
    ax1.set_title("Training Loss Convergence", fontsize=14, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True)

    # Validation Dice curve
    val_epochs = list(range(val_interval, len(train_losses) + 1, val_interval))
    if len(val_epochs) > len(val_metrics):
        val_epochs = val_epochs[:len(val_metrics)]
    elif len(val_metrics) > len(val_epochs):
        val_epochs = [i * val_interval for i in range(1, len(val_metrics) + 1)]

    ax2.plot(val_epochs, val_metrics, color="#2ecc71", marker="o", linewidth=2.2, label="Validation Mean Dice")
    if len(val_metrics) > 0:
        best_idx = int(np.argmax(val_metrics))
        best_epoch = val_epochs[best_idx]
        best_score = val_metrics[best_idx]
        ax2.scatter([best_epoch], [best_score], color="#f39c12", s=120, zorder=5, label=f"Best: {best_score:.4f} (ep {best_epoch})")

    ax2.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Mean Dice Score", fontsize=12, fontweight="bold")
    ax2.set_title("Validation Overlap Metric", fontsize=14, fontweight="bold")
    ax2.set_ylim([0.0, 1.0])
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="lower right", frameon=True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_per_class_dice(
    df_metrics: pd.DataFrame,
    save_path: Optional[str] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Renders bar chart of Dice scores per individual vertebra,
    color-coded by anatomical spinal region.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    region_colors = {
        "Cervical": "#3498db",  # Blue
        "Thoracic": "#e67e22",  # Orange
        "Lumbar": "#2ecc71",    # Green
        "Other": "#95a5a6",     # Gray
    }

    colors = [region_colors.get(r, "#95a5a6") for r in df_metrics["Region"]]
    x_positions = np.arange(len(df_metrics))

    bars = ax.bar(x_positions, df_metrics["Dice"], color=colors, edgecolor="black", linewidth=0.8, width=0.65)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(df_metrics["Vertebra"], rotation=60, ha="right", fontsize=10, fontweight="bold")
    ax.set_ylabel("Dice Similarity Coefficient (DSC)", fontsize=12, fontweight="bold")
    ax.set_title("OrthoSegNet Per-Vertebra Segmentation Accuracy (VerSe2020)", fontsize=14, fontweight="bold")
    ax.set_ylim([0.0, 1.0])
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Legend for regions
    import matplotlib.patches as mpatches
    legend_handles = [
        mpatches.Patch(color=col, label=reg)
        for reg, col in [("Cervical (C1–C7)", "#3498db"), ("Thoracic (T1–T12)", "#e67e22"), ("Lumbar (L1–L6)", "#2ecc71")]
    ]
    ax.legend(handles=legend_handles, loc="lower left", frameon=True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig

