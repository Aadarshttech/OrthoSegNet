"""
Modular training script for OrthoSegNet 3D Vertebral CT Bone Segmentation.
Supports mixed precision (AMP), Cosine Annealing LR scheduling, and multi-class Dice evaluation.
"""

import os
import argparse
import time
from typing import Dict, Any, Optional
import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.utils import set_determinism

from .config import (
    DEFAULT_CONFIG,
    NUM_CLASSES,
    DEFAULT_PATCH_SIZE,
    TARGET_PIXDIM,
    BONE_HU_WINDOW,
)
from .dataset import (
    VerSeDatasetManager,
    create_verse_dataloaders,
    generate_synthetic_spine_dataset,
)
from .transforms import (
    get_train_transforms,
    get_val_transforms,
    get_post_transforms,
)
from .model import build_orthosegnet_model, print_model_summary
from .loss import build_segmentation_loss
from .metrics import VertebraEvaluator
from .visualize import plot_training_curves, plot_per_class_dice


def train_orthosegnet(
    data_dir: Optional[str] = None,
    output_dir: str = "results",
    checkpoint_dir: str = "checkpoints",
    max_epochs: int = 50,
    batch_size: int = 2,
    learning_rate: float = 1e-4,
    val_interval: int = 5,
    patch_size: tuple = DEFAULT_PATCH_SIZE,
    num_classes: int = NUM_CLASSES,
    device: Optional[torch.device] = None,
    use_amp: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Executes full training and validation lifecycle.
    """
    set_determinism(seed=seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing OrthoSegNet training on device: {device}")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 1. Dataset discovery
    manager = VerSeDatasetManager(data_dir=data_dir)
    train_files, val_files = manager.get_train_val_split(val_ratio=0.2, seed=seed)
    print(f"Dataset split: {len(train_files)} training cases | {len(val_files)} validation cases")

    # 2. Transforms and DataLoaders
    train_tf = get_train_transforms(patch_size=patch_size, pixdim=TARGET_PIXDIM, bone_hu_window=BONE_HU_WINDOW)
    val_tf = get_val_transforms(pixdim=TARGET_PIXDIM, bone_hu_window=BONE_HU_WINDOW)
    post_pred, post_label = get_post_transforms(num_classes=num_classes)

    train_loader, val_loader = create_verse_dataloaders(
        train_files, val_files, train_tf, val_tf, batch_size=batch_size, num_workers=2
    )

    # 3. Model, Loss, Optimizer, Scheduler
    model = build_orthosegnet_model(out_channels=num_classes, device=device)
    print_model_summary(model)

    loss_fn = build_segmentation_loss(loss_type="dice_ce")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    scaler = GradScaler(enabled=use_amp and device.type == "cuda")
    evaluator = VertebraEvaluator(num_classes=num_classes)

    best_metric = -1.0
    best_epoch = -1
    train_losses = []
    val_metrics = []

    best_model_path = os.path.join(checkpoint_dir, "best_orthosegnet.pth")

    start_time = time.time()
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        step_count = 0

        for batch_data in train_loader:
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=use_amp and device.type == "cuda"):
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)

            if use_amp and device.type == "cuda":
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            epoch_loss += loss.item()
            step_count += 1

        scheduler.step()
        avg_loss = epoch_loss / max(1, step_count)
        train_losses.append(avg_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch [{epoch+1:02d}/{max_epochs:02d}] - Loss: {avg_loss:.4f} - LR: {current_lr:.6f}")

        # Validation step
        if (epoch + 1) % val_interval == 0 or (epoch + 1) == max_epochs:
            model.eval()
            evaluator.reset()

            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs = val_data["image"].to(device)
                    val_labels = val_data["label"].to(device)

                    with autocast(enabled=use_amp and device.type == "cuda"):
                        val_outputs = sliding_window_inference(
                            inputs=val_inputs,
                            roi_size=patch_size,
                            sw_batch_size=4,
                            predictor=model,
                            overlap=DEFAULT_CONFIG["sliding_window_overlap"],
                            mode="gaussian",
                        )

                    # Decollate batch and apply one-hot post-transforms
                    val_preds = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_trues = [post_label(i) for i in decollate_batch(val_labels)]
                    evaluator.update(val_preds, val_trues)

            df_metrics, regional = evaluator.compute_summary()
            mean_dice = regional["Overall"]
            val_metrics.append(mean_dice)

            print(f"  --> Validation Mean Dice: {mean_dice:.4f} | Cervical: {regional['Cervical']:.4f} | "
                  f"Thoracic: {regional['Thoracic']:.4f} | Lumbar: {regional['Lumbar']:.4f}")

            if mean_dice > best_metric:
                best_metric = mean_dice
                best_epoch = epoch + 1
                torch.save(model.state_dict(), best_model_path)
                print(f"  >>> Checkpoint Saved! New Best Dice: {best_metric:.4f} at epoch {best_epoch}")

    total_time = time.time() - start_time
    print("=" * 65)
    print(f"Training Complete in {total_time/60:.2f} minutes.")
    print(f"Best Validation Mean Dice: {best_metric:.4f} at epoch {best_epoch}")
    print(f"Best weights saved to: {best_model_path}")

    # Generate and save final training figures
    plot_training_curves(
        train_losses=train_losses,
        val_metrics=val_metrics,
        val_interval=val_interval,
        save_path=os.path.join(output_dir, "training_curves.png"),
    )

    # Export metrics table
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.eval()
        evaluator.reset()
        with torch.no_grad():
            for val_data in val_loader:
                v_in = val_data["image"].to(device)
                v_lbl = val_data["label"].to(device)
                v_out = sliding_window_inference(v_in, patch_size, 4, model, mode="gaussian")
                evaluator.update(
                    [post_pred(i) for i in decollate_batch(v_out)],
                    [post_label(i) for i in decollate_batch(v_lbl)],
                )
        df_metrics, regional = evaluator.compute_summary()
        metrics_csv_path = os.path.join(output_dir, "vertebra_segmentation_metrics.csv")
        df_metrics.to_csv(metrics_csv_path, index=False)
        plot_per_class_dice(df_metrics, save_path=os.path.join(output_dir, "per_class_dice.png"))

    return {
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "train_losses": train_losses,
        "val_metrics": val_metrics,
        "best_model_path": best_model_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train OrthoSegNet 3D Vertebral Bone Segmentation")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory containing VerSe CT and mask NIfTI files")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory for plots and results")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory for model weights")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Patch batch size per step")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--val_interval", type=int, default=5, help="Validation interval in epochs")
    args = parser.parse_args()

    train_orthosegnet(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        val_interval=args.val_interval,
    )

