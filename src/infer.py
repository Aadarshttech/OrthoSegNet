"""
Inference engine and NIfTI mask exporter for OrthoSegNet.
Generates multi-class vertebral segmentation masks directly compatible with BoneForge3D.
"""

import os
from typing import Tuple, Optional
import numpy as np
import nibabel as nib
import torch
from scipy.ndimage import zoom
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch

from .config import (
    DEFAULT_CONFIG,
    DEFAULT_PATCH_SIZE,
    TARGET_PIXDIM,
    BONE_HU_WINDOW,
    NUM_CLASSES,
)
from .transforms import get_infer_transforms
from .model import build_orthosegnet_model


class OrthoSegInferer:
    """
    Handles sliding-window 3D volumetric inference on clinical CT scans.
    Produces full-volume segmentation masks aligned to patient anatomy.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model: Optional[torch.nn.Module] = None,
        device: Optional[torch.device] = None,
        patch_size: Tuple[int, int, int] = DEFAULT_PATCH_SIZE,
        num_classes: int = NUM_CLASSES,
        overlap: float = 0.5,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.overlap = overlap

        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model = build_orthosegnet_model(out_channels=num_classes, device=self.device)
            if checkpoint_path and os.path.exists(checkpoint_path):
                state_dict = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print(f"Loaded weights from {checkpoint_path}")
            else:
                print("Initialized model with random weights (no checkpoint loaded)")

        self.model.eval()
        self.transform = get_infer_transforms(pixdim=TARGET_PIXDIM, bone_hu_window=BONE_HU_WINDOW)

    @torch.no_grad()
    def segment_volume(
        self,
        ct_path: str,
        sw_batch_size: int = 4,
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float]]:
        """
        Runs sliding-window inference on a CT volume.

        Returns:
            pred_mask_resampled: Resampled 3D prediction array (uint8)
            ct_resampled: Resampled CT volume array (float32)
            zooms: Voxel spacing (dx, dy, dz)
        """
        data = self.transform({"image": ct_path})
        input_tensor = data["image"].unsqueeze(0).to(self.device)  # shape: (1, 1, H, W, D)

        # Sliding window inference with Gaussian weighting to eliminate boundary seam artifacts
        raw_output = sliding_window_inference(
            inputs=input_tensor,
            roi_size=self.patch_size,
            sw_batch_size=sw_batch_size,
            predictor=self.model,
            overlap=self.overlap,
            mode="gaussian",
        )

        pred_mask = torch.argmax(raw_output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        ct_arr = input_tensor.squeeze(0).squeeze(0).cpu().numpy()

        return pred_mask, ct_arr, TARGET_PIXDIM

    def segment_and_export_nifti(
        self,
        ct_path: str,
        output_nifti_path: str,
        align_to_original: bool = True,
    ) -> str:
        """
        Segments a CT scan and exports the output mask as a NIfTI volume (.nii.gz).
        If align_to_original=True, resamples the mask back to the native scanner grid
        and preserves original affine matrix and voxel zooms for direct compatibility
        with BoneForge3D and 3D Slicer.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_nifti_path)), exist_ok=True)

        orig_nii = nib.load(ct_path)
        orig_shape = orig_nii.shape
        orig_affine = orig_nii.affine
        orig_zooms = orig_nii.header.get_zooms()

        pred_mask, ct_proc, proc_zooms = self.segment_volume(ct_path)

        if align_to_original:
            # Resample discrete labels back to native voxel grid using nearest-neighbor
            scale_factors = [
                orig_shape[0] / pred_mask.shape[0],
                orig_shape[1] / pred_mask.shape[1],
                orig_shape[2] / pred_mask.shape[2],
            ]
            final_mask = zoom(pred_mask, scale_factors, order=0)
            final_affine = orig_affine
            final_zooms = orig_zooms
        else:
            final_mask = pred_mask
            final_affine = np.diag([*proc_zooms, 1.0])
            final_zooms = proc_zooms

        out_nii = nib.Nifti1Image(final_mask.astype(np.uint8), final_affine)
        out_nii.header.set_zooms(final_zooms)
        nib.save(out_nii, output_nifti_path)

        print(f"Saved segmentation mask to: {output_nifti_path}")
        print(f"Mask shape: {final_mask.shape} | Unique labels: {np.unique(final_mask)}")
        return output_nifti_path

