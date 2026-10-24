"""
Dataset loading, HuggingFace automated download, synthetic CT data generation,
and DataLoader utilities for VerSe2020 vertebral CT segmentation.
"""

import os
import glob
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import nibabel as nib
import torch
from monai.data import Dataset, CacheDataset, DataLoader

logger = logging.getLogger("OrthoSegNet.dataset")


class VerSeDatasetManager:
    """
    Manages discovery, downloading, and splitting of VerSe CT volumes and masks.
    Supports Kaggle datasets, Hugging Face Hub (MedOtter/VerSe), local paths,
    and synthetic fallbacks.
    """

    KAGGLE_SEARCH_PATHS = [
        "/kaggle/input/verse2020",
        "/kaggle/input/verse-2020",
        "/kaggle/input/verse",
        "/kaggle/input/vertebrae-segmentation-verse2020",
    ]

    HF_REPO_ID = "MedOtter/VerSe"

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = data_dir or self._discover_data_dir()

    def _discover_data_dir(self) -> str:
        """Auto-detects the active data location."""
        # 1. Check Kaggle input paths
        for k_path in self.KAGGLE_SEARCH_PATHS:
            if os.path.exists(k_path):
                logger.info(f"Detected Kaggle dataset path: {k_path}")
                return k_path

        # 2. Check local data directories
        local_candidates = [
            "data",
            "verse_data",
            os.path.join(os.path.dirname(__file__), "..", "data"),
            os.path.join(os.path.dirname(__file__), "..", "sample_data"),
        ]
        for l_path in local_candidates:
            abs_l_path = os.path.abspath(l_path)
            if os.path.exists(abs_l_path) and len(glob.glob(os.path.join(abs_l_path, "**", "*.nii*"), recursive=True)) > 0:
                logger.info(f"Found existing local dataset at: {abs_l_path}")
                return abs_l_path

        default_dir = os.path.abspath("data")
        os.makedirs(default_dir, exist_ok=True)
        return default_dir

    def download_huggingface_cases(
        self,
        target_dir: Optional[str] = None,
        max_cases: int = 5,
        target_case_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, str]]:
        """
        Downloads paired CT and segmentation masks from HuggingFace dataset MedOtter/VerSe.
        Default includes sub-verse500 for direct interoperability with BoneForge3D.
        """
        try:
            from huggingface_hub import HfApi, hf_hub_download
        except ImportError:
            raise ImportError("huggingface_hub is required for automated download: pip install huggingface_hub")

        dest_dir = target_dir or self.data_dir
        os.makedirs(dest_dir, exist_ok=True)

        api = HfApi()
        all_files = api.list_repo_files(repo_id=self.HF_REPO_ID, repo_type="dataset")

        # Discover paired CT scans and masks in dataset-01training
        raw_files = [f for f in all_files if f.startswith("dataset-01training/rawdata/") and f.endswith(".nii.gz")]
        deriv_files = [f for f in all_files if f.startswith("dataset-01training/derivatives/") and f.endswith(".nii.gz")]

        # Group by subject ID
        case_map: Dict[str, Dict[str, str]] = {}
        for r in raw_files:
            sub_id = r.split("/")[2]
            case_map.setdefault(sub_id, {})["ct"] = r

        for d in deriv_files:
            sub_id = d.split("/")[2]
            if sub_id in case_map:
                case_map[sub_id]["seg"] = d

        paired_cases = {k: v for k, v in case_map.items() if "ct" in v and "seg" in v}
        logger.info(f"Found {len(paired_cases)} paired cases in {self.HF_REPO_ID}")

        # Prioritize requested cases (e.g. sub-verse500)
        selected_sub_ids = []
        if target_case_ids:
            for cid in target_case_ids:
                if cid in paired_cases and cid not in selected_sub_ids:
                    selected_sub_ids.append(cid)

        # Ensure sub-verse500 is included if present
        if "sub-verse500" in paired_cases and "sub-verse500" not in selected_sub_ids:
            selected_sub_ids.append("sub-verse500")

        for cid in paired_cases:
            if len(selected_sub_ids) >= max_cases:
                break
            if cid not in selected_sub_ids:
                selected_sub_ids.append(cid)

        downloaded_pairs = []
        for sub_id in selected_sub_ids:
            ct_hf_path = paired_cases[sub_id]["ct"]
            seg_hf_path = paired_cases[sub_id]["seg"]

            logger.info(f"Downloading {sub_id} CT ({ct_hf_path})...")
            local_ct = hf_hub_download(
                repo_id=self.HF_REPO_ID,
                filename=ct_hf_path,
                repo_type="dataset",
                local_dir=dest_dir,
            )

            logger.info(f"Downloading {sub_id} Mask ({seg_hf_path})...")
            local_seg = hf_hub_download(
                repo_id=self.HF_REPO_ID,
                filename=seg_hf_path,
                repo_type="dataset",
                local_dir=dest_dir,
            )

            downloaded_pairs.append({
                "subject_id": sub_id,
                "image": local_ct,
                "label": local_seg,
            })

        return downloaded_pairs

    def find_paired_files(self, search_dir: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Scans directory tree for paired CT volumes and multi-label masks.
        Handles both VerSe2020 rawdata/derivatives structures and flattened structures.
        """
        root = search_dir or self.data_dir
        if not os.path.exists(root):
            return []

        all_nii = glob.glob(os.path.join(root, "**", "*.nii*"), recursive=True)
        ct_candidates = [
            f for f in all_nii
            if ("_ct." in f.lower() or "_ct.nii" in f.lower() or "_raw." in f.lower() or "image" in f.lower())
            and "seg" not in f.lower() and "msk" not in f.lower()
        ]

        paired = []
        for ct_path in sorted(ct_candidates):
            ct_dir = os.path.dirname(ct_path)
            ct_name = os.path.basename(ct_path)

            # Match standard VerSe patterns:
            # sub-verse001_CT.nii.gz -> sub-verse001_seg.nii.gz
            # sub-gl003_dir-ax_ct.nii.gz -> sub-gl003_dir-ax_seg-vert_msk.nii.gz
            base_prefix = ct_name.split("_dir-ax")[0].split("_CT")[0].split("_ct")[0]

            # Look in same directory or parallel derivatives folder
            seg_matches = []
            search_folders = [ct_dir, os.path.abspath(os.path.join(ct_dir, "..", "derivatives", base_prefix)), root]

            for s_folder in search_folders:
                if os.path.exists(s_folder):
                    matches = glob.glob(os.path.join(s_folder, f"*{base_prefix}*seg*.nii*")) + \
                              glob.glob(os.path.join(s_folder, f"*{base_prefix}*msk*.nii*"))
                    seg_matches.extend([m for m in matches if m != ct_path])

            if seg_matches:
                paired.append({
                    "subject_id": base_prefix,
                    "image": ct_path,
                    "label": seg_matches[0],
                })

        # Deduplicate
        seen_images = set()
        unique_paired = []
        for item in paired:
            if item["image"] not in seen_images:
                seen_images.add(item["image"])
                unique_paired.append(item)

        return unique_paired

    def get_train_val_split(
        self,
        val_ratio: float = 0.2,
        seed: int = 42,
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """
        Discovers data pairs and splits into training and validation sets.
        If no files exist, automatically falls back to synthetic generation.
        """
        pairs = self.find_paired_files()

        if len(pairs) == 0:
            logger.warning("No paired VerSe files detected. Generating synthetic spine data for testing...")
            synth_dir = os.path.join(self.data_dir, "synthetic_verse")
            pairs = generate_synthetic_spine_dataset(target_dir=synth_dir, num_cases=4)

        np.random.seed(seed)
        shuffled = np.random.permutation(pairs).tolist()
        val_count = max(1, int(len(shuffled) * val_ratio))
        val_files = shuffled[:val_count]
        train_files = shuffled[val_count:]

        if len(train_files) == 0:
            train_files = val_files

        return train_files, val_files


def generate_synthetic_spine_dataset(
    target_dir: str = "data/synthetic_verse",
    num_cases: int = 3,
    spatial_shape: Tuple[int, int, int] = (96, 96, 128),
    voxel_spacing: Tuple[float, float, float] = (1.5, 1.5, 2.0),
    seed: int = 42,
) -> List[Dict[str, str]]:
    """
    Generates realistic 3D volumetric CT vertebrae and anatomical masks.
    Simulates:
    - Body attenuation: Air (-1000 HU), Soft tissue background (~35 HU)
    - Vertebral bodies (cylindrical shell + interior cancellous bone: 300 to 1400 HU)
    - Posterior elements (pedicles, lamina, spinous processes)
    - Spinal canal lumen (low HU fluid space)
    - VerSe anatomical labels (T10=17, T11=18, T12=19, L1=20, L2=21, L3=22, L4=23, L5=24)
    """
    os.makedirs(target_dir, exist_ok=True)
    rng = np.random.RandomState(seed)

    H, W, D = spatial_shape
    dx, dy, dz = voxel_spacing

    # Affine matrix with physical zooms
    affine = np.diag([dx, dy, dz, 1.0])

    generated_cases = []

    for i in range(num_cases):
        sub_id = f"sub-syn{i+1:03d}"
        case_dir = os.path.join(target_dir, sub_id)
        os.makedirs(case_dir, exist_ok=True)

        ct_path = os.path.join(case_dir, f"{sub_id}_dir-ax_ct.nii.gz")
        seg_path = os.path.join(case_dir, f"{sub_id}_dir-ax_seg-vert_msk.nii.gz")

        # 1. Base HU grid: Background soft tissue + mild sensor noise
        ct_data = rng.normal(loc=35.0, scale=20.0, size=(H, W, D)).astype(np.float32)

        # Add air outside patient body ellipse
        y_coords, x_coords = np.ogrid[:H, :W]
        center_y, center_x = H // 2, W // 2
        body_mask_2d = ((y_coords - center_y)**2 / (0.42 * H)**2 + (x_coords - center_x)**2 / (0.38 * W)**2) <= 1.0
        for z in range(D):
            ct_data[~body_mask_2d, z] = rng.normal(loc=-950.0, scale=30.0, size=(~body_mask_2d).sum())

        seg_data = np.zeros((H, W, D), dtype=np.uint8)

        # 2. Simulate 6 consecutive vertebrae along Z-axis (T12 to L5)
        # VerSe labels: T12=19, L1=20, L2=21, L3=22, L4=23, L5=24
        vert_labels = [19, 20, 21, 22, 23, 24]
        num_verts = len(vert_labels)
        z_slice_per_vert = (D - 20) // num_verts

        for v_idx, label_id in enumerate(vert_labels):
            z_start = 10 + v_idx * z_slice_per_vert
            z_end = z_start + z_slice_per_vert - 3  # intervertebral disc gap
            z_mid = (z_start + z_end) // 2

            # Vertebral body center (anterior)
            body_cy = center_y + int(0.04 * H)
            body_cx = center_x
            radius_x = int(0.14 * W)
            radius_y = int(0.12 * H)

            # Spinal canal center (posterior to vertebral body)
            canal_cy = center_y - int(0.10 * H)
            canal_cx = center_x
            canal_r = int(0.06 * W)

            # Spinous process (posterior extension)
            spinous_cy = center_y - int(0.22 * H)
            spinous_cx = center_x

            for z in range(z_start, min(z_end, D)):
                # Z-tapering factor
                z_factor = 1.0 - 0.25 * ((z - z_mid) / (z_slice_per_vert / 2.0))**2

                # Vertebral body (cylinder)
                dist_body = ((y_coords - body_cy)**2 / (radius_y * z_factor)**2 +
                             (x_coords - body_cx)**2 / (radius_x * z_factor)**2)
                body_mask = dist_body <= 1.0

                # Cortical bone rim vs cancellous core
                cortical_rim = (dist_body <= 1.0) & (dist_body >= 0.75)
                cancellous_core = dist_body < 0.75

                # Posterior arch (ring around canal)
                dist_canal = ((y_coords - canal_cy)**2 + (x_coords - canal_cx)**2) / (canal_r**2)
                arch_mask = (dist_canal <= 2.2) & (dist_canal >= 0.8) & (y_coords <= canal_cy + 2)

                # Spinous process
                dist_spinous = ((y_coords - spinous_cy)**2 / (0.10 * H)**2 + (x_coords - spinous_cx)**2 / (0.03 * W)**2)
                spinous_mask = (dist_spinous <= 1.0) & (y_coords <= canal_cy)

                # Combine vertebra components
                vert_mask = (body_mask | arch_mask | spinous_mask) & body_mask_2d

                # Assign label in segmentation mask
                seg_data[vert_mask, z] = label_id

                # Assign realistic HU attenuations:
                # Cortical bone: 1100-1600 HU
                # Trabecular interior: 350-550 HU
                # Posterior elements: 800-1300 HU
                # Spinal canal lumen: 10-25 HU (CSF)
                ct_data[cancellous_core & body_mask_2d, z] = rng.normal(420.0, 45.0, size=(cancellous_core & body_mask_2d).sum())
                ct_data[cortical_rim & body_mask_2d, z] = rng.normal(1350.0, 110.0, size=(cortical_rim & body_mask_2d).sum())
                ct_data[(arch_mask | spinous_mask) & body_mask_2d, z] = rng.normal(1050.0, 95.0, size=((arch_mask | spinous_mask) & body_mask_2d).sum())
                ct_data[(dist_canal < 0.8) & body_mask_2d, z] = rng.normal(15.0, 10.0, size=((dist_canal < 0.8) & body_mask_2d).sum())

        # Save NIfTI files
        ct_nii = nib.Nifti1Image(ct_data, affine)
        ct_nii.header.set_zooms(voxel_spacing)
        nib.save(ct_nii, ct_path)

        seg_nii = nib.Nifti1Image(seg_data, affine)
        seg_nii.header.set_zooms(voxel_spacing)
        nib.save(seg_nii, seg_path)

        generated_cases.append({
            "subject_id": sub_id,
            "image": os.path.abspath(ct_path),
            "label": os.path.abspath(seg_path),
        })

    logger.info(f"Generated {len(generated_cases)} synthetic VerSe cases in {target_dir}")
    return generated_cases


def create_verse_dataloaders(
    train_files: List[Dict[str, str]],
    val_files: List[Dict[str, str]],
    train_transforms,
    val_transforms,
    batch_size: int = 2,
    num_workers: int = 2,
    cache_dataset: bool = False,
    cache_rate: float = 1.0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Constructs PyTorch/MONAI DataLoaders for training and validation.
    """
    if cache_dataset:
        train_ds = CacheDataset(
            data=train_files,
            transform=train_transforms,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )
        val_ds = CacheDataset(
            data=val_files,
            transform=val_transforms,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )
    else:
        train_ds = Dataset(data=train_files, transform=train_transforms)
        val_ds = Dataset(data=val_files, transform=val_transforms)

    # Pin memory for faster CPU to GPU transfer
    pin_mem = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_mem,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
    )

    return train_loader, val_loader

