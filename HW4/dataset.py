import os
from typing import List
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random


class RainSnowDataset(Dataset):
    def __init__(
        self,
        train_val: str,
        skipped_id: List[int],
        degraded_dir: str = "hw4_realse_dataset/train/degraded/",
        clean_dir: str = "hw4_realse_dataset/train/clean/",
        patch_size: int = None,
        augment: bool = True,
    ):
        """
        Args:
          train_val:    'train' or 'val'
          skipped_id:   list of integer IDs to hold out for validation
          degraded_dir: path containing rain-*.png and snow-*.png
          clean_dir:    path containing rain_clean-*.png and snow_clean-*.png
          patch_size:   if set, randomly crop this size from each image
          augment:      if True, random horizontal/vertical flips + rotations
        """
        assert train_val in (
            "train",
            "val",
        ), "`train_val` must be 'train' or 'val'"

        self.degraded_dir = degraded_dir
        self.clean_dir = clean_dir
        self.patch_size = patch_size
        self.augment = augment
        self.skipped_id = set(skipped_id)
        self.train_val = train_val

        # Build list of (degraded_path, clean_path)
        files = sorted(os.listdir(degraded_dir))
        self.pairs = []
        for fname in files:
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            # expect e.g. "rain-23.png"
            base = os.path.splitext(fname)[0]  # "rain-23"
            try:
                de, id_str = base.split("-", 1)
                img_id = int(id_str)
            except ValueError:
                continue  # skip anything that doesn't match pattern

            if train_val == "train" and (img_id in self.skipped_id):
                continue
            if train_val == "val" and (img_id not in self.skipped_id):
                continue

            clean_name = f"{de}_clean-{img_id}.png"  # "rain_clean-23.png"
            degraded_path = os.path.join(degraded_dir, fname)
            clean_path = os.path.join(clean_dir, clean_name)
            if not os.path.exists(clean_path):
                raise FileNotFoundError(f"Missing clean for {fname}")
            self.pairs.append((degraded_path, clean_path))

        if len(self.pairs) == 0:
            raise RuntimeError(
                f"No {train_val} image pairs found in provided dirs."
            )

        # Transforms
        self.to_tensor = transforms.ToTensor()
        if augment:
            self.aug = transforms.RandomChoice(
                [
                    transforms.RandomHorizontalFlip(1.0),
                    transforms.RandomVerticalFlip(1.0),
                    transforms.RandomRotation(90),
                    transforms.RandomRotation(180),
                    transforms.RandomRotation(270),
                ]
            )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        deg_path, clean_path = self.pairs[idx]
        deg = Image.open(deg_path).convert("RGB")
        clean = Image.open(clean_path).convert("RGB")

        # Crop (same for both)
        if self.patch_size is not None:
            # compose crop on a single PIL image then apply identically
            i, j, h, w = transforms.RandomCrop.get_params(
                deg, output_size=(self.patch_size, self.patch_size)
            )
            deg = deg.crop((j, i, j + w, i + h))
            clean = clean.crop((j, i, j + w, i + h))

        # Augment (same for both)
        if self.augment:
            op = self.aug
            seed = torch.randint(0, 2**32, ()).item()
            torch.manual_seed(seed)
            random.seed(seed)
            deg = op(deg)
            torch.manual_seed(seed)
            random.seed(seed)
            clean = op(clean)

        # To tensor
        deg_t = self.to_tensor(deg)
        clean_t = self.to_tensor(clean)

        return deg_t, clean_t
