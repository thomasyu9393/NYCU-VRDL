import os
import json
import cv2
import skimage.io as sio
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class TrainDataset(Dataset):
    def __init__(self, image_dir, transforms=None):
        """
        image_dir/
          └─ {img_id}/
        ids: list of image identifiers (folder names in train/)
        """
        self.image_dir = image_dir
        self.ids = sorted(os.listdir(image_dir))
        self.transforms = transforms

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = os.path.join(self.image_dir, img_id, "image.tif")
        image = cv2.imread(img_path)  # BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # RGB

        # Load masks and build target
        masks = []
        labels = []
        boxes = []

        for fname in os.listdir(os.path.join(self.image_dir, img_id)):
            if fname.startswith("class") and fname.endswith(".tif"):
                class_id = int(fname[len("class"):-len(".tif")])
                mask = sio.imread(os.path.join(self.image_dir, img_id, fname))
                mask_arr = np.array(mask)
                for inst_id in np.unique(mask_arr):
                    if inst_id == 0:
                        continue
                    # binary mask for this instance
                    inst_mask = (mask_arr == inst_id).astype(np.uint8)
                    # find bbox
                    pos = np.where(inst_mask)
                    xmin, ymin = pos[1].min(), pos[0].min()
                    xmax, ymax = pos[1].max(), pos[0].max()
                    masks.append(inst_mask)
                    labels.append(class_id)
                    boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into torch.Tensor
        if masks:
            masks = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            iscrowd = torch.zeros((len(labels),), dtype=torch.int64)
        else:
            # no objects in image
            masks = torch.zeros(
                (0, image.shape[0], image.shape[1]), dtype=torch.uint8
            )
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": iscrowd,
        }

        if self.transforms:
            image = self.transforms(image)

        return image, target


class TestDataset(Dataset):
    def __init__(self, image_dir, mapping_json, transforms=None):
        self.image_dir = image_dir
        self.transforms = transforms or T.ToTensor()

        # load mapping: list of dicts with keys: file_name, id, height, width
        with open(mapping_json, "r") as f:
            infos = json.load(f)
        self.infos = infos

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, idx):
        info = self.infos[idx]
        img_path = os.path.join(self.image_dir, info["file_name"])
        image = cv2.imread(img_path)  # BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # RGB
        if self.transforms:
            image = self.transforms(image)
        image_id = info["id"]
        return image, image_id
