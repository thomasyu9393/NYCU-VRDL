import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CocoDigitDataset(Dataset):
    def __init__(self, img_dir, ann_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform or transforms.ToTensor()

        # Load the COCO format annotations
        with open(ann_file, 'r') as f:
            data = json.load(f)

        self.images = {img['id']: img for img in data['images']}
        self.annotations = {}
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)

        self.image_ids = list(self.images.keys())

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        img_info = self.images[image_id]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")

        # Get annotations for this image
        anns = self.annotations.get(image_id, [])
        boxes = []
        labels = []
        areas = []
        for ann in anns:
            x_min, y_min, w, h = ann['bbox']
            boxes.append([x_min, y_min, x_min + w, y_min + h])
            labels.append(ann['category_id'])
            areas.append(w * h)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor(image_id),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64),
            'area': areas
        }

        if self.transform is not None:
            image = self.transform(image)

        return image, target

    def __len__(self):
        return len(self.image_ids)


class TestDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.image_files = sorted([
            fname for fname in os.listdir(img_dir) if fname.endswith('.png')
        ])
        self.transform = transform or transforms.ToTensor()

    def __getitem__(self, index):
        img_name = self.image_files[index]
        image_id = int(os.path.splitext(img_name)[0])  # e.g., "12.jpg" -> 12
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        # Return dummy "target" with just image_id for compatibility
        target = {"image_id": torch.tensor(image_id)}
        return image, target

    def __len__(self):
        return len(self.image_files)


def get_dataset(args, mode='train'):
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts [0,255] -> [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    if mode == 'train':
        dataset = CocoDigitDataset(
            os.path.join(args.data_dir, "train"),
            os.path.join(args.data_dir, "train.json"),
            transform=transform
        )
    elif mode == 'valid':
        dataset = CocoDigitDataset(
            os.path.join(args.data_dir, "valid"),
            os.path.join(args.data_dir, "valid.json"),
            transform=transform
        )
    elif mode == 'test':
        dataset = TestDataset(
            os.path.join(args.data_dir, "test"),
            transform=transform
        )
    return dataset


def get_dataloader(args, dataset, mode='train'):
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True if mode == 'train' else False,
        collate_fn=lambda x: tuple(zip(*x)),
        num_workers=args.num_workers
    )
    return dataloader
