import os
import json
import cv2
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from dataset import TrainDataset
import utils
from model import get_model

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def main():
    # ===== Dataset =====
    dataset = TrainDataset(
        image_dir="./train", transforms=transforms.ToTensor()
    )
    num_images = len(dataset)
    print(f"main: num_images: {num_images}")

    indices = list(range(num_images))
    train_ratio = 0.9
    split = int(train_ratio * num_images)

    random.seed(1234)
    random.shuffle(indices)

    train_idx, val_idx = indices[:split], indices[split:]
    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set = torch.utils.data.Subset(dataset, val_idx)

    # ===== DataLoader =====
    train_loader = DataLoader(
        train_set,
        batch_size=4,
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x)),
        num_workers=1,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=4,
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x)),
        num_workers=1,
    )

    # ===== Validation GT =====
    coco_gt = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "class1"},
            {"id": 2, "name": "class2"},
            {"id": 3, "name": "class3"},
            {"id": 4, "name": "class4"},
        ],
    }
    ann_id = 1
    for idx in val_idx:
        image_id = int(idx)
        img_path = os.path.join("./train", dataset.ids[idx], "image.tif")
        image = cv2.imread(img_path)  # BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # RGB
        h, w, _ = image.shape

        coco_gt["images"].append(
            {
                "id": image_id,
                "width": w,
                "height": h,
                "file_name": dataset.ids[idx],
            }
        )

        _, target = dataset[idx]
        for box, label, mask in zip(
            target["boxes"], target["labels"], target["masks"]
        ):
            rle = utils.encode_mask(binary_mask=mask)
            coco_gt["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": [*box.tolist()],
                    "area": float((box[2] - box[0]) * (box[3] - box[1])),
                    "segmentation": rle,
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    with open("val_gt.json", "w") as f:
        json.dump(coco_gt, f)
    print(f"main: Saved valid ann count = {ann_id} to val_gt.json")

    # ===== Model =====
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    writer = SummaryWriter(log_dir="runs/0505")

    model_name = "maskrcnn_resnet50_fpn_v2"
    model = get_model(model_name)

    model_size = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"main: model {model_name} size {model_size:.2f}M parameters")

    model = model.to(device)
    print(f"main: device: {device}")

    lr = 1e-4
    num_epochs = 20
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )
    writer.add_text(
        "Parameters", f"Learning Rate: {lr}, " f"Epochs: {num_epochs}"
    )

    print("main: Starting training...")

    # ===== Training Loop =====
    mask_thr = 0.5
    best_mAP, best_epoch = 0.0, 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Training [{epoch+1}/{num_epochs}]")
        for step, (images, targets) in enumerate(pbar):
            images = [img.to(device) for img in images]
            targets = [
                {k: v.to(device) for k, v in t.items()} for t in targets
            ]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            running_loss += losses.item()

            # log per-step
            writer.add_scalar(
                "TrainingLoss/total",
                losses.item(),
                epoch * len(train_loader) + step,
            )
            for k, v in loss_dict.items():
                writer.add_scalar(
                    f"TrainingLoss/{k}",
                    v.item(),
                    epoch * len(train_loader) + step,
                )

            # log LR
            writer.add_scalar(
                "LearningRate",
                optimizer.param_groups[0]["lr"],
                epoch * len(train_loader) + step,
            )

        scheduler.step()
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        writer.add_scalar("TrainingLoss/avg", epoch_loss, epoch)

        # ===== Evaulation =====
        model.eval()

        torch.save(
            model.state_dict(), os.path.join("./", f"epoch_{epoch+1}.pth")
        )

        results = []
        pbar = tqdm(val_loader, desc="Evaluating")
        for images, targets in pbar:
            images = [img.to(device) for img in images]

            with torch.no_grad():
                outputs = model(images)

            for output, target in zip(outputs, targets):
                idx = target["image_id"].item()

                for box, mask, score, label in zip(
                    output["boxes"],
                    output["masks"],
                    output["scores"],
                    output["labels"],
                ):
                    mask = (mask[0] > mask_thr).cpu().numpy().astype(np.uint8)
                    rle = utils.encode_mask(binary_mask=mask)

                    results.append(
                        {
                            "image_id": idx,
                            "category_id": int(label),
                            "bbox": [*box.cpu().numpy().tolist()],
                            "score": float(score),
                            "segmentation": rle,
                        }
                    )

        with open("val_results.json", "w") as f:
            json.dump(results, f)

        coco = COCO("val_gt.json")
        coco_dets = coco.loadRes("val_results.json")

        coco_eval = COCOeval(coco, coco_dets, iouType="segm")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        mAP = coco_eval.stats[0]
        writer.add_scalar("Val/mAP", mAP, epoch)
        if mAP > best_mAP:
            best_mAP, best_epoch = mAP, epoch
            torch.save(
                model.state_dict(), os.path.join("./", "best_model.pth")
            )
            print(f"main: New best epoch {epoch+1} with mAP: {best_mAP:.4f}")

    writer.close()
    print(
        f"main: End training with best mAP: {best_mAP:.4f} at epoch {best_epoch+1}"
    )


if __name__ == "__main__":
    main()
    print("end of main")
