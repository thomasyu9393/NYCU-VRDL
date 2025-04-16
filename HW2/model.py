import os
import json
import zipfile
import pandas as pd
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_mobilenet_v3_large_320_fpn,
    fasterrcnn_resnet50_fpn_v2
)
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class Model(torch.nn.Module):
    def __init__(self, model_name='fasterrcnn_resnet50_fpn_v2'):
        super(Model, self).__init__()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        if model_name == 'fasterrcnn_mobilenet_v3_large_fpn':
            self.model = fasterrcnn_mobilenet_v3_large_fpn(
                weights='COCO_V1'
            )
        elif model_name == 'fasterrcnn_mobilenet_v3_large_320_fpn':
            self.model = fasterrcnn_mobilenet_v3_large_320_fpn(
                weights='COCO_V1'
            )
        elif model_name == 'fasterrcnn_resnet50_fpn_v2':
            self.model = fasterrcnn_resnet50_fpn_v2(
                weights='COCO_V1'
            )
        self.losses_history = {
            'total_loss': [],
            'loss_classifier': [],
            'loss_box_reg': [],
            'loss_objectness': [],
            'loss_rpn_box_reg': [],
        }
        self.mAP_history = []
        self.acc_history = []
        self.num_classes = 11
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, self.num_classes
        )
        self.model = self.model.to(self.device)
        print(f'__init__: Model {model_name}')

    def load_weights(self, weights_path):
        if not os.path.exists(weights_path):
            print("load_weights: Not loading weights.")
            return False
        weights = torch.load(
            weights_path,
            map_location=self.device,
            weights_only=True
        )
        self.model.load_state_dict(weights)
        print(f"load_weights: Checkpoint weights from {weights_path}")
        return True

    def train(
            self,
            train_loader, valid_loader,
            num_epochs=10, lr=1e-4,
            ckpt_dir='ckpt', gt_file='nycu-hw2-data/valid.json'):

        os.makedirs(ckpt_dir, exist_ok=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=1e-6
        )

        best_mAP, best_epoch = 0.0, 0
        for epoch in range(num_epochs):
            avg_loss_dict = self._train_one_epoch(
                train_loader, optimizer, epoch, num_epochs
            )
            for k, v in avg_loss_dict.items():
                self.losses_history[k].append(v)
            self.plot_losses(ckpt_dir=ckpt_dir)
            scheduler.step()

            mAP, acc = self._evaluate(
                valid_loader,
                pred_file=f'{ckpt_dir}/valid_pred.json',
                gt_file=gt_file
            )
            self.mAP_history.append(mAP)
            self.acc_history.append(acc)
            print(f"train: Epoch {epoch+1}",
                  f"- Valid mAP: {mAP:.4f}, acc: {acc:.4f}")

            # Save best model
            if mAP > best_mAP:
                best_mAP, best_epoch = mAP, epoch
                torch.save(
                    self.model.state_dict(),
                    os.path.join(ckpt_dir, 'best_model.pth')
                )
                print(f'train: New best_model.pth with mAP: {best_mAP:.4f}')

            # Save last model
            torch.save(
                self.model.state_dict(),
                os.path.join(ckpt_dir, 'last_model.pth')
            )

        print(f'train: Training completed. \
               Best mAP: {best_mAP:.4f} at epoch {best_epoch+1}')
        return best_mAP

    def _train_one_epoch(
            self,
            data_loader,
            optimizer,
            epoch, num_epochs, stage=None):

        self.model.train()

        total_samples = 0
        total_batches = len(data_loader)
        epoch_loss_dict = {
            'total_loss': 0.0,
            'loss_classifier': 0.0,
            'loss_box_reg': 0.0,
            'loss_objectness': 0.0,
            'loss_rpn_box_reg': 0.0
        }

        pbar = tqdm(data_loader, desc=f'Training [{epoch+1}/{num_epochs}]')
        for images, targets in pbar:
            batch_size = len(images)
            total_samples += batch_size
            images = list(image.to(self.device) for image in images)
            targets = [
                {k: v.to(self.device) for k, v in t.items()}
                for t in targets
            ]

            loss_dict = self.model(images, targets)
            if stage == 'rpn':
                losses = loss_dict.get('loss_objectness', 0) + \
                    loss_dict.get('loss_rpn_box_reg', 0)
            elif stage == 'roi':
                losses = loss_dict.get('loss_classifier', 0) + \
                    loss_dict.get('loss_box_reg', 0)
            else:
                losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            # Accumulate average losses per sample
            epoch_loss_dict['total_loss'] += losses.item() / batch_size
            for k in loss_dict:
                epoch_loss_dict[k] += loss_dict[k].item() / batch_size

        avg_loss_dict = {
            k: v / total_batches
            for k, v in epoch_loss_dict.items()
        }
        return avg_loss_dict

    def plot_losses(self, ckpt_dir):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(self.losses_history['total_loss'], label='total_loss')
        # for k, v in self.losses_history.items():
        #     plt.plot(v, label=k)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(ckpt_dir, 'losses.png'))
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(self.mAP_history, label='mAP')
        plt.plot(self.acc_history, label='acc')
        plt.xlabel("Epoch")
        plt.ylabel("mAP/acc")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(ckpt_dir, 'mAP_acc.png'))
        plt.close()

    def _evaluate(self,
                  valid_loader,
                  iou_type='bbox', threshold=0.5,
                  pred_file='valid_pred.json',
                  gt_file='nycu-hw2-data/valid.json'):
        self.model.eval()
        results = []
        pred_labels = []

        with torch.no_grad():
            pbar = tqdm(valid_loader, desc='Evaluating')
            for images, targets in pbar:
                images = list(img.to(self.device) for img in images)
                outputs = self.model(images)

                for i, output in enumerate(outputs):
                    image_id = targets[i]['image_id'].item()
                    boxes = output['boxes'].cpu().numpy()
                    scores = output['scores'].cpu().numpy()
                    labels = output['labels'].cpu().numpy()

                    single_results = []
                    for box, score, label in zip(boxes, scores, labels):
                        if label == 0 or score < threshold:
                            continue

                        x1, y1, x2, y2 = box.tolist()
                        width = x2 - x1
                        height = y2 - y1

                        pred_dict = {
                            "image_id": image_id,
                            "category_id": int(label),
                            "bbox": [x1, y1, width, height],
                            "score": float(score)
                        }
                        results.append(pred_dict)
                        single_results.append(pred_dict)

                    # Sort by x-coordinate to get digits from left to right
                    single_results = sorted(
                        single_results, key=lambda x: x["bbox"][0]
                    )
                    if single_results:
                        pred_digits = [
                            str(p["category_id"] - 1)
                            for p in single_results
                        ]
                        pred_number = ''.join(pred_digits)
                    else:
                        pred_number = "-1"

                    pred_labels.append((image_id, pred_number))

        csv_out = pred_file.replace('.json', '.csv')
        df = pd.DataFrame(pred_labels, columns=['image_id', 'pred_label'])
        df.to_csv(csv_out, index=False)
        acc = self._compute_acc(csv_out, 'valid.csv')

        # Save predictions
        with open(pred_file, 'w') as f:
            json.dump(results, f)

        # Load ground truth and predictions for COCO evaluation
        coco_gt = COCO(gt_file)
        coco_dt = coco_gt.loadRes(pred_file)

        coco_eval = COCOeval(coco_gt, coco_dt, iouType=iou_type)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Return the mAP @ IoU=0.5:0.95 (the standard COCO metric)
        return coco_eval.stats[0], acc  # stats[0] = mAP

    def _compute_acc(self, pred_file, gt_file):
        pred_df = pd.read_csv(pred_file)
        valid_df = pd.read_csv(gt_file)

        # Merge on image_id
        merged = pd.merge(
            valid_df, pred_df,
            on='image_id',
            suffixes=('_true', '_pred')
        )

        # Compare labels
        cnt = (merged['pred_label_true'] == merged['pred_label_pred']).sum()
        total = len(merged)

        accuracy = cnt / total if total > 0 else 0.0
        return accuracy

    def test(
            self,
            test_loader,
            score_thr=0.5,
            zip_out='submission.zip'):

        self.model.eval()
        results = []
        pred_labels = []

        with torch.no_grad():
            for images, targets in tqdm(test_loader, desc='Testing'):
                images = list(img.to(self.device) for img in images)
                outputs = self.model(images)

                for i, output in enumerate(outputs):
                    image_id = targets[i]['image_id'].item()
                    boxes = output['boxes'].cpu().numpy()
                    scores = output['scores'].cpu().numpy()
                    labels = output['labels'].cpu().numpy()

                    single_results = []
                    for box, score, label in zip(boxes, scores, labels):
                        if label == 0 or score < score_thr:
                            continue

                        # [x_min, y_min, x_max, y_max] to [x_min, y_min, w, h]
                        x_min, y_min, x_max, y_max = box.tolist()
                        w, h = x_max - x_min, y_max - y_min
                        pred_dict = {
                            "image_id": image_id,
                            "bbox": [x_min, y_min, w, h],
                            "score": float(score),
                            "category_id": int(label)
                        }
                        results.append(pred_dict)
                        single_results.append(pred_dict)

                    # Sort by x-coordinate to get digits from left to right
                    single_results = sorted(
                        single_results,
                        key=lambda x: x["bbox"][0]
                    )
                    if single_results:
                        pred_digits = [
                            str(p["category_id"] - 1)
                            for p in single_results
                        ]
                        pred_number = ''.join(pred_digits)
                    else:
                        pred_number = "-1"

                    pred_labels.append((image_id, pred_number))

        # Save predictions
        json_out = zip_out.replace('.zip', '.json')
        with open(json_out, 'w') as f:
            json.dump(results, f)

        csv_out = zip_out.replace('.zip', '.csv')
        df = pd.DataFrame(pred_labels, columns=['image_id', 'pred_label'])
        df.to_csv(csv_out, index=False)

        with zipfile.ZipFile(zip_out, 'w') as zipf:
            zipf.write(json_out, arcname='pred.json')
            zipf.write(csv_out, arcname='pred.csv')

        print(f"test: Processed {len(pred_labels)} images.")
        print(f"test: Generated {json_out} and {csv_out} into {zip_out}")
        return True
