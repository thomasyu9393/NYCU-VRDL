import os
import json
import zipfile
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm
from model import Model
from ensemble_boxes import weighted_boxes_fusion


def test_ensemble(
    args,
    model_name,
    test_loader,
    model_ckpts,
    aggregation='wbf',
    iou_thr=0.55,
    score_thr=0.6,
    zip_out='submission.zip'
):
    models = []
    for i, model_ckpt in enumerate(model_ckpts):
        model = Model(model_name=model_name)
        model.load_weights(model_ckpt)
        model.model.eval()
        models.append(model)

    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts [0,255] -> [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    results = []
    pred_labels = []

    test_path = 'nycu-hw2-data/test'
    for image_name in tqdm(os.listdir(test_path)):
        image_path = os.path.join(test_path, image_name)
        image = Image.open(image_path).convert("RGB")
        img_w, img_h = image.size  # get image dimensions
        image_id = int(image_name.split('.')[0])

        image_tensor = transform(image)
        image_tensor = image_tensor.to(models[0].device)

        all_boxes = []
        all_scores = []
        all_labels = []

        with torch.no_grad():
            for model in models:
                output = model.model([image_tensor])[0]
                output = {k: v.cpu().numpy() for k, v in output.items()}

                boxes = output['boxes']  # [x_min, y_min, x_max, y_max]
                scores = output['scores']
                labels = output['labels']

                # Normalize boxes to [0, 1] if necessary
                boxes_norm = boxes.copy()
                boxes_norm[:, [0, 2]] /= img_w  # normalize x coordinates
                boxes_norm[:, [1, 3]] /= img_h  # normalize y coordinates

                all_boxes.append(boxes_norm.tolist())
                all_scores.append(scores.tolist())
                all_labels.append(labels.tolist())

        # Apply Weighted Boxes Fusion.
        agg_boxes, agg_scores, agg_labels = weighted_boxes_fusion(
            all_boxes, all_scores, all_labels,
            iou_thr=iou_thr, skip_box_thr=score_thr, weights=None
        )
        agg_boxes = [
            [box[0] * img_w, box[1] * img_h, box[2] * img_w, box[3] * img_h]
            for box in agg_boxes
        ]

        single_results = []
        for box, score, label in zip(agg_boxes, agg_scores, agg_labels):
            if label == 0:
                continue

            x_min, y_min, x_max, y_max = box
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
        single_results = sorted(single_results, key=lambda x: x["bbox"][0])
        if single_results:
            pred_digits = [str(p["category_id"] - 1) for p in single_results]
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

    print(f"ensemble_test: Processed {len(pred_labels)} images.",
          f"Detected {len(results)} total boxes.")
    print(f"ensemble_test: Generated {json_out} and {csv_out} into {zip_out}")
