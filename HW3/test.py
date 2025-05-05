import os
import time
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import json
import zipfile
from tqdm import tqdm

from dataset import TestDataset
from utils import encode_mask
from model import get_model


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    test_dir = "../test_release/"
    mapping_json = "../test_image_name_to_ids.json"
    output_json = os.path.join(args.out_dir, "test-results.json")
    output_zip = os.path.join(args.out_dir, "submission.zip")

    test_set = TestDataset(
        test_dir, mapping_json, transforms=transforms.ToTensor()
    )

    loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x)),
        num_workers=1,
    )

    model_name = "maskrcnn_resnet50_fpn_v2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_name)
    model = model.to(device)

    if args.ckpt:
        if not os.path.exists(args.ckpt):
            raise ValueError(f"{args.ckpt} does not exist!")
        weights = torch.load(args.ckpt, map_location=device, weights_only=True)
        model.load_state_dict(weights)
        print(f"main: Checkpoint weights from {args.ckpt}")

    model.eval()

    results = []
    with torch.no_grad():
        pbar = tqdm(loader, desc="Testing")
        for images, image_ids in pbar:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for output, img_id in zip(outputs, image_ids):
                boxes = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
                labels = output["labels"].cpu().numpy()
                masks = output["masks"].cpu().numpy()

                for box, score, label, mask in zip(
                    boxes, scores, labels, masks
                ):
                    if score < args.score_thr:
                        continue

                    # binarize mask at 0.5 and encode as RLE
                    bin_mask = (mask[0] > 0.5).astype(np.uint8)
                    rle = encode_mask(binary_mask=bin_mask)

                    result = {
                        "image_id": int(img_id),
                        "category_id": int(label),
                        "bbox": [
                            float(box[0]),
                            float(box[1]),
                            float(box[2] - box[0]),
                            float(box[3] - box[1]),
                        ],
                        "score": float(score),
                        "segmentation": rle,
                    }
                    results.append(result)

    # save JSON
    with open(output_json, "w") as f:
        json.dump(results, f)
    print(f"Saved results to {output_json}")

    # compress into submission.zip
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(output_json, arcname="test-results.json")
    print(f"Compressed submission to {output_zip}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--score_thr", type=float, default=0.005)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="./")
    args = parser.parse_args()

    # Set GPU ID
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    start_time = time.time()
    main(args)
    end_time = time.time()
    print(f"main: Duration {end_time - start_time:.2f} seconds.")

    # python test.py --device 0 --ckpt ./best_model.pth --out_dir tmp
