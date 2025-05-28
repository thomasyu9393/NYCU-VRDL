import os
import argparse
import numpy as np
from tqdm import tqdm
import zipfile

import torch
from torchvision import transforms
from PIL import Image

from model import PromptIR
from utils import torch_to_np, np_to_pil


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PromptIR(decoder=True).to(device)

    if os.path.exists(args.ckpt):
        checkpoint = torch.load(
            args.ckpt, map_location=device, weights_only=False
        )
    else:
        raise ValueError(f"ckpt {args.ckpt} not exists")
    model.load_state_dict(checkpoint)
    print(f"main: Loaded checkpoint {args.ckpt}")
    model.eval()

    paths = [os.path.join(args.test_dir, f"{i}.png") for i in range(100)]
    paths = [p for p in paths if os.path.isfile(p)]
    print(f"main: len(paths): {len(paths)}")

    transform = transforms.ToTensor()
    with torch.no_grad():
        for path in tqdm(paths, desc="Testing"):
            img = Image.open(path).convert("RGB")
            img_tensor = (
                transform(img).unsqueeze(0).to(device)
            )  # shape [1, 3, H, W]
            restored = model(img_tensor)

            fname = os.path.basename(path)
            out_path = os.path.join(args.output_dir, fname)

            restored = restored.cpu().clamp(0, 1)
            restored = torch_to_np(restored)
            restored = np_to_pil(restored)
            restored.save(out_path)

    print(f"Done! Restored images saved to {args.output_dir}")

    output_npz = os.path.join(args.output_dir, "pred.npz")
    output_zip = os.path.join(args.output_dir, "submission.zip")

    # Initialize dictionary to hold image arrays
    images_dict = {}

    # Loop through all files in the folder
    for filename in os.listdir(args.output_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            file_path = os.path.join(args.output_dir, filename)

            # Load image and convert to RGB
            image = Image.open(file_path).convert("RGB")
            img_array = np.array(image)

            # Rearrange to (3, H, W)
            img_array = np.transpose(img_array, (2, 0, 1))

            # Add to dictionary
            images_dict[filename] = img_array

    # Save to .npz file
    np.savez(output_npz, **images_dict)
    print(f"Saved {len(images_dict)} images to {output_npz}")

    # Compress into submission.zip
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(output_npz, arcname="pred.npz")
    print(f"Compressed submission to {output_zip}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument(
        "--test_dir", type=str, default="hw4_realse_dataset/test/degraded"
    )
    parser.add_argument("--output_dir", type=str, default="output")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    main(args)
    # python test.py --ckpt ckpt/model_epoch_3.pth --device 1
