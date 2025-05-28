import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pytorch_msssim import SSIM

from dataset import RainSnowDataset
from model import PromptIR
from schedulers import LinearWarmupCosineAnnealingLR
from utils import (
    AverageMeter, compute_psnr_ssim,
    torch_to_np, np_to_pil,
    VGGPerceptualLoss,
)


def main(args):
    print(args)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.tmp_dir, exist_ok=True)

    valid_id = [
        1, 2, 5, 6, 9,
        10, 57, 62, 67, 72,
        77, 1349, 1359, 1369, 1376,
        1378, 1460, 1461, 1599, 1600
    ]
    print(f"main: len(valid_id): {len(valid_id)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PromptIR(decoder=True).to(device)
    print(f"main: device: {device}")

    trainset = RainSnowDataset(
        train_val="train",
        skipped_id=valid_id,
        patch_size=args.patch_size,
    )
    valset = RainSnowDataset(
        train_val="val",
        skipped_id=valid_id,
        patch_size=None,
        augment=False,
    )

    trainloader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valloader = DataLoader(
        valset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer=optimizer,
        warmup_epochs=15,
        max_epochs=args.epochs,
        eta_min=1e-6,
    )

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=num_epochs, eta_min=1e-6
    # )

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer,
    #     T_0=15, T_mult=2,
    #     eta_min=1e-6
    # )

    writer = SummaryWriter(log_dir="runs/exp")
    writer.add_text(
        "Parameters", f"Learning Rate: {args.lr}, Epochs: {args.epochs}"
    )

    l1_loss = nn.L1Loss()
    print("Loss: Use L1")

    perceptual_loss = None
    w_perc = 0.0
    if args.perceptual:
        layers = ["relu1_2", "relu2_2", "relu3_3"]
        w_perc = 0.005
        print(f"Loss: Use perceptual layers {layers} weight {w_perc}")
        perceptual_loss = VGGPerceptualLoss(layers=layers).to(device)

    ssim_loss = None
    w_ssim = 0.0
    if args.ssim:
        ssim_loss = SSIM(
            data_range=1.0, size_average=True
        )  # returns a similarity
        w_ssim = 0.1
        print(f"Loss: Use SSIM weight {w_ssim}")

    best_PSNR, best_epoch = 0.0, 0
    for epoch in range(args.epochs):
        # ===== Training =====
        model.train()
        running_loss = 0.0
        image_counter = 0

        for batch_idx, (degrad_patch, clean_patch) in enumerate(
            tqdm(trainloader, desc=f"Training [{epoch+1}/{args.epochs}]")
        ):
            degrad_patch = degrad_patch.to(device)
            clean_patch = clean_patch.to(device)

            # Forward
            restored = model(degrad_patch)
            losses_l1 = l1_loss(restored, clean_patch)
            if perceptual_loss:
                losses_perc = perceptual_loss(restored, clean_patch)
            else:
                losses_perc = torch.tensor(0.0, device=device)
            if ssim_loss:
                losses_ssim = 1 - ssim_loss(restored, clean_patch)
            else:
                losses_ssim = torch.tensor(0.0, device=device)

            losses = losses_l1 + w_perc * losses_perc + w_ssim * losses_ssim

            # Backward
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            batch_size = clean_patch.size(0)
            running_loss += losses.item() * batch_size
            image_counter += batch_size

            # Log per-step
            writer.add_scalar(
                "TrainingLoss/step",
                losses.item(),
                epoch * len(trainloader) + batch_idx,
            )
            writer.add_scalar(
                "TrainingLoss/l1",
                losses_l1.item(),
                epoch * len(trainloader) + batch_idx,
            )
            writer.add_scalar(
                "TrainingLoss/perceptual",
                losses_perc.item(),
                epoch * len(trainloader) + batch_idx,
            )
            writer.add_scalar(
                "TrainingLoss/ssim",
                losses_ssim.item(),
                epoch * len(trainloader) + batch_idx,
            )

        scheduler.step()
        epoch_loss = running_loss / image_counter
        writer.add_scalar("TrainingLoss/avg", epoch_loss, epoch + 1)
        print(f"> Epoch {epoch+1} - Train Loss {epoch_loss:.4f}")

        # ===== Validation =====
        model.eval()
        epoch_output_dir = os.path.join(args.tmp_dir, f"epoch_{epoch+1:02d}")
        os.makedirs(epoch_output_dir, exist_ok=True)

        psnr = AverageMeter()
        ssim = AverageMeter()
        running_loss = 0.0
        image_counter = 0
        with torch.no_grad():
            for idx, (degrad_patch, clean_patch) in enumerate(
                tqdm(valloader, desc="Evaluating")
            ):
                degrad_patch = degrad_patch.to(device)
                clean_patch = clean_patch.to(device)

                restored = model(degrad_patch)
                temp_psnr, temp_ssim, N = compute_psnr_ssim(
                    restored, clean_patch
                )
                psnr.update(temp_psnr, N)
                ssim.update(temp_ssim, N)

                losses_l1 = l1_loss(restored, clean_patch)
                if perceptual_loss:
                    losses_perc = perceptual_loss(restored, clean_patch)
                else:
                    losses_perc = torch.tensor(0.0, device=device)
                if ssim_loss:
                    losses_ssim = 1 - ssim_loss(restored, clean_patch)
                else:
                    losses_ssim = torch.tensor(0.0, device=device)

                losses = (
                    losses_l1 + w_perc * losses_perc + w_ssim * losses_ssim
                )
                batch_size = clean_patch.size(0)
                running_loss += losses.item() * batch_size
                image_counter += batch_size

                out_path = os.path.join(
                    epoch_output_dir, f"epoch_{epoch+1:02d}_{idx:02d}.png"
                )
                restored = restored.cpu().clamp(0, 1)
                restored = torch_to_np(restored)
                restored = np_to_pil(restored)
                restored.save(out_path)

            psnr_avg = float(psnr.avg)
            ssim_avg = float(ssim.avg)
            epoch_loss = running_loss / image_counter
            print(
                f"> Epoch {epoch+1} - Val Loss {epoch_loss:.4f} - PSNR: {psnr_avg:.2f}, SSIM: {ssim_avg:.4f}"
            )

        writer.add_scalar("Validation/Loss", epoch_loss, epoch + 1)
        writer.add_scalar("Validation/PSNR", psnr_avg, epoch + 1)
        writer.add_scalar("Validation/SSIM", ssim_avg, epoch + 1)

        # Save a checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save(
                model.state_dict(),
                f"{args.ckpt_dir}/model_epoch_{epoch+1}.pth",
            )
        torch.save(model.state_dict(), f"{args.ckpt_dir}/last_model.pth")
        if psnr_avg > best_PSNR:
            best_PSNR = psnr_avg
            best_epoch = epoch
            torch.save(model.state_dict(), f"{args.ckpt_dir}/best_model.pth")

    print(f"main: best PSNR: {best_PSNR:.4f} at epoch {best_epoch+1}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--ckpt_dir", type=str, default="ckpt")
    parser.add_argument("--tmp_dir", type=str, default="tmp")
    parser.add_argument("--perceptual", action="store_true")
    parser.add_argument("--ssim", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    main(args)
