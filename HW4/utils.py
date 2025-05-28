import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def np_to_pil(img_np):
    """
    Converts image in np.array format to PIL image.

    From C x W x H [0..1] to  W x H x C [0...255]
    :param img_np:
    :return:
    """
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        assert img_np.shape[0] == 3, img_np.shape
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)


def torch_to_np(img_var):
    """
    Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    :param img_var:
    :return:
    """
    return img_var.detach().cpu().numpy()[0]


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all statistics"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Update statistics"""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_psnr_ssim(recoverd, clean):
    assert recoverd.shape == clean.shape
    recoverd = np.clip(recoverd.detach().cpu().numpy(), 0, 1)
    clean = np.clip(clean.detach().cpu().numpy(), 0, 1)

    recoverd = recoverd.transpose(0, 2, 3, 1)
    clean = clean.transpose(0, 2, 3, 1)
    psnr = 0
    ssim = 0

    for i in range(recoverd.shape[0]):
        psnr += peak_signal_noise_ratio(clean[i], recoverd[i], data_range=1)
        ssim += structural_similarity(
            clean[i], recoverd[i], data_range=1, channel_axis=-1
        )

    return (
        psnr / recoverd.shape[0],
        ssim / recoverd.shape[0],
        recoverd.shape[0],
    )


class VGGPerceptualLoss(nn.Module):
    def __init__(self, layers=["relu1_2", "relu2_2"], use_l1=True):
        super().__init__()
        vgg = torchvision.models.vgg19(weights="DEFAULT").features.eval()
        for p in vgg.parameters():
            p.requires_grad = False

        # map layer names to indices
        layer_map = {
            "relu1_1": 0,
            "relu1_2": 2,
            "relu2_1": 5,
            "relu2_2": 7,
            "relu3_1": 10,
            "relu3_2": 12,
            "relu3_3": 14,
            "relu3_4": 16,
            "relu4_1": 19,
            "relu4_2": 21,
            "relu4_3": 23,
            "relu4_4": 25,
            "relu5_1": 28,
            "relu5_2": 30,
            "relu5_3": 32,
            "relu5_4": 34,
        }

        self.selected_idxs = [layer_map[layer] for layer in layers]
        self.vgg = vgg
        self.criterion = nn.L1Loss() if use_l1 else nn.MSELoss()
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, input, target):
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std

        feat_in, feat_tgt = input, target
        loss = 0.0
        for idx in range(max(self.selected_idxs) + 1):
            feat_in = self.vgg[idx](feat_in)
            feat_tgt = self.vgg[idx](feat_tgt)
            if idx in self.selected_idxs:
                loss += self.criterion(feat_in, feat_tgt)
        return loss
