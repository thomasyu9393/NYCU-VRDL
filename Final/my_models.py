import torch
from torch import nn
from torch.nn import functional as F
import torchvision

from unet_models import UNetModule


class ResNet50_UNet(nn.Module):
    module = UNetModule
    def __init__(self, n_classes):
        super().__init__()
        resnet = torchvision.models.resnet50(
            weights='DEFAULT'
        )

        # 1) Capture the conv1→bn1→relu output (128×128) as e0_pre
        self.conv1 = resnet.conv1   # stride=2 → 128×128
        self.bn1   = resnet.bn1
        self.relu  = resnet.relu

        # 2) Then maxpool (→64×64) + layer1 (→256×64×64) is e1
        self.maxpool = resnet.maxpool
        self.layer1  = resnet.layer1  # 256×64×64
        self.layer2  = resnet.layer2  # 512×32×32
        self.layer3  = resnet.layer3  # 1024×16×16
        self.layer4  = resnet.layer4  # 2048×8×8

        # Bridge to reduce channels from 2048→1024
        self.bridge = nn.Conv2d(2048, 1024, kernel_size=1)

        # Decoder stage 1:  1024→ up(2) → (1024×16×16), concat with layer3 (1024×16×16)
        self.up1 = nn.ConvTranspose2d(1024, 1024, kernel_size=2, stride=2)
        self.dec1 = self.module(1024 + 1024, 512)

        # Decoder stage 2:  512→ up(2) → (512×32×32), concat with layer2 (512×32×32)
        self.up2 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec2 = self.module(512 + 512, 256)

        # Decoder stage 3:  256→ up(2) → (256×64×64), concat with layer1 (256×64×64)
        self.up3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.dec3 = self.module(256 + 256, 128)

        # Decoder stage 4:  128→ up(2) → (128×128×128), concat with e0_pre (64×128×128)
        self.up4 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec4 = self.module(128 + 64, 64)

        # Final 1×1 to produce `n_classes` density‐channels
        self.out_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        e0_pre = self.relu(self.bn1(self.conv1(x)))  # → (64, 128, 128)
        e1 = self.maxpool(e0_pre)                    # → (64,  64,  64)
        e1 = self.layer1(e1)                         # → (256, 64,  64)
        e2 = self.layer2(e1)                         # → (512, 32,  32)
        e3 = self.layer3(e2)                         # → (1024,16, 16)
        e4 = self.layer4(e3)                         # → (2048, 8,  8)

        # Bridge
        b = self.bridge(e4)                          # → (1024, 8,  8)

        # Decoder stage 1
        d1 = self.up1(b)                             # → (1024,16,16)
        d1 = torch.cat([d1, e3], dim=1)              # → (2048,16,16)
        d1 = self.dec1(d1)                           # → (512, 16,16)

        # Decoder stage 2
        d2 = self.up2(d1)                            # → (512, 32,32)
        d2 = torch.cat([d2, e2], dim=1)              # → (1024,32,32)
        d2 = self.dec2(d2)                           # → (256, 32,32)

        # Decoder stage 3
        d3 = self.up3(d2)                            # → (256, 64,64)
        d3 = torch.cat([d3, e1], dim=1)              # → (512, 64,64)
        d3 = self.dec3(d3)                           # → (128, 64,64)

        # Decoder stage 4
        d4 = self.up4(d3)                            # → (128,128,128)
        d4 = torch.cat([d4, e0_pre], dim=1)          # → (128+64=192, 128,128)
        d4 = self.dec4(d4)                           # → (64, 128,128)

        out = self.out_conv(d4)  # → (n_classes, 128,128)
        out_full = F.interpolate(
            out,
            size=(x.shape[2], x.shape[3]),  # i.e. (256, 256)
            mode='bilinear', align_corners=False
        )
        return F.log_softmax(out_full, dim=1)  # Now out_full is (batch, n_classes, 256, 256)
