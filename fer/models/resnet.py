"""ResNet-18 for small images, and the ImageNet-pretrained ResNet-18 used as a reference."""
import torch
import torch.nn.functional as F
from torch import nn

from .cnn import channels


class BasicBlock(nn.Module):
    """Two 3x3 convs whose output is added to the block's input (the skip connection)."""

    def __init__(self, i, o, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(i, o, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(o)
        self.conv2 = nn.Conv2d(o, o, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(o)
        self.skip = nn.Identity()
        if stride != 1 or i != o:
            self.skip = nn.Sequential(nn.Conv2d(i, o, 1, stride, bias=False), nn.BatchNorm2d(o))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.skip(x))


class SpatialReadout(nn.Module):
    """k learned weightings of the final grid, each pooling the features with its own
    weights: unlike global average pooling, where a feature appears counts (the faces are
    aligned). Starts close to average pooling."""

    def __init__(self, side, k=4):
        super().__init__()
        self.maps = nn.Parameter(0.1 * torch.randn(k, side * side))

    def forward(self, x):
        w = self.maps.softmax(-1)
        return torch.einsum("bcn,kn->bkc", x.flatten(2), w).flatten(1)


class ResNet18(nn.Module):
    """3x3 stem without max pooling (the CIFAR variant), so the 48x48 input is not shrunk
    right away. Stages at 48, 24, 12 and 6 pixels.

    For the second tuning stage: width scales the channels, depth is the number of blocks
    per stage (2 is ResNet-18, 1 ResNet-10, 3 ResNet-26), stages the number of stages.
    in_ch and stem_stride are for larger color input, readout for the position-aware heads
    (part two)."""

    def __init__(self, classes=7, dropout=0.0, width=1.0, depth=2, stages=4, in_ch=1, stem_stride=1, readout="gap", size=48):
        super().__init__()
        widths = [channels(64 * 2**k * width) for k in range(stages)]
        self.stem = nn.Sequential(nn.Conv2d(in_ch, widths[0], 3, stem_stride, 1, bias=False), nn.BatchNorm2d(widths[0]), nn.ReLU(inplace=True))
        stages, c = [], widths[0]
        for k, w in enumerate(widths):
            stride = 1 if k == 0 else 2
            stages.append(nn.Sequential(BasicBlock(c, w, stride), *[BasicBlock(w, w, 1) for _ in range(depth - 1)]))
            c = w
        self.stages = nn.Sequential(*stages)
        side = size // stem_stride // 2 ** (len(widths) - 1)
        if readout == "gap":
            self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(c, classes))
        elif readout == "spatial":
            self.head = nn.Sequential(SpatialReadout(side), nn.Dropout(dropout), nn.Linear(4 * c, classes))
        else:  # "flatten": a weight for every position and channel
            self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(c * side * side, classes))

    def forward(self, x):
        return self.head(self.stages(self.stem(x)))


class PretrainedResNet18(nn.Module):
    """torchvision's ResNet-18 with ImageNet weights. The grayscale face is scaled to 96x96
    and copied into three channels, since the weights expect colour photos of that scale."""

    MEAN, STD = 0.502, 0.240  # our normalisation, undone before ImageNet's is applied

    def __init__(self, classes=7, dropout=0.0):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        self.net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, classes))
        self.register_buffer("im_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("im_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = F.interpolate(x * self.STD + self.MEAN, size=96, mode="bilinear", align_corners=False)
        return self.net((x.expand(-1, 3, -1, -1) - self.im_mean) / self.im_std)
