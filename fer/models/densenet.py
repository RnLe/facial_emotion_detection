"""DenseNet-BC for small images (depth 100, growth rate 12), as in Huang et al. 2017.

Memory-efficient version (Pleiss et al. 2017, and torchvision's `memory_efficient`): the
concatenation and the 1x1 bottleneck are recomputed in the backward pass instead of
stored. The network is the same; without this, the concatenations at full 48x48
resolution filled the whole 12 GB GPU and an epoch took ten times as long as ResNet-18's."""
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


class DenseLayer(nn.Module):
    """BN-ReLU-1x1 conv (bottleneck to 4k channels), BN-ReLU-3x3 conv to k new channels,
    computed from everything the block has produced so far."""

    def __init__(self, c, k):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(c)
        self.conv1 = nn.Conv2d(c, 4 * k, 1, bias=False)
        self.norm2 = nn.BatchNorm2d(4 * k)
        self.conv2 = nn.Conv2d(4 * k, k, 3, padding=1, bias=False)

    def bottleneck(self, *features):
        return self.conv1(F.relu(self.norm1(torch.cat(features, 1))))

    def forward(self, features):
        if self.training:
            h = checkpoint(self.bottleneck, *features, use_reentrant=False)
        else:
            h = self.bottleneck(*features)
        return self.conv2(F.relu(self.norm2(h)))


class DenseBlock(nn.Module):
    def __init__(self, n, c, k):
        super().__init__()
        self.layers = nn.ModuleList(DenseLayer(c + i * k, k) for i in range(n))

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            features.append(layer(features))
        return torch.cat(features, 1)


def transition(c, o):
    """Compress the channels (1x1 conv) and halve the resolution between dense blocks."""
    return nn.Sequential(nn.BatchNorm2d(c), nn.ReLU(inplace=True), nn.Conv2d(c, o, 1, bias=False), nn.AvgPool2d(2))


class DenseNetBC(nn.Module):
    def __init__(self, classes=7, dropout=0.0, depth=100, growth=12, compression=0.5, stem_stride=2):
        super().__init__()
        n = (depth - 4) // 6  # bottleneck layers per block: 16 for depth 100
        c = 2 * growth
        # The CIFAR version keeps full resolution (32 px) in its first block. At 48 px that
        # block alone made an epoch six times slower than ResNet-18's: DenseNet's time goes
        # into copying and re-normalising the growing stack of features, not into
        # convolutions. One stride-2 stem conv (the ImageNet version even downsamples 4x)
        # puts the blocks at 24, 12 and 6 px.
        layers = [nn.Conv2d(1, c, 3, stride=stem_stride, padding=1, bias=False)]
        for block in range(3):
            layers.append(DenseBlock(n, c, growth))
            c += n * growth
            if block < 2:
                o = int(c * compression)
                layers.append(transition(c, o))
                c = o
        self.features = nn.Sequential(*layers, nn.BatchNorm2d(c), nn.ReLU(inplace=True))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(c, classes))

    def forward(self, x):
        return self.head(self.features(x))
