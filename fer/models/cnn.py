"""The two plain CNNs: the simplest useful one, and a deeper VGG-style one."""
from torch import nn


class SimpleCNN(nn.Module):
    """Three conv + pool stages and a small dense head. 48 -> 24 -> 12 -> 6 pixels."""

    def __init__(self, classes=7, dropout=0.5):
        super().__init__()
        layers, c = [], 1
        for w in (32, 64, 128):
            layers += [nn.Conv2d(c, w, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            c = w
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 6 * 6, 256), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(256, classes)
        )

    def forward(self, x):
        return self.head(self.features(x))


def conv_bn(i, o, norm=True):
    if not norm:
        return [nn.Conv2d(i, o, 3, padding=1), nn.ReLU(inplace=True)]
    return [nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True)]


def channels(c):
    """Round a channel count to a multiple of 8 (what the GPU's matrix units work in)."""
    return max(8, round(c / 8) * 8)


class VGG(nn.Module):
    """Four stages of two 3x3 convs with batch norm, 64 to 512 channels. 48 -> 3 pixels.

    The shape can be changed for the second tuning stage: width scales every layer's
    channels (and the dense head), depth is the number of convs per stage, stages the
    number of conv + pool stages. norm=False drops batch norm (convs get a bias instead),
    for the grokking test, where the size of the weights has to matter."""

    def __init__(self, classes=7, dropout=0.5, width=1.0, depth=2, stages=4, norm=True):
        super().__init__()
        layers, c = [], 1
        for k in range(stages):
            w = channels(64 * 2**k * width)
            for _ in range(depth):
                layers += conv_bn(c, w, norm)
                c = w
            layers.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*layers)
        side, hidden = 48 >> stages, channels(512 * width)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c * side * side, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, classes),
        )

    def forward(self, x):
        return self.head(self.features(x))
