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


def conv_bn(i, o):
    return [nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True)]


class VGG(nn.Module):
    """Four stages of two 3x3 convs with batch norm, 64 to 512 channels. 48 -> 3 pixels."""

    def __init__(self, classes=7, dropout=0.5, widths=(64, 128, 256, 512)):
        super().__init__()
        layers, c = [], 1
        for w in widths:
            layers += conv_bn(c, w) + conv_bn(w, w) + [nn.MaxPool2d(2)]
            c = w
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c * 3 * 3, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, classes),
        )

    def forward(self, x):
        return self.head(self.features(x))
