"""ConvNeXt-Atto (Liu et al. 2022; the Atto size from timm): a ResNet redesigned with
transformer habits. Large 7x7 depthwise convs, LayerNorm instead of batch norm, GELU, an
inverted bottleneck, few activations. The stem is a 2x2 patchify layer (instead of 4x4),
so a 48x48 face gives stages at 24, 12, 6 and 3 pixels."""
import torch
import torch.nn.functional as F
from torch import nn

from .transformer import DropPath


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm over the channels of a (B, C, H, W) tensor."""

    def forward(self, x):
        return F.layer_norm(x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0.0, layer_scale=1e-6):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.gamma = nn.Parameter(layer_scale * torch.ones(dim))
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        h = self.dw(x).permute(0, 2, 3, 1)
        h = self.mlp(self.norm(h)) * self.gamma
        return x + self.drop_path(h.permute(0, 3, 1, 2))


class ConvNeXt(nn.Module):
    def __init__(self, classes=7, dropout=0.0, drop_path=0.1, depths=(2, 2, 6, 2), dims=(40, 80, 160, 320)):
        super().__init__()
        rates = torch.linspace(0, drop_path, sum(depths)).tolist()
        layers = [nn.Conv2d(1, dims[0], 2, 2), LayerNorm2d(dims[0])]
        k = 0
        for s, (depth, dim) in enumerate(zip(depths, dims)):
            if s > 0:
                layers += [LayerNorm2d(dims[s - 1]), nn.Conv2d(dims[s - 1], dim, 2, 2)]
            layers += [ConvNeXtBlock(dim, rates[k + j]) for j in range(depth)]
            k += depth
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.LayerNorm(dims[-1]), nn.Dropout(dropout), nn.Linear(dims[-1], classes))
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.head(self.features(x))
