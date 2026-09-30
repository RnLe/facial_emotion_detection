"""The attention models: a plain vision transformer (ViT-Lite-7/4) and the compact
convolutional transformer CCT-7/3x2 (Hassani et al. 2021), which makes its tokens with two
small conv layers instead of cutting the image into patches."""
import torch
from torch import nn


class DropPath(nn.Module):
    """Stochastic depth: during training, skip a residual branch for random samples."""

    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        keep = torch.rand(x.shape[0], *([1] * (x.dim() - 1)), device=x.device) >= self.p
        return x * keep / (1 - self.p)


class Block(nn.Module):
    """Pre-norm transformer layer: attention, then a two-layer MLP, each with a skip."""

    def __init__(self, dim, heads, mlp_ratio=2, dropout=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * mlp_ratio, dim))
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.drop_path(self.attn(h, h, h, need_weights=False)[0])
        return x + self.drop_path(self.mlp(self.norm2(x)))


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class Encoder(nn.Module):
    def __init__(self, dim, depth, heads, mlp_ratio, dropout, drop_path):
        super().__init__()
        rates = torch.linspace(0, drop_path, depth).tolist()  # deeper layers are skipped more often
        self.blocks = nn.Sequential(*[Block(dim, heads, mlp_ratio, dropout, r) for r in rates])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(self.blocks(x))


class ViT(nn.Module):
    """4x4 patches (144 tokens for a 48x48 face), a class token, learned positions."""

    def __init__(self, classes=7, dropout=0.0, drop_path=0.1, dim=256, depth=7, heads=4, mlp_ratio=2, patch=4):
        super().__init__()
        tokens = (48 // patch) ** 2
        self.patch = nn.Conv2d(1, dim, patch, patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, tokens + 1, dim))
        self.encoder = Encoder(dim, depth, heads, mlp_ratio, dropout, drop_path)
        self.head = nn.Linear(dim, classes)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.apply(init_weights)

    def forward(self, x):
        x = self.patch(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], 1) + self.pos
        return self.head(self.encoder(x)[:, 0])


class CCT(nn.Module):
    """Two conv + max-pool layers turn the face into 12x12 = 144 tokens; seven transformer
    layers follow; a learned weighted average over the tokens (sequence pooling) replaces
    the class token."""

    def __init__(self, classes=7, dropout=0.0, drop_path=0.1, dim=256, depth=7, heads=4, mlp_ratio=2):
        super().__init__()
        self.tokenizer = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1, bias=False), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(64, dim, 3, padding=1, bias=False), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2, 1),
        )
        self.pos = nn.Parameter(torch.zeros(1, 144, dim))
        self.encoder = Encoder(dim, depth, heads, mlp_ratio, dropout, drop_path)
        self.pool = nn.Linear(dim, 1)
        self.head = nn.Linear(dim, classes)
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.apply(init_weights)

    def forward(self, x):
        x = self.tokenizer(x).flatten(2).transpose(1, 2) + self.pos
        x = self.encoder(x)
        weights = self.pool(x).softmax(1)
        return self.head((weights * x).sum(1))
