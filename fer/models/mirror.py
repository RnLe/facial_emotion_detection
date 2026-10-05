"""ResNet-18 that cannot tell a face from its mirror image (part two, phase 5).

Our winners change their prediction on 8 to 9% of the faces when the face is mirrored,
although they trained with random flips; FMAE on 3%. Here the invariance is built in:
every feature comes as a pair (a, b) that swaps, and is mirrored, when the input is
mirrored. A filter and its mirror image share their weights (a group convolution over
the horizontal flip; Cohen and Welling 2016 do this for rotations and flips together).

- Batch norm shares its statistics and scale within a pair.
- Downsampling is 2x2 average pooling before the block's first conv: a stride-2 3x3 conv
  samples the even columns, which a mirror maps onto the odd ones.
- The head pools each pair into a + b and |a - b|. Both are unchanged by mirroring; the
  second keeps how asymmetric a feature is, without its side.

match="params" keeps ResNet-18's parameter count (64/sqrt(2) pairs where it has 64
channels, about twice its compute); match="compute" keeps its compute (32 pairs, half
the parameters).
"""
import math

import torch
import torch.nn.functional as F
from torch import nn


def pair_count(c, match):
    n = c / math.sqrt(2) if match == "params" else c / 2
    return max(4, round(n / 4) * 4)


class MirrorConv(nn.Module):
    """i pairs in (or i plain channels when lift), o pairs out. Input and output lay out
    all a halves first, then all b halves."""

    def __init__(self, i, o, k=3, lift=False):
        super().__init__()
        self.lift = lift
        self.weight = nn.Parameter(torch.empty(o, i if lift else 2 * i, k, k))
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        self.padding = k // 2

    def full_weight(self):
        w = self.weight
        if self.lift:
            return torch.cat([w, w.flip(-1)], 0)
        half = w.shape[1] // 2
        # b = mirrored filter, applied with the input halves swapped
        return torch.cat([w, torch.cat([w[:, half:], w[:, :half]], 1).flip(-1)], 0)

    def forward(self, x):
        return F.conv2d(x, self.full_weight(), padding=self.padding)


class MirrorBN(nn.Module):
    """Batch norm over pairs: both halves share statistics and scale. The pooled statistics
    come from the per-channel ones (the variance of a union is the mean of the variances
    plus the variance of the means), so the tensor is never reshaped."""

    def __init__(self, c, momentum=0.1, eps=1e-5):
        super().__init__()
        self.c, self.momentum, self.eps = c, momentum, eps
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))
        self.register_buffer("running_mean", torch.zeros(c))
        self.register_buffer("running_var", torch.ones(c))

    def forward(self, x):
        c = self.c
        if self.training:
            xf = x.float()
            m2, v2 = xf.mean((0, 2, 3)), xf.var((0, 2, 3), unbiased=False)
            mean = (m2[:c] + m2[c:]) / 2
            var = (v2[:c] + v2[c:]) / 2 + ((m2[:c] - m2[c:]) / 2) ** 2
            with torch.no_grad():
                n = x.numel() / c
                self.running_mean.lerp_(mean, self.momentum)
                self.running_var.lerp_(var * n / (n - 1), self.momentum)
        else:
            mean, var = self.running_mean, self.running_var
        scale = self.weight * torch.rsqrt(var + self.eps)
        shift = self.bias - mean * scale
        return x * scale.repeat(2).view(1, -1, 1, 1).to(x.dtype) + shift.repeat(2).view(1, -1, 1, 1).to(x.dtype)


class MirrorBlock(nn.Module):
    def __init__(self, i, o, down):
        super().__init__()
        self.conv1, self.bn1 = MirrorConv(i, o), MirrorBN(o)
        self.conv2, self.bn2 = MirrorConv(o, o), MirrorBN(o)
        self.pool = nn.AvgPool2d(2) if down else nn.Identity()
        self.skip = nn.Identity()
        if down or i != o:
            self.skip = nn.Sequential(MirrorConv(i, o, k=1), MirrorBN(o))

    def forward(self, x):
        x = self.pool(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.skip(x))


class MirrorResNet18(nn.Module):
    def __init__(self, classes=7, dropout=0.0, width=1.0, depth=2, stages=4, match="params"):
        super().__init__()
        pairs = [pair_count(64 * 2**k * width, match) for k in range(stages)]
        self.stem = nn.Sequential(MirrorConv(1, pairs[0], lift=True), MirrorBN(pairs[0]), nn.ReLU(inplace=True))
        blocks, c = [], pairs[0]
        for k, p in enumerate(pairs):
            blocks.append(nn.Sequential(MirrorBlock(c, p, down=k > 0), *[MirrorBlock(p, p, False) for _ in range(depth - 1)]))
            c = p
        self.stages = nn.Sequential(*blocks)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * c, classes)
        self.c = c

    def features(self, x):
        f = self.stages(self.stem(x)).mean((2, 3))
        a, b = f[:, :self.c], f[:, self.c:]
        return torch.cat([a + b, (a - b).abs()], 1)

    def forward(self, x):
        return self.fc(self.drop(self.features(x)))
