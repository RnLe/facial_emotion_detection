"""Data augmentation on the GPU, one whole batch at a time.

Only well-tried transforms, each a plausible change of the same face:
- mirror left/right (an expression is symmetric enough),
- small rotation, zoom and shift (head pose and framing), as one affine warp,
- brightness and contrast (lighting),
- random erasing: blank out a patch (hands, hair, glasses cover parts of faces).

One `strength` in [0, 1] scales all of them, so the tuning can decide how much
augmentation each model wants. strength=1 means: rotation up to 15 degrees,
zoom and shift up to 10%, brightness up to 0.2, contrast up to 30%, and an
erased patch in a quarter of the images.
"""
import math

import torch
import torch.nn.functional as F


def augment(x, strength=1.0, generator=None):
    """x: float batch (B, 1, H, W) in [0, 1]. Returns a new batch, same shape."""
    b = x.shape[0]
    dev = x.device

    def u(lo, hi):  # one uniform number per image
        return torch.empty(b, device=dev).uniform_(lo, hi, generator=generator)

    flip = torch.rand(b, device=dev, generator=generator) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    if strength <= 0:
        return x

    angle = u(-15, 15) * strength * math.pi / 180
    scale = 1 + u(-0.1, 0.1) * strength
    shift = torch.stack([u(-0.1, 0.1), u(-0.1, 0.1)], 1) * strength * 2  # grid runs from -1 to 1
    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.stack([torch.stack([cos, -sin, shift[:, 0]], 1), torch.stack([sin, cos, shift[:, 1]], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)

    mean = x.mean((1, 2, 3), keepdim=True)
    x = (x - mean) * (1 + u(-0.3, 0.3) * strength)[:, None, None, None] + mean
    x = x + (u(-0.2, 0.2) * strength)[:, None, None, None]

    erase = torch.rand(b, device=dev, generator=generator) < 0.25 * strength
    if erase.any():
        h, w = x.shape[-2:]
        area = u(0.02, 0.2) * h * w
        ratio = torch.exp(u(math.log(0.3), math.log(3.3)))
        eh = (area * ratio).sqrt().clamp(1, h)
        ew = (area / ratio).sqrt().clamp(1, w)
        y0 = u(0, 1) * (h - eh)
        x0 = u(0, 1) * (w - ew)
        ys = torch.arange(h, device=dev)[None, :, None]
        xs = torch.arange(w, device=dev)[None, None, :]
        box = (ys >= y0[:, None, None]) & (ys < (y0 + eh)[:, None, None]) & (xs >= x0[:, None, None]) & (xs < (x0 + ew)[:, None, None])
        box &= erase[:, None, None]
        noise = torch.rand(x.shape, device=dev, generator=generator)
        x = torch.where(box[:, None], noise, x)
    return x.clamp(0, 1)
