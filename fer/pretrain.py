"""Masked-face pretraining (part two, phase 5): the network learns to fill in masked parts
of our own training faces, without labels (SimMIM, Xie et al. 2022, here with a CNN).

Each 48x48 face is cut into 6x6 patches of 8x8 pixels and a random 60% of them are
replaced by a learned mask value. The encoder is the classifier without its head (ResNet-18:
its last stage is a 6x6 grid, one cell per patch); a 1x1 conv predicts the 64 pixels of
each cell's patch. The loss is the squared error on the masked patches, each target patch
normalised by its own mean and spread (MAE's normalised pixels, He et al. 2022).
"""
import math
import time

import torch
import torch.nn.functional as F
from torch import nn

from .models import build
from .train import MEAN, STD, param_groups, save

PATCH = 8


class MaskedFace(nn.Module):
    def __init__(self, encoder, channels):
        super().__init__()
        self.encoder = encoder
        self.mask_value = nn.Parameter(torch.zeros(()))
        self.decoder = nn.Conv2d(channels, PATCH * PATCH, 1)

    def forward(self, x, mask):
        """x: normalised faces (B, 1, 48, 48); mask: (B, 6, 6) booleans, True = hidden."""
        m = mask.repeat_interleave(PATCH, 1).repeat_interleave(PATCH, 2)[:, None].to(x.dtype)
        x = x * (1 - m) + self.mask_value * m
        f = self.encoder.stages(self.encoder.stem(x))
        return F.pixel_shuffle(self.decoder(f), PATCH)


def patches(x):
    """(B, 1, 48, 48) -> (B, 36, 64)"""
    b = x.shape[0]
    return x.reshape(b, 6, PATCH, 6, PATCH).permute(0, 1, 3, 2, 4).reshape(b, 36, PATCH * PATCH)


def masked_pretrain(cfg, data, record=None, log=print):
    """cfg: model settings of the classifier (model, width, depth, stages), epochs,
    batch_size, lr, weight_decay, warmup_epochs, mask_ratio, augment, seed.
    Returns the encoder's state dict (the classifier's keys, head left out)."""
    from .augment import augment

    torch.manual_seed(cfg["seed"])
    g = torch.Generator(device="cuda").manual_seed(cfg["seed"])
    model_args = {k: cfg[k] for k in ("width", "depth", "stages") if k in cfg}
    encoder = build(cfg["model"], **model_args)
    channels = encoder.head[-1].in_features
    net = MaskedFace(encoder, channels).cuda().to(memory_format=torch.channels_last)
    fast = torch.compile(net)
    train_idx = data.idx["train"]
    bs = cfg["batch_size"]
    steps = len(train_idx) // bs
    total, warmup = steps * cfg["epochs"], steps * cfg["warmup_epochs"]
    opt = torch.optim.AdamW(param_groups(net, cfg["weight_decay"]), lr=cfg["lr"], fused=True)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warmup) if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total - warmup)))
    )
    n_hidden = round(cfg["mask_ratio"] * 36)
    state = {"config": cfg, "status": "running", "started": time.time(), "history": []}
    for epoch in range(cfg["epochs"]):
        net.train()
        t0 = time.perf_counter()
        perm = train_idx[torch.randperm(len(train_idx), device="cuda", generator=g)]
        x01 = augment(data.images[perm].float().div_(255), cfg["augment"], g)
        xs = ((x01 - MEAN) / STD).contiguous(memory_format=torch.channels_last)
        loss_sum = torch.zeros((), device="cuda")
        for s in range(steps):
            x = xs[s * bs:(s + 1) * bs]
            order = torch.rand(len(x), 36, device="cuda", generator=g).argsort(1)
            mask = torch.zeros(len(x), 36, dtype=torch.bool, device="cuda")
            mask.scatter_(1, order[:, :n_hidden], True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = fast(x, mask.view(-1, 6, 6))
            target = patches(x.float())
            target = (target - target.mean(-1, keepdim=True)) / (target.var(-1, keepdim=True) + 1e-6).sqrt()
            err = ((patches(pred.float()) - target) ** 2).mean(-1)
            loss = (err * mask).sum() / mask.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.detach()
        row = {"epoch": epoch + 1, "loss": (loss_sum / steps).item(), "lr": sched.get_last_lr()[0], "seconds": time.perf_counter() - t0}
        state["history"].append(row)
        save(state, record)
        if log:
            log(f"masked ep {epoch + 1:3d}  loss {row['loss']:.4f}  {row['seconds']:.1f}s")
    weights = {k: v for k, v in encoder.state_dict().items() if not k.startswith("head.")}
    save({**state, "status": "done", "finished": time.time()}, record)
    return weights
