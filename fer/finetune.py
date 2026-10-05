"""Fine-tuning and scoring the reference models on our data.

The images stay at 48x48 grayscale on the GPU (GPUData). Each batch is augmented there as
in our own runs, then scaled up to the model's input size (bilinear, as in the references'
own evaluation), copied into three channels, and normalised with the model's statistics.
Fine-tuning runs a fixed number of epochs and keeps the last weights: the RAF-DB
references were trained on RAF-DB's train set, which holds our RAF validation images, so
selecting on validation would favour them.
"""
import math
import re
import time

import torch
import torch.nn.functional as F

from .augment import augment
from .metrics import scores
from .reference import to_ours
from .train import class_weights


def prepare(x, spec):
    """Images in [0, 1], one or three channels, any size, to the model's input."""
    if x.shape[-1] != spec.size:
        x = F.interpolate(x, size=(spec.size, spec.size), mode="bilinear", align_corners=False)
    if x.shape[1] == 1:
        x = x.expand(-1, 3, -1, -1)
    mean = torch.tensor(spec.mean, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(spec.std, device=x.device).view(1, 3, 1, 1)
    return ((x - mean) / std).contiguous()


def raw(data, idx):
    return data.images[idx].float().div_(255)


@torch.no_grad()
def predict(model, spec, images, batch_size=256):
    """Class probabilities (CLASSES order) for uint8 images (N, 1, h, w) or (N, 3, h, w)."""
    model.eval()
    out = []
    for k in range(0, len(images), batch_size):
        x = prepare(images[k:k + batch_size].float().div(255), spec)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(to_ours(model(x).float(), spec.order).softmax(1))
    return torch.cat(out)


def evaluate(model, spec, data, split):
    """Scores on one split of a GPUData set, overall and per source."""
    idx = data.idx[split]
    prob = predict(model, spec, data.images[idx])
    true, src = data.label[idx], data.source[idx]
    out = {split: scores(prob, true)}
    for name, k in [("fer", 0), ("raf", 1)]:
        if (src == k).any() and (src != k).any():
            out[f"{split}_{name}"] = scores(prob[src == k], true[src == k])
    return out, prob


def layer_id(name, depth):
    """Depth of a parameter in a ViT for layer-wise learning-rate decay (BEiT, MAE):
    embeddings 0, block i is i + 1, the final norm and the head depth + 1."""
    m = re.search(r"(?:blocks|resblocks)\.(\d+)\.", name)
    if m:
        return int(m.group(1)) + 1
    if re.search(r"patch_embed|cls_token|pos_embed|conv1|class_embedding|positional_embedding|ln_pre", name):
        return 0
    return depth + 1


def param_groups(model, lr, weight_decay, layer_decay=None, depth=None):
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        no_decay = p.ndim == 1 or re.search(r"cls_token|pos_embed|class_embedding|positional_embedding", name)
        scale = 1.0 if layer_decay is None else layer_decay ** (depth + 1 - layer_id(name, depth))
        key = (scale, bool(no_decay))
        g = groups.setdefault(key, {"params": [], "lr": lr * scale, "weight_decay": 0.0 if no_decay else weight_decay})
        g["params"].append(p)
    return list(groups.values())


def finetune(model, spec, data, cfg, log=print, on_epoch=None):
    """cfg: epochs, batch_size, lr, weight_decay, warmup_epochs, label_smoothing,
    class_weights, augment, seed; optional layer_decay with depth (ViTs)."""
    torch.manual_seed(cfg["seed"])
    g = torch.Generator(device="cuda").manual_seed(cfg["seed"])
    train_idx = data.idx["train"]
    bs = cfg["batch_size"]
    steps = len(train_idx) // bs
    total, warmup = steps * cfg["epochs"], int(steps * cfg["warmup_epochs"])
    opt = torch.optim.AdamW(
        param_groups(model, cfg["lr"], cfg["weight_decay"], cfg.get("layer_decay"), cfg.get("depth")), fused=True
    )
    for group in opt.param_groups:
        group["base_lr"] = group["lr"]
    weights = class_weights(data.label[train_idx], cfg["class_weights"])
    history = []
    for epoch in range(cfg["epochs"]):
        model.train()
        t0 = time.perf_counter()
        perm = train_idx[torch.randperm(len(train_idx), device="cuda", generator=g)]
        loss_sum = torch.zeros((), device="cuda")
        correct = torch.zeros((), device="cuda")
        for s in range(steps):
            step = epoch * steps + s
            f = (step + 1) / max(1, warmup) if step < warmup else 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * f
            idx = perm[s * bs:(s + 1) * bs]
            x = raw(data, idx)
            if cfg["augment"]:
                x = augment(x, cfg["augment"], g)
            y = data.label[idx]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = to_ours(model(prepare(x, spec)).float(), spec.order)
                loss = F.cross_entropy(out, y, weight=weights, label_smoothing=cfg["label_smoothing"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += loss.detach()
            correct += (out.argmax(1) == y).sum()
        val, _ = evaluate(model, spec, data, "val")
        row = {
            "epoch": epoch + 1,
            "train_loss": (loss_sum / steps).item(),
            "train_accuracy": (correct / (steps * bs)).item(),
            **{f"{k}_accuracy": v["accuracy"] for k, v in val.items()},
            "val_macro_f1": val["val"]["macro_f1"],
            "seconds": time.perf_counter() - t0,
        }
        history.append(row)
        if on_epoch:
            on_epoch(history)
        if log:
            log(" ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()))
    return history
