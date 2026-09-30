"""One training run: the whole dataset sits on the GPU, batches are drawn and augmented
there, the model runs in bfloat16 and is compiled. Model selection uses the validation
macro-F1 after every epoch; the test set is only scored with the selected weights."""
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .augment import augment
from .data import CLASSES
from .metrics import scores
from .models import build

MEAN, STD = 0.502, 0.240  # training-set pixel statistics (results/data/summary.json)
torch.backends.cudnn.benchmark = True  # input size never changes: let cuDNN pick the fastest kernels


class GPUData:
    """All images as uint8 on the GPU, with index tensors per split."""

    def __init__(self, path="data/processed/dataset.npz", device="cuda"):
        d = np.load(path)
        self.images = torch.tensor(d["images"], device=device)[:, None]
        self.label = torch.tensor(d["label"], device=device)
        self.source = torch.tensor(d["source"], device=device)
        split = torch.tensor(d["split"], device=device)
        self.idx = {name: torch.where(split == k)[0] for k, name in enumerate(["train", "val", "test"])}

    def batch(self, idx, strength=None, generator=None):
        """Images for idx, augmented if strength is given, normalised, ready for the model.
        Called once per epoch for the whole shuffled training set: one augmentation call
        for all images is far faster than one per batch."""
        x = self.images[idx].float().div_(255)
        if strength is not None:
            x = augment(x, strength, generator)
        return ((x - MEAN) / STD).contiguous(memory_format=torch.channels_last), self.label[idx]


def class_weights(labels, mode):
    if mode == "none":
        return None
    counts = torch.bincount(labels, minlength=len(CLASSES)).float()
    w = (counts.sum() / counts) ** (0.5 if mode == "sqrt" else 1.0)
    return w / (w * counts).sum() * counts.sum()  # average weight per image stays 1


def param_groups(model, weight_decay):
    """No weight decay on biases, norm scales, positions and tokens (the usual practice)."""
    decay, no_decay = [], []
    for p in model.parameters():
        (decay if p.ndim > 1 else no_decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


@torch.no_grad()
def predict(model, data, split, batch_size=1024):
    model.eval()
    probs = []
    for k in range(0, len(data.idx[split]), batch_size):
        x, _ = data.batch(data.idx[split][k:k + batch_size])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            probs.append(model(x).float().softmax(1))
    return torch.cat(probs), data.label[data.idx[split]]


def save(record, path):
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(path) + ".tmp")
        tmp.write_text(json.dumps(record))
        tmp.replace(path)


def run(cfg, data=None, trial=None, test=False, compile=True, log=print, record=None):
    """Train one model with the settings in cfg; return the history and the scores.

    cfg keys: model, epochs, batch_size, lr, weight_decay, warmup_epochs, label_smoothing,
    augment, class_weights, seed, and the model's own settings (dropout, drop_path).
    With an Optuna trial, the validation macro-F1 is reported each epoch for pruning.
    With a record path, the run's state is written there after every epoch (for following
    the training live).
    """
    data = data or GPUData()
    state = {"config": cfg, "status": "running", "started": time.time(), "history": []}
    save(state, record)
    torch.manual_seed(cfg["seed"])
    g = torch.Generator(device="cuda").manual_seed(cfg["seed"])
    model_args = {k: cfg[k] for k in ("dropout", "drop_path") if k in cfg}
    model = build(cfg["model"], **model_args).cuda().to(memory_format=torch.channels_last)
    fast = torch.compile(model) if compile else model

    train_idx = data.idx["train"]
    steps = len(train_idx) // cfg["batch_size"]
    total = steps * cfg["epochs"]
    warmup = steps * cfg["warmup_epochs"]
    opt = torch.optim.AdamW(param_groups(model, cfg["weight_decay"]), lr=cfg["lr"], fused=True)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warmup) if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total - warmup)))
    )
    weights = class_weights(data.label[train_idx], cfg["class_weights"])

    history, best, best_state = [], -1.0, None
    for epoch in range(cfg["epochs"]):
        model.train()
        t0 = time.perf_counter()
        perm = train_idx[torch.randperm(len(train_idx), device="cuda", generator=g)]
        xs, ys = data.batch(perm, cfg["augment"], g)
        loss_sum = torch.zeros((), device="cuda")
        correct = torch.zeros((), device="cuda")
        bs = cfg["batch_size"]
        for s in range(steps):
            x, y = xs[s * bs:(s + 1) * bs], ys[s * bs:(s + 1) * bs]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = fast(x)
                loss = F.cross_entropy(out.float(), y, weight=weights, label_smoothing=cfg["label_smoothing"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.detach()
            correct += (out.argmax(1) == y).sum()
        prob, true = predict(fast, data, "val")
        val = scores(prob, true)
        row = {
            "epoch": epoch + 1,
            "train_loss": (loss_sum / steps).item(),
            "train_accuracy": (correct / (steps * cfg["batch_size"])).item(),
            "val_loss": F.nll_loss(prob.clamp_min(1e-12).log(), true).item(),
            "val_accuracy": val["accuracy"],
            "val_macro_f1": val["macro_f1"],
            "lr": sched.get_last_lr()[0],
            "seconds": time.perf_counter() - t0,
        }
        history.append(row)
        state["history"] = history
        save(state, record)
        if val["macro_f1"] > best:
            best = val["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())
        if log:
            log(f"{cfg['model']} ep {epoch + 1:3d}  loss {row['train_loss']:.3f}  train {row['train_accuracy']:.3f}  "
                f"val {row['val_accuracy']:.3f} / F1 {row['val_macro_f1']:.3f}  {row['seconds']:.1f}s")
        if trial is not None:
            trial.report(val["macro_f1"], epoch)
            if trial.should_prune():
                import optuna

                state["status"] = "pruned"
                save(state, record)
                raise optuna.TrialPruned()

    model.load_state_dict(best_state)
    result = {"config": cfg, "history": history, "best_val_macro_f1": best}
    result["val"] = scores(*predict(model, data, "val"))
    if test:
        prob, true = predict(model, data, "test")
        result["test"] = scores(prob, true)
        src = data.source[data.idx["test"]]
        for name, k in [("test_fer", 0), ("test_raf", 1)]:
            result[name] = scores(prob[src == k], true[src == k])
        result["test_pred"] = prob.argmax(1).tolist()  # for bootstrap intervals later
        if record:
            torch.save(model.state_dict(), Path(record).with_suffix(".pt"))
    result["params"] = sum(p.numel() for p in model.parameters())
    result["peak_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9
    save({**state, **result, "status": "done", "finished": time.time()}, record)
    return result, model


def config(name, **overrides):
    """The default settings for a model (configs/defaults.yaml), with overrides."""
    import yaml

    with open("configs/defaults.yaml") as f:
        d = yaml.safe_load(f)
    return {**d["common"], **d[name], "model": name, **overrides}
