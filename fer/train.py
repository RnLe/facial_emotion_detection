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

    def __init__(self, path="data/processed/dataset.npz", device="cuda", arrays=None):
        d = arrays if arrays is not None else np.load(path)
        images = torch.tensor(d["images"], device=device)
        # grayscale (N, H, W) or color (N, H, W, 3)
        self.images = images.permute(0, 3, 1, 2).contiguous() if images.ndim == 4 else images[:, None]
        self.label = torch.tensor(d["label"], device=device)
        self.source = torch.tensor(d["source"], device=device)
        split = torch.tensor(d["split"], device=device)
        self.idx = {name: torch.where(split == k)[0] for k, name in enumerate(["train", "val", "test"])}
        self.original_index = d["original_index"] if "original_index" in d else None
        self._votes = None

    def vote_targets(self):
        """Per face a distribution over CLASSES: the FER+ vote shares for FER2013 faces
        (contempt, unknown and not-a-face votes left out), one-hot for RAF-DB faces."""
        if self._votes is None:
            from .data import FERPLUS_EMOTIONS, FERPLUS_TO_CLASS, load_fer2013

            votes = load_fer2013()["votes"]
            t = np.eye(len(CLASSES), dtype=np.float32)[self.label.cpu().numpy()]
            fer = (self.source == 0).cpu().numpy()
            for name, c in (FERPLUS_TO_CLASS.items() if fer.any() else []):
                t[fer, c] = votes[self.original_index[fer], FERPLUS_EMOTIONS.index(name)]
            if fer.any():
                t[fer] /= t[fer].sum(1, keepdims=True)
            self._votes = torch.tensor(t, device=self.label.device)
        return self._votes

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
def predict(model, data, split, batch_size=1024, tta=False):
    """Class probabilities for one split; tta also averages over the mirrored faces."""
    model.eval()
    probs = []
    for k in range(0, len(data.idx[split]), batch_size):
        x, _ = data.batch(data.idx[split][k:k + batch_size])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = model(x).float().softmax(1)
            if tta:
                p = (p + model(x.flip(-1)).float().softmax(1)) / 2
        probs.append(p)
    return torch.cat(probs), data.label[data.idx[split]]


def sam_step(model, loss_fn, opt, rho):
    """Sharpness-aware minimisation (Foret et al. 2021): step from the worst point within
    rho of the weights (first-order). loss_fn() runs the forward pass and returns
    (loss, output)."""
    loss, out = loss_fn()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    params = [p for p in model.parameters() if p.grad is not None]
    with torch.no_grad():
        norm = torch.norm(torch.stack([p.grad.norm() for p in params]))
        eps = [p.grad * (rho / (norm + 1e-12)) for p in params]
        for p, e in zip(params, eps):
            p.add_(e)
    opt.zero_grad(set_to_none=True)
    loss_fn()[0].backward()
    with torch.no_grad():
        for p, e in zip(params, eps):
            p.sub_(e)
    opt.step()
    return loss, out


def save(record, path):
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(path) + ".tmp")
        tmp.write_text(json.dumps(record))
        tmp.replace(path)


def run(cfg, data=None, trial=None, test=False, compile=True, log=print, record=None):
    """Train one model with the settings in cfg; return the history and the scores.

    cfg keys: model, epochs, batch_size, lr, weight_decay, warmup_epochs, label_smoothing,
    augment, class_weights, seed, and the model's own settings (dropout, drop_path, and
    width, depth, stages for the shape stage).
    Optional (part two): targets="votes" (FER+ vote shares as soft targets), logit_adjust
    (tau of Menon et al. 2021: the loss sees logits + tau log prior), ema (decay of an
    average of the weights, which is then validated and kept), sam (rho), tta (validate
    and test with mirrored faces too), teacher (path of cached log-probabilities for every
    face of the dataset) with kd_alpha and kd_temperature, keep_last.
    With an Optuna trial, the validation macro-F1 is reported each epoch for pruning.
    With a record path, the run's state is written there after every epoch (for following
    the training live).
    """
    data = data or GPUData()
    state = {"config": cfg, "status": "running", "started": time.time(), "history": []}
    save(state, record)
    torch.manual_seed(cfg["seed"])
    g = torch.Generator(device="cuda").manual_seed(cfg["seed"])
    model_args = {k: cfg[k] for k in ("dropout", "drop_path", "width", "depth", "stages", "in_ch", "stem_stride", "readout", "match") if k in cfg}
    model = build(cfg["model"], **model_args).cuda().to(memory_format=torch.channels_last)
    fast = torch.compile(model) if cfg.get("compile", compile) else model

    train_idx = data.idx["train"]
    steps = len(train_idx) // cfg["batch_size"]
    total = steps * cfg["epochs"]
    warmup = steps * cfg["warmup_epochs"]
    opt = torch.optim.AdamW(param_groups(model, cfg["weight_decay"]), lr=cfg["lr"], fused=True)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warmup) if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total - warmup)))
    )
    weights = class_weights(data.label[train_idx], cfg["class_weights"])
    targets = data.vote_targets() if cfg.get("targets") == "votes" else None
    log_prior = None
    if cfg.get("logit_adjust"):
        counts = torch.bincount(data.label[train_idx], minlength=len(CLASSES)).float()
        log_prior = cfg["logit_adjust"] * (counts / counts.sum()).log()
    teacher = None
    if cfg.get("teacher"):
        teacher = torch.load(cfg["teacher"])["dataset"].float().cuda()
        temp, alpha = cfg.get("kd_temperature", 2.0), cfg.get("kd_alpha", 0.5)
    ema = None
    if cfg.get("ema"):
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

        ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(cfg["ema"]), use_buffers=True)
    tta = bool(cfg.get("tta"))

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
            batch = perm[s * bs:(s + 1) * bs]
            x, y = xs[s * bs:(s + 1) * bs], ys[s * bs:(s + 1) * bs]
            target = targets[batch] if targets is not None else y

            def loss_fn():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = fast(x).float()
                logits = out + log_prior if log_prior is not None else out
                loss = F.cross_entropy(logits, target, weight=weights, label_smoothing=cfg["label_smoothing"])
                if teacher is not None:
                    kd = F.kl_div(F.log_softmax(out / temp, 1), F.log_softmax(teacher[batch] / temp, 1), log_target=True, reduction="batchmean")
                    loss = (1 - alpha) * loss + alpha * temp**2 * kd
                return loss, out

            if cfg.get("sam"):
                loss, out = sam_step(model, loss_fn, opt, cfg["sam"])
            else:
                loss, out = loss_fn()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            sched.step()
            if ema is not None:
                ema.update_parameters(model)
            loss_sum += loss.detach()
            correct += (out.argmax(1) == y).sum()
        judged = ema.module if ema is not None else fast
        prob, true = predict(judged, data, "val", tta=tta)
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
            best_state = copy.deepcopy((ema.module if ema is not None else model).state_dict())
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

    if cfg.get("keep_last"):  # keep_last: the final weights (for the overfitting check)
        if ema is not None:
            model.load_state_dict(ema.module.state_dict())
    else:
        model.load_state_dict(best_state)
    result = {"config": cfg, "history": history, "best_val_macro_f1": best}
    result["val"] = scores(*predict(model, data, "val", tta=tta))
    if test:
        prob, true = predict(model, data, "test", tta=tta)
        result["test"] = scores(prob, true)
        src = data.source[data.idx["test"]]
        for name, k in [("test_fer", 0), ("test_raf", 1)]:
            if (src == k).any():
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
