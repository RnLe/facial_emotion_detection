"""Long runs: the three strongest models trained far past 60 epochs, to see whether they
converge, overfit, or start to generalise late (grokking).

Three runs per model:
  tuned     the stage 1 best settings, default shape
  larger    the stage 2 best shape and settings
  memorize  the stage 1 settings without augmentation, label smoothing and dropout, so the
            training set can be learned by heart, and with weight decay set to give
            lr x weight decay = 1e-3 per step, as in the original grokking runs
            (Power et al. 2022: AdamW, lr 1e-3, weight decay 1)
  grok      continues the memorize run after its first cooldown, where it has learned
            every training face (100%) and scores 81.5% on validation (VGG). Same
            settings at a tenth of the learning rate, low enough to stay memorised (the
            cooldown fitted every training face from lr 8.7e-4 down), with weight decay
            unchanged: lr x weight decay = 1e-4 per step. Grokking would show as
            validation accuracy rising long after training accuracy reached 100%.
            Run it with --variants grok and --epochs past 110 (it starts at epoch 110).

The learning rate warms up and then stays constant, so no run is tied to a fixed length:
--epochs 300 continues a run from where it stopped. To see what a run would reach if it
ended at a given point, a cooldown branches off at the end of each block: 10 epochs with
the learning rate falling linearly to zero (warmup-stable-decay, Hagele et al. 2024),
scored on validation and test. The main run never sees the cooldown.

Every epoch records loss and accuracy on the augmented batches, on 4,316 fixed training
faces without augmentation, and on validation, plus the measures in fer/measures.py.
Checkpoints every 5 epochs; a stopped run resumes from its last checkpoint.
"""
import argparse
import json
import os
import time

import optuna
import torch
import torch.nn.functional as F

from fer.measures import entropy, neural_collapse, predict_with_features, spectral_entropy, weight_measures
from fer.metrics import scores
from fer.models import build
from fer.runs import RUNS, add_to_plan, record_path
from fer.train import GPUData, class_weights, config, param_groups, save

p = argparse.ArgumentParser()
p.add_argument("--models", nargs="+", default=["vgg", "resnet", "densenet"])
p.add_argument("--variants", nargs="+", default=["tuned", "memorize", "larger"])
p.add_argument("--epochs", type=int, default=100, help="train every run up to this epoch")
p.add_argument("--cooldown", type=int, default=10)
p.add_argument("--every", type=int, default=5, help="checkpoint every this many epochs")
args = p.parse_args()


def best_params(name):
    storage = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(str(RUNS / "optuna" / f"{name}.log")))
    return optuna.load_study(study_name=name, storage=storage).best_params


def settings(model, variant):
    s = best_params(model)
    if variant == "larger":
        s = {**s, **best_params(f"{model}_shape")}
    if variant in ("memorize", "grok"):
        s = {**s, "augment": 0.0, "label_smoothing": 0.0, "dropout": 0.0, "weight_decay": 1e-3 / s["lr"]}
    if variant == "grok":
        s["lr"] /= 10
    return config(model, **s, epochs=args.epochs, stage="long", iteration=variant, schedule="constant")


def save_checkpoint(path, model, opt, g, epoch, step, history):
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "g": g.get_state(), "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(), "epoch": epoch, "step": step, "history": history}, tmp)
    os.replace(tmp, path)


def train(cfg, data, probe, record, checkpoint, until, lr_at, start=None, test=False):
    """Train from `start` (a checkpoint file) or from scratch up to epoch `until`, with the
    learning rate lr_at(step). Writes the record every epoch and a checkpoint every
    args.every epochs and at the end."""
    torch.manual_seed(cfg["seed"])
    g = torch.Generator(device="cuda").manual_seed(cfg["seed"])
    model_args = {k: cfg[k] for k in ("dropout", "width", "depth", "stages") if k in cfg}
    model = build(cfg["model"], **model_args).cuda().to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(param_groups(model, cfg["weight_decay"]), lr=cfg["lr"], fused=True)
    epoch, step, history = 0, 0, []
    if start is not None and start.exists():
        c = torch.load(start, weights_only=False)
        model.load_state_dict(c["model"])
        opt.load_state_dict(c["opt"])
        g.set_state(c["g"])
        torch.set_rng_state(c["rng"])
        torch.cuda.set_rng_state(c["cuda_rng"])
        epoch, step, history = c["epoch"], c["step"], c["history"]
        if checkpoint != start:  # a branch (the cooldown) keeps its own history
            history = []
    fast = torch.compile(model) if cfg.get("compile", True) else model

    train_idx = data.idx["train"]
    bs = cfg["batch_size"]
    steps = len(train_idx) // bs
    weights = class_weights(data.label[train_idx], cfg["class_weights"])
    strength = cfg["augment"] if cfg["augment"] > 0 else None  # None: no mirroring either
    state = {"config": cfg, "status": "running", "started": time.time(), "history": history,
             "params": sum(p.numel() for p in model.parameters())}
    save(state, record)
    first = epoch
    for epoch in range(first, until):
        model.train()
        t0 = time.perf_counter()
        perm = train_idx[torch.randperm(len(train_idx), device="cuda", generator=g)]
        xs, ys = data.batch(perm, strength, g)
        loss_sum = torch.zeros((), device="cuda")
        correct = torch.zeros((), device="cuda")
        for s in range(steps):
            for group in opt.param_groups:
                group["lr"] = lr_at(step)
            x, y = xs[s * bs:(s + 1) * bs], ys[s * bs:(s + 1) * bs]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = fast(x)
                loss = F.cross_entropy(out.float(), y, weight=weights, label_smoothing=cfg["label_smoothing"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            loss_sum += loss.detach()
            correct += (out.argmax(1) == y).sum()
        seconds = time.perf_counter() - t0

        prob_v, feat_v = predict_with_features(model, data, data.idx["val"])
        prob_t, feat_t = predict_with_features(model, data, probe)
        true_v, true_t = data.label[data.idx["val"]], data.label[probe]
        val, trn = scores(prob_v, true_v), scores(prob_t, true_t)
        row = {
            "epoch": epoch + 1,
            "train_loss": (loss_sum / steps).item(),
            "train_accuracy": (correct / (steps * bs)).item(),
            "train_clean_loss": F.nll_loss(prob_t.clamp_min(1e-12).log(), true_t).item(),
            "train_clean_accuracy": trn["accuracy"],
            "val_loss": F.nll_loss(prob_v.clamp_min(1e-12).log(), true_v).item(),
            "val_accuracy": val["accuracy"],
            "val_macro_f1": val["macro_f1"],
            "gap": trn["accuracy"] - val["accuracy"],
            "pred_entropy": entropy(prob_v).mean().item(),
            "repr_entropy": spectral_entropy(feat_v),
            "nc1": neural_collapse(feat_t, true_t),
            **weight_measures(model),
            "lr": lr_at(step - 1),
            "seconds": seconds,
            "measure_seconds": time.perf_counter() - t0 - seconds,
        }
        history.append(row)
        state["history"] = history
        state["best_val_macro_f1"] = max(h["val_macro_f1"] for h in history)
        save(state, record)
        if (epoch + 1) % args.every == 0 or epoch + 1 == until:
            save_checkpoint(checkpoint, model, opt, g, epoch + 1, step, history)

    prob, _ = predict_with_features(model, data, data.idx["val"])
    result = {**state, "status": "done", "finished": time.time(), "val": scores(prob, data.label[data.idx["val"]])}
    if test:
        prob, _ = predict_with_features(model, data, data.idx["test"])
        true = data.label[data.idx["test"]]
        result["test"] = scores(prob, true)
        src = data.source[data.idx["test"]]
        for name, k in [("test_fer", 0), ("test_raf", 1)]:
            result[name] = scores(prob[src == k], true[src == k])
        result["test_pred"] = prob.argmax(1).tolist()
    save(result, record)
    return step


torch.cuda.set_per_process_memory_fraction(0.85)
data = GPUData()
probe = data.idx["train"][torch.randperm(len(data.idx["train"]), generator=torch.Generator().manual_seed(0))[:len(data.idx["val"])].to(data.idx["train"].device)]
runs = [(m, v) for m in args.models for v in args.variants]
add_to_plan([{**settings(m, v), "stage": "long", "iteration": v} for m, v in runs]
            + [{**settings(m, v), "epochs": args.cooldown, "stage": "cooldown", "iteration": f"{v}_e{args.epochs}"} for m, v in runs])

for model, variant in runs:
    cfg = settings(model, variant)
    record = record_path("long", model, variant)
    checkpoint = record.with_suffix(".ckpt")
    done = torch.load(checkpoint, weights_only=False)["epoch"] if checkpoint.exists() else 0
    steps = len(data.idx["train"]) // cfg["batch_size"]
    warmup, lr = steps * cfg["warmup_epochs"], cfg["lr"]
    start, schedule = checkpoint, lambda s: lr * min(1.0, (s + 1) / max(1, warmup))
    if variant == "grok":  # branches off the memorize run's first cooldown, no warmup
        start = checkpoint if checkpoint.exists() else record_path("cooldown", model, "memorize_e100").with_suffix(".ckpt")
        schedule = lambda s: lr
        done = done or 110
    if done < args.epochs:
        print(f"{model} {variant}: epochs {done} to {args.epochs}", flush=True)
        torch.cuda.empty_cache()
        train(cfg, data, probe, record, checkpoint, args.epochs, schedule, start=start)
        milestone = record.with_name(f"{variant}_e{args.epochs}.ckpt")
        milestone.write_bytes(checkpoint.read_bytes())  # kept for the cooldown and later work

    # Cooldown from the checkpoint at epoch args.epochs: the learning rate falls linearly to zero.
    cool = record_path("cooldown", model, f"{variant}_e{args.epochs}")
    milestone = record.with_name(f"{variant}_e{args.epochs}.ckpt")
    if milestone.exists() and not (cool.exists() and json.loads(cool.read_text()).get("status") == "done"):
        print(f"{model} {variant}: cooldown from epoch {args.epochs}", flush=True)
        torch.cuda.empty_cache()
        start_step, n = args.epochs * steps, args.cooldown * steps
        ccfg = {**cfg, "stage": "cooldown", "iteration": f"{variant}_e{args.epochs}", "epochs": args.epochs + args.cooldown, "schedule": "linear to zero"}
        train(ccfg, data, probe, cool, cool.with_suffix(".ckpt"), args.epochs + args.cooldown,
              lambda s: lr * max(0.0, 1 - (s - start_step) / n), start=milestone, test=True)
        r = json.loads(cool.read_text())
        print(f"{model} {variant}: after cooldown val acc {r['val']['accuracy']:.3f}, test acc {r['test']['accuracy']:.3f}, "
              f"test F1 {r['test']['macro_f1']:.3f}", flush=True)
