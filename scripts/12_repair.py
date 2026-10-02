"""Repair: per-layer ranks for a parameter budget, then fine-tuning to win back accuracy.

For each model (the three larger long-run models after their cooldown) and factorisation
(output-based low rank, Tucker-2, spatial split):
1. Sensitivity: every layer alone at every candidate rank, scored by the KL divergence
   from the original predictions on 2,048 training faces (other faces than the 4,096 that
   fit the projections; validation is not used). Cached per model and method.
2. Allocation: the ranks with the smallest summed KL for 2x, 4x and 8x fewer parameters
   (fer.compress.allocate). A budget no choice of ranks can reach is skipped.
3. Repair: --epochs on the full training set with the model's own settings (augmentation,
   label smoothing, class weights, weight decay, batch size). The learning rate falls
   linearly from --lr-scale times the original to 0. --distill adds distillation from the
   uncompressed model: loss = (1 - a) CE + a T^2 KL(teacher || student), T = 2, a = 0.5.
Validation before and after the repair, the test set once at the end, speed as in
11_compress.py. Records in runs/repair/<model>/<method>_<budget>x<tag>.json, the repaired
model next to it as .pt.
"""
import argparse
import copy
import json
import time
import warnings

import torch
import torch.nn.functional as F

from fer.compress import allocate, compress_ranks, macs, output_stats, sensitivity, size_bytes, speed
from fer.metrics import scores
from fer.models import build
from fer.runs import RUNS, add_to_plan, record_path
from fer.train import GPUData, class_weights, param_groups, save

p = argparse.ArgumentParser()
p.add_argument("--models", nargs="+", default=["vgg", "resnet", "densenet"])
p.add_argument("--methods", nargs="+", default=["low_rank", "tucker2", "spatial"])
p.add_argument("--budgets", nargs="+", type=float, default=[2, 4, 8])
p.add_argument("--epochs", type=int, default=10)
p.add_argument("--lr-scale", type=float, default=0.1)
p.add_argument("--distill", action="store_true")
p.add_argument("--tag", default="", help="suffix for the record name (pilot runs)")
args = p.parse_args()
warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True


def load(name):
    rec = json.loads((RUNS / "cooldown" / name / "larger_e100.json").read_text())
    cfg = rec["config"]
    net = build(name, **{k: cfg[k] for k in ("dropout", "width", "depth", "stages") if k in cfg})
    net.load_state_dict(torch.load(RUNS / "cooldown" / name / "larger_e100.ckpt", weights_only=False, map_location="cpu")["model"])
    return net.cuda().eval(), cfg


@torch.no_grad()
def evaluate(model, split, test_sources=False):
    model.eval()
    idx = data.idx[split]
    probs = []
    for k in range(0, len(idx), 1024):
        x, _ = data.batch(idx[k:k + 1024])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            probs.append(model(x).float().softmax(1))
    probs, true = torch.cat(probs), data.label[idx]
    out = {split: scores(probs, true)}
    if test_sources:
        src = data.source[idx]
        for name, k in [("test_fer", 0), ("test_raf", 1)]:
            out[name] = scores(probs[src == k], true[src == k])
        out["test_pred"] = probs.argmax(1).tolist()
    return out


def repair(model, teacher, cfg, state, record):
    model.train().to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(param_groups(model, cfg["weight_decay"]), lr=cfg["lr"] * args.lr_scale, fused=True)
    g = torch.Generator(device="cuda").manual_seed(cfg["seed"])
    train_idx, bs = data.idx["train"], cfg["batch_size"]
    steps = len(train_idx) // bs
    weights = class_weights(data.label[train_idx], cfg["class_weights"])
    lr0, total, step = cfg["lr"] * args.lr_scale, steps * args.epochs, 0
    for epoch in range(args.epochs):
        model.train()
        t0 = time.perf_counter()
        xs, ys = data.batch(train_idx[torch.randperm(len(train_idx), device="cuda", generator=g)], cfg["augment"], g)
        loss_sum = torch.zeros((), device="cuda")
        for s in range(steps):
            for group in opt.param_groups:
                group["lr"] = lr0 * (1 - step / total)
            x, y = xs[s * bs:(s + 1) * bs], ys[s * bs:(s + 1) * bs]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x).float()
                loss = F.cross_entropy(out, y, weight=weights, label_smoothing=cfg["label_smoothing"])
                if teacher is not None:
                    with torch.no_grad():
                        t = teacher(x).float()
                    kd = F.kl_div(F.log_softmax(out / 2, 1), F.log_softmax(t / 2, 1), log_target=True, reduction="batchmean")
                    loss = 0.5 * loss + 0.5 * 4 * kd
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            loss_sum += loss.detach()
        val = evaluate(model, "val")["val"]
        state["history"].append({"epoch": epoch + 1, "train_loss": (loss_sum / steps).item(), "val_accuracy": val["accuracy"],
                                 "val_macro_f1": val["macro_f1"], "lr": lr0 * (1 - step / total), "seconds": time.perf_counter() - t0})
        save(state, record)


data = GPUData()
g = torch.Generator().manual_seed(0)
perm = torch.randperm(len(data.idx["train"]), generator=g).to(data.idx["train"].device)
x_fit = data.batch(data.idx["train"][perm[:4096]])[0].contiguous()
x_sens = data.batch(data.idx["train"][perm[4096:6144]])[0].contiguous()
x_val = data.batch(data.idx["val"])[0].cpu().contiguous()
x1, x256 = x_val[:1].clone(), x_val[:256].clone()
tag = ("_kd" if args.distill else "") + args.tag
add_to_plan([{"model": m, "stage": "repair", "iteration": f"{meth}_{b:g}x{tag}", "epochs": args.epochs}
             for m in args.models for meth in args.methods for b in args.budgets])

for name in args.models:
    base, cfg = load(name)
    total = sum(p.numel() for p in base.parameters())
    stats = output_stats(base, x_fit)
    for method in args.methods:
        cache = RUNS / "repair" / name / f"{method}_sensitivity.json"
        if cache.exists():
            table = json.loads(cache.read_text())
        else:
            t0 = time.perf_counter()
            table = sensitivity(base, method, x_sens, stats)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(table))
            print(f"{name} {method}: sensitivity of {len(table)} layers in {time.perf_counter() - t0:.0f} s", flush=True)
        for budget in args.budgets:
            record = record_path("repair", name, f"{method}_{budget:g}x{tag}")
            if record.exists() and json.loads(record.read_text()).get("status") == "done":
                continue
            plan = allocate(table, total, total / budget)
            if plan is None:
                print(f"{name} {method} {budget:g}x: out of reach", flush=True)
                continue
            ranks, _ = plan
            model = compress_ranks(base, method, ranks, stats)
            rcfg = {**cfg, "stage": "repair", "iteration": f"{method}_{budget:g}x{tag}", "epochs": args.epochs, "method": method,
                    "budget": budget, "lr_scale": args.lr_scale, "distill": args.distill, "schedule": "linear to zero"}
            state = {"config": rcfg, "status": "running", "started": time.time(), "history": [], "ranks": ranks,
                     "params": sum(p.numel() for p in model.parameters()), "bytes": size_bytes(model),
                     "macs": macs(copy.deepcopy(model).cpu(), x1), "base_params": total,
                     "zero_shot": evaluate(model, "val")["val"]}
            save(state, record)
            repair(model, base if args.distill else None, cfg, state, record)
            state.update(evaluate(model, "val"))
            state.update(evaluate(model, "test", test_sources=True))
            torch.save(model.to(memory_format=torch.contiguous_format), record.with_suffix(".pt"))
            state["speed"] = speed(model, x1, x256)
            state["status"], state["finished"] = "done", time.time()
            save(state, record)
            z, v = state["zero_shot"], state["val"]
            print(f"{name} {method} {budget:g}x{tag}: {state['params'] / 1e6:.2f} M, val {100 * z['accuracy']:.1f} -> "
                  f"{100 * v['accuracy']:.1f} after repair, CPU 1 face {state['speed']['cpu_1_ms']:.1f} ms", flush=True)
            model.cuda()
