"""int8 on top of compression, and quantisation-aware training for DenseNet.

1. Every repaired factorised model (runs/repair/<model>/*.pt) quantised to int8 after
   training as in 11_compress.py (DenseNet with its concatenations left in float), scored
   on validation and test, timed on the CPU.
2. DenseNet (the larger long-run model) with quantisation-aware training: fake int8 in the
   forward pass during --epochs of training at a low learning rate, so the network can
   adapt to the one scale its concatenations share. Observers and batch norm statistics
   are frozen for the last two epochs. Once with everything in int8, once with the
   concatenations in float.
Records in runs/quant/<model>/<name>.json.
"""
import argparse
import copy
import json
import time
import warnings

import torch
import torch.ao.nn.intrinsic.qat as nniqat
import torch.nn.functional as F
from torch import nn
from torch.ao.quantization import disable_observer, get_default_qat_qconfig_mapping
from torch.ao.quantization.quantize_fx import convert_fx, prepare_qat_fx

from fer.compress import int8, size_bytes, speed
from fer.metrics import scores
from fer.models import build
from fer.runs import RUNS, record_path
from fer.train import GPUData, class_weights, param_groups, save

p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int, default=5)
p.add_argument("--lr-scale", type=float, default=0.05)
p.add_argument("--skip-repaired", action="store_true")
args = p.parse_args()
warnings.filterwarnings("ignore")
torch.backends.quantized.engine = "x86"


@torch.no_grad()
def cpu_scores(model, split):
    x, y = data.batch(data.idx[split])
    x, y = x.cpu().contiguous(), y.cpu()
    probs = torch.cat([model(x[k:k + 512]).float().softmax(1) for k in range(0, len(x), 512)])
    s = scores(probs, y)
    return {"accuracy": s["accuracy"], "macro_f1": s["macro_f1"]}


def finish(model, record, extra):
    r = {**extra, "bytes": size_bytes(model), "val": cpu_scores(model, "val"), "test": cpu_scores(model, "test"),
         "speed": speed(model, x1, x256, gpu=False), "status": "done", "finished": time.time()}
    save(r, record)
    return r


data = GPUData()
g0 = torch.Generator().manual_seed(0)
perm = torch.randperm(len(data.idx["train"]), generator=g0).to(data.idx["train"].device)
x_fit = data.batch(data.idx["train"][perm[:1024]])[0].cpu().contiguous()
x_val = data.batch(data.idx["val"])[0].cpu().contiguous()
x1, x256 = x_val[:1].clone(), x_val[:256].clone()

# 1. int8 on top of the repaired models
if not args.skip_repaired:
    for path in sorted((RUNS / "repair").glob("*/*.pt")):
        name, iteration = path.parent.name, path.stem
        record = record_path("quant", name, f"{iteration}_int8")
        if record.exists():
            continue
        q = int8(torch.load(path, weights_only=False, map_location="cpu"), x_fit,
                 keep_float=(torch.cat,) if name == "densenet" else ())
        r = finish(q, record, {"config": {"model": name, "stage": "quant", "iteration": f"{iteration}_int8"}})
        print(f"{name} {iteration} + int8: {r['bytes'] / 2**20:.1f} MB, val {100 * r['val']['accuracy']:.1f}, "
              f"CPU 1 face {r['speed']['cpu_1_ms']:.2f} ms", flush=True)

# 2. quantisation-aware training for DenseNet
rec = json.loads((RUNS / "cooldown" / "densenet" / "larger_e100.json").read_text())
cfg = rec["config"]
base = build("densenet", **{k: cfg[k] for k in ("dropout", "width", "depth", "stages") if k in cfg})
base.load_state_dict(torch.load(RUNS / "cooldown" / "densenet" / "larger_e100.ckpt", weights_only=False, map_location="cpu")["model"])
for keep_float, label in [((), "qat"), ((torch.cat,), "qat_cat_float")]:
    record = record_path("quant", "densenet", label)
    if record.exists() and json.loads(record.read_text()).get("status") == "done":
        continue
    mapping = get_default_qat_qconfig_mapping("x86")
    for op in keep_float:
        mapping.set_object_type(op, None)
    model = prepare_qat_fx(copy.deepcopy(base).train(), mapping, (x1,)).cuda()
    opt = torch.optim.AdamW(param_groups(model, cfg["weight_decay"]), lr=cfg["lr"] * args.lr_scale)
    g = torch.Generator(device="cuda").manual_seed(cfg["seed"])
    train_idx, bs = data.idx["train"], cfg["batch_size"]
    steps, weights = len(train_idx) // bs, class_weights(data.label[train_idx], cfg["class_weights"])
    lr0, total, step = cfg["lr"] * args.lr_scale, steps * args.epochs, 0
    state = {"config": {**cfg, "stage": "quant", "iteration": label, "epochs": args.epochs, "lr_scale": args.lr_scale},
             "status": "running", "started": time.time(), "history": []}
    for epoch in range(args.epochs):
        frozen = epoch >= args.epochs - 2
        model.train()
        if frozen:
            model.apply(disable_observer)
            model.apply(nniqat.freeze_bn_stats)
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        t0 = time.perf_counter()
        xs, ys = data.batch(train_idx[torch.randperm(len(train_idx), device="cuda", generator=g)], cfg["augment"], g)
        for s in range(steps):
            for group in opt.param_groups:
                group["lr"] = lr0 * (1 - step / total)
            x, y = xs[s * bs:(s + 1) * bs].contiguous(), ys[s * bs:(s + 1) * bs]
            loss = F.cross_entropy(model(x), y, weight=weights, label_smoothing=cfg["label_smoothing"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
        model.eval()
        with torch.no_grad():
            xv, yv = data.batch(data.idx["val"])
            probs = torch.cat([model(xv[k:k + 512].contiguous()).softmax(1) for k in range(0, len(xv), 512)])
        val = scores(probs, yv)
        state["history"].append({"epoch": epoch + 1, "val_accuracy": val["accuracy"], "val_macro_f1": val["macro_f1"],
                                 "seconds": time.perf_counter() - t0})
        save(state, record)
    q = convert_fx(copy.deepcopy(model).cpu().eval())
    r = finish(q, record, state)
    print(f"densenet {label}: fake-quant val {100 * state['history'][-1]['val_accuracy']:.1f}, int8 val "
          f"{100 * r['val']['accuracy']:.1f}, {r['bytes'] / 2**20:.1f} MB, CPU 1 face {r['speed']['cpu_1_ms']:.2f} ms", flush=True)
