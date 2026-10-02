"""Compression of the three larger long-run models (after their cooldown), each method on its
own and without fine-tuning: size, speed and validation scores.

- baseline: the model as trained, fp32. bn_folded: the same with batch norm folded into the
  convs before it, the fair reference for int8 (which folds it too).
- low_rank, tucker2, spatial: swept over tau, the share of output variance (low_rank) or of
  squared singular values (tucker2, spatial) each layer keeps. See fer/compress.py.
- int8: CPU only; PyTorch has no int8 kernels for the GPU without extra libraries. For
  DenseNet also int8_cat_float, with the concatenations left in float.
- baseline_end: the baseline timed again after the sweep, to see how much the timings drift
  (other programs share the machine).

Speed: median time for one face and for 256 faces on the GPU (fp32, eager) and on the CPU
(8 threads, fp32), plus one face on one CPU thread (closest to a browser). Projections and
the int8 calibration use 4,096 training faces; scores are on validation. The test set is
not touched here.
"""
import argparse
import json
import warnings
from pathlib import Path

import torch
from torch.fx.experimental.optimization import fuse

from fer.compress import compress, int8, macs, output_stats, size_bytes, speed
from fer.metrics import scores
from fer.models import build
from fer.runs import RUNS
from fer.train import GPUData

p = argparse.ArgumentParser()
p.add_argument("--models", nargs="+", default=["vgg", "resnet", "densenet"])
p.add_argument("--methods", nargs="+", default=["low_rank", "tucker2", "spatial"])
p.add_argument("--taus", nargs="+", type=float, default=[0.999, 0.995, 0.99, 0.98, 0.95, 0.9, 0.8])
p.add_argument("--out", default="results/compress.json")
args = p.parse_args()
torch.backends.cudnn.benchmark = True
warnings.filterwarnings("ignore")  # PyTorch's FX quantisation is deprecated and says so at length


def load(model):
    rec = json.loads((RUNS / "cooldown" / model / "larger_e100.json").read_text())
    cfg = rec["config"]
    net = build(model, **{k: cfg[k] for k in ("dropout", "width", "depth", "stages") if k in cfg})
    state = torch.load(RUNS / "cooldown" / model / "larger_e100.ckpt", weights_only=False, map_location="cpu")["model"]
    net.load_state_dict(state)
    return net.eval()


@torch.inference_mode()
def score(model, device):
    model = model.to(device).eval()
    probs = torch.cat([model(x_val[k:k + 512].to(device)).float().softmax(1).cpu() for k in range(0, len(x_val), 512)])
    s = scores(probs, y_val)
    return {"val_accuracy": s["accuracy"], "val_macro_f1": s["macro_f1"]}


def row(model, gpu=True, **extra):
    r = {**extra, "params": sum(p.numel() for p in model.parameters()), "bytes": size_bytes(model)}
    if gpu:
        r["macs"] = macs(model.cpu(), x1)
    r.update(score(model, "cuda" if gpu else "cpu"))
    r.update(speed(model, x1, x256, gpu))
    return r


data = GPUData()
g = torch.Generator().manual_seed(0)
fit_idx = data.idx["train"][torch.randperm(len(data.idx["train"]), generator=g)[:4096].to(data.idx["train"].device)]
x_fit = data.batch(fit_idx)[0].contiguous()
x_val, y_val = data.batch(data.idx["val"])
x_val, y_val = x_val.cpu().contiguous(), y_val.cpu()
x1, x256 = x_val[:1].clone(), x_val[:256].clone()

results = json.loads(Path(args.out).read_text()) if Path(args.out).exists() else {}
for name in args.models:
    base = load(name).cuda()
    res = results.setdefault(name, {})
    if "baseline" not in res:
        res["baseline"] = row(base)
        res["bn_folded"] = row(fuse(load(name)))
        res["int8"] = row(int8(base, x_fit[:1024]), gpu=False, macs=res["baseline"]["macs"])
        if name == "densenet":
            res["int8_cat_float"] = row(int8(base, x_fit[:1024], keep_float=(torch.cat,)), gpu=False, macs=res["baseline"]["macs"])
        print(name, "baseline", {k: round(v, 3) for k, v in res["baseline"].items()}, flush=True)
    stats = output_stats(base.cuda(), x_fit)
    for method in args.methods:
        res[method] = [r for r in res.get(method, []) if r["tau"] not in args.taus]  # keep other taus
        for tau in args.taus:
            small, replaced = compress(base.cuda(), method, tau, stats)
            r = row(small, tau=tau, replaced=replaced)
            res[method].append(r)
            res[method].sort(key=lambda r: -r["tau"])
            print(f"{name} {method} tau {tau}: params {r['params'] / 1e6:.2f} M, MACs {r['macs'] / 1e6:.0f} M, "
                  f"val {100 * r['val_accuracy']:.1f}, cpu 1 {r['cpu_1_ms']:.1f} ms, gpu 1 {r['gpu_1_ms']:.2f} ms", flush=True)
    res["baseline_end"] = row(load(name))
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
