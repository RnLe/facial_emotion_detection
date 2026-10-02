"""Same-size networks trained from scratch: the fair baseline for the repaired compressed
models (Liu et al. 2019 found that small networks trained from scratch often match pruned
ones). For each model and budget (4x and 8x fewer parameters than the larger long-run
model), the width whose parameter count comes closest to the budget, with the stage 2
depth and stages and the larger run's training settings, trained with the stage finals'
recipe (60 epochs, warmup and cosine decay, the best validation epoch, one seed). Scored on
validation and test. Records in runs/scratch/<model>/<budget>x.json.
"""
import argparse
import json

from fer.models import build
from fer.runs import RUNS, add_to_plan, is_done, record_path
from fer.train import GPUData, run

p = argparse.ArgumentParser()
p.add_argument("--models", nargs="+", default=["vgg", "resnet", "densenet"])
p.add_argument("--budgets", nargs="+", type=float, default=[4, 8])
p.add_argument("--epochs", type=int, default=60)
args = p.parse_args()


def params(name, **kw):
    return sum(p.numel() for p in build(name, **kw).parameters())


def matched_width(name, cfg, target):
    """Bisection on the width multiplier for the parameter count closest to target."""
    shape = {k: cfg[k] for k in ("depth", "stages") if k in cfg}
    lo, hi = 0.05, cfg["width"]
    for _ in range(30):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if params(name, width=mid, **shape) < target else (lo, mid)
    return min((lo, hi), key=lambda w: abs(params(name, width=w, **shape) - target))


plans = []
for name in args.models:
    cfg = json.loads((RUNS / "cooldown" / name / "larger_e100.json").read_text())["config"]
    total = params(name, **{k: cfg[k] for k in ("width", "depth", "stages")})
    for budget in args.budgets:
        width = matched_width(name, cfg, total / budget)
        keep = {k: v for k, v in cfg.items() if k not in ("schedule", "iteration", "stage", "epochs", "width")}
        plans.append({**keep, "width": width, "epochs": args.epochs, "stage": "scratch", "iteration": f"{budget:g}x",
                      "budget": budget, "target_params": round(total / budget)})
add_to_plan(plans)

data = GPUData()
for cfg in plans:
    record = record_path("scratch", cfg["model"], cfg["iteration"])
    if is_done("scratch", cfg["model"], cfg["iteration"]):
        continue
    result, _ = run(cfg, data, test=True, record=record, log=None)
    print(f"{cfg['model']} {cfg['iteration']} from scratch: width {cfg['width']:.3f}, {result['params'] / 1e6:.2f} M "
          f"(target {cfg['target_params'] / 1e6:.2f} M), val {100 * result['val']['accuracy']:.1f}, "
          f"test {100 * result['test']['accuracy']:.1f}", flush=True)
