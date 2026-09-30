"""Baselines: every model with its educated defaults (configs/defaults.yaml), two seeds,
60 epochs, scored once on the test set with the weights of its best validation epoch.

Several workers can share the GPU: `--worker 0 --workers 2` and `--worker 1 --workers 2`
each take every second job. Finished runs are skipped, so the script can be restarted.
"""
import argparse

import torch

from fer.models import MODELS
from fer.runs import add_to_plan, is_done, record_path
from fer.train import GPUData, config, run

p = argparse.ArgumentParser()
p.add_argument("--worker", type=int, default=0)
p.add_argument("--workers", type=int, default=1)
p.add_argument("--seeds", type=int, default=2)
args = p.parse_args()

# seed first, then model: after the first round every model has one finished run
jobs = [(m, s) for s in range(args.seeds) for m in MODELS]
add_to_plan([{**config(m, seed=s), "stage": "baseline", "iteration": s} for m, s in jobs])

torch.cuda.set_per_process_memory_fraction(0.85 / args.workers)
data = GPUData()
for k, (model, seed) in enumerate(jobs):
    if k % args.workers != args.worker or is_done("baseline", model, seed):
        continue
    cfg = config(model, seed=seed, stage="baseline", iteration=seed)
    torch.cuda.empty_cache()
    result, _ = run(cfg, data=data, test=True, record=record_path("baseline", model, seed))
    print(f"{model} seed {seed}: val F1 {result['best_val_macro_f1']:.3f}, test acc {result['test']['accuracy']:.3f}, test F1 {result['test']['macro_f1']:.3f}", flush=True)
