"""Final runs: each model's best tuned settings, retrained with three seeds, scored once on
the test set. Together with the baselines this gives the gain from tuning per model.

Stage 2 (--stage shape) does the same for the shape studies, with two picks per model: the
best trial, and the smallest network within one point (validation macro-F1) of the best,
to see whether a much smaller network holds up on the test set. Both are picked on the
validation set. The second pick is only retrained if it is smaller than the default
network; otherwise the stage 1 final runs already cover it."""
import argparse
import time

import optuna
import torch

from fer.models import MODELS
from fer.runs import RUNS, add_to_plan, is_done, record_path
from fer.train import GPUData, config, run

p = argparse.ArgumentParser()
p.add_argument("--stage", choices=["tune", "shape"], default="tune")
p.add_argument("--models", nargs="+", default=[m for m in MODELS if m != "resnet_pretrained"])
p.add_argument("--seeds", type=int, default=3)
p.add_argument("--trials", type=int, help="trials a study needs to count as tuned (default 30, stage 2: 40)")
p.add_argument("--worker", type=int, default=0)
p.add_argument("--workers", type=int, default=1)
args = p.parse_args()
shape_stage = args.stage == "shape"
args.trials = args.trials or (40 if shape_stage else 30)


def study(name):
    storage = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(str(RUNS / "optuna" / f"{name}.log")))
    return optuna.load_study(study_name=name, storage=storage)


def tuned(name):
    try:
        s = study(name)
    except (KeyError, FileNotFoundError):
        return False
    return len([t for t in s.trials if t.state == optuna.trial.TrialState.COMPLETE]) >= args.trials


def compact(s):
    """The smallest network within one point of the best validation macro-F1."""
    done = [t for t in s.trials if t.state == optuna.trial.TrialState.COMPLETE]
    return min((t for t in done if t.value >= s.best_value - 0.01), key=lambda t: (t.user_attrs["params"], -t.value))


def picks(model):
    """The settings to retrain for one model, as (stage, settings)."""
    if not shape_stage:
        return [("final", study(model).best_params)]
    s, start = study(f"{model}_shape"), study(model).best_params  # batch size and class weights come from stage 1
    out = [("shape_final", {**start, **s.best_params})]
    small, default = compact(s), s.trials[0]  # trial 0: the stage 1 best with the default shape
    if small.number != s.best_trial.number and small.user_attrs["params"] < default.user_attrs["params"]:
        out.append(("shape_compact", {**start, **small.params}))
    return out


args.models = [m for m in args.models if tuned(f"{m}_shape" if shape_stage else m)]
print("final runs for", args.models, flush=True)
chosen = {m: picks(m) for m in args.models}
jobs = [(stage, m, s, settings) for s in range(args.seeds) for m in args.models for stage, settings in chosen[m]]
add_to_plan([{**config(m, **settings, seed=s), "stage": stage, "iteration": s} for stage, m, s, settings in jobs])
torch.cuda.set_per_process_memory_fraction(0.85 / args.workers)
data = GPUData()
for k, (stage, model, seed, settings) in enumerate(jobs):
    if k % args.workers != args.worker or is_done(stage, model, seed):
        continue
    record = record_path(stage, model, seed)
    if record.exists() and time.time() - record.stat().st_mtime < 600:  # another process is on it
        continue
    torch.cuda.empty_cache()
    cfg = config(model, **settings, seed=seed, stage=stage, iteration=seed)
    result, _ = run(cfg, data=data, test=True, log=None, record=record)
    print(f"{stage} {model} seed {seed}: test acc {result['test']['accuracy']:.3f}, test F1 {result['test']['macro_f1']:.3f}", flush=True)
