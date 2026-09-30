"""Final runs: each model's best tuned settings, retrained with three seeds, scored once on
the test set. Together with the baselines this gives the gain from tuning per model."""
import argparse

import optuna
import torch

from fer.models import MODELS
from fer.runs import RUNS, add_to_plan, is_done, record_path
from fer.train import GPUData, config, run

p = argparse.ArgumentParser()
p.add_argument("--models", nargs="+", default=[m for m in MODELS if m != "resnet_pretrained"])
p.add_argument("--seeds", type=int, default=3)
p.add_argument("--worker", type=int, default=0)
p.add_argument("--workers", type=int, default=1)
args = p.parse_args()


def best_config(model, seed):
    storage = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(str(RUNS / "optuna" / f"{model}.log")))
    best = optuna.load_study(study_name=model, storage=storage).best_params
    return config(model, **best, seed=seed, stage="final", iteration=seed)


def tuned(model):
    try:
        storage = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(str(RUNS / "optuna" / f"{model}.log")))
        study = optuna.load_study(study_name=model, storage=storage)
    except (KeyError, FileNotFoundError):
        return False
    return len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]) >= 30


args.models = [m for m in args.models if tuned(m)]
print("final runs for", args.models, flush=True)
jobs = [(m, s) for s in range(args.seeds) for m in args.models]
add_to_plan([{**config(m, seed=s), "stage": "final", "iteration": s} for m, s in jobs])
torch.cuda.set_per_process_memory_fraction(0.85 / args.workers)
data = GPUData()
for k, (model, seed) in enumerate(jobs):
    if k % args.workers != args.worker or is_done("final", model, seed):
        continue
    torch.cuda.empty_cache()
    result, _ = run(best_config(model, seed), data=data, test=True, log=None, record=record_path("final", model, seed))
    print(f"{model} seed {seed}: test acc {result['test']['accuracy']:.3f}, test F1 {result['test']['macro_f1']:.3f}", flush=True)
