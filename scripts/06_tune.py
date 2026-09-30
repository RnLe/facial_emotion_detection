"""Hyperparameter tuning with Optuna, the same budget for every model.

The search ranges are educated guesses centred on each model's defaults: the learning rate
from a tenth to ten times the default, the other settings over the range people use in
practice. The default settings go in as the first trial, so every study contains its
baseline. TPE picks the settings, after 10 random trials that sample the space without
bias (used for the spread across settings).

Every trial runs the same shortened schedule (25 epochs) to the end, without pruning: the
question is how sensitive a model is to its settings, and that needs trials that can be
compared. The best settings are then retrained for the full 60 epochs (07_final.py).
Objective: the best validation macro-F1 of a run. The test set is never touched here.

Workers can share one study: run the script twice with different --worker numbers.
"""
import argparse
import json

import optuna
import torch

from fer.models import MODELS
from fer.runs import RUNS, add_to_plan, record_path
from fer.train import GPUData, config, run

p = argparse.ArgumentParser()
p.add_argument("--models", nargs="+", help="default: all, the best baselines first")
p.add_argument("--trials", type=int, default=30)
p.add_argument("--worker", type=int, default=0)
p.add_argument("--workers", type=int, default=1)
p.add_argument("--epochs", type=int, default=25)
args = p.parse_args()


def by_baseline(models):
    """Models ordered by their mean baseline validation accuracy, best first, so the
    strongest models are tuned first (validation only; the test set plays no part)."""
    acc = {}
    for m in models:
        runs = [json.loads(f.read_text()) for f in (RUNS / "baseline" / m).glob("*.json")]
        done = [r["val"]["accuracy"] for r in runs if r.get("status") == "done"]
        acc[m] = sum(done) / len(done) if done else 0.0
    return sorted(models, key=lambda m: -acc[m])


args.models = args.models or by_baseline([m for m in MODELS if m != "resnet_pretrained"])


def suggest(trial, model):
    d = config(model, epochs=args.epochs)
    cfg = {
        **d,
        "lr": trial.suggest_float("lr", d["lr"] / 10, d["lr"] * 10, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 5e-4, 0.5, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.2),
        "augment": trial.suggest_float("augment", 0.0, 1.5),
        "class_weights": trial.suggest_categorical("class_weights", ["none", "sqrt", "inverse"]),
        "warmup_epochs": trial.suggest_int("warmup_epochs", 0, 10),
    }
    if "drop_path" in d:
        cfg["drop_path"] = trial.suggest_float("drop_path", 0.0, 0.3)
    else:
        cfg["dropout"] = trial.suggest_float("dropout", 0.0, 0.6)
    return cfg


def defaults(model):
    d = config(model)
    out = {k: d[k] for k in ("lr", "weight_decay", "batch_size", "label_smoothing", "augment", "class_weights", "warmup_epochs")}
    out["drop_path" if "drop_path" in d else "dropout"] = d.get("drop_path", d.get("dropout"))
    return out


torch.cuda.set_per_process_memory_fraction(0.85 / args.workers)
data = GPUData()
(RUNS / "optuna").mkdir(parents=True, exist_ok=True)
for model in args.models:
    add_to_plan([{**config(model, epochs=args.epochs), "stage": "tune", "iteration": k} for k in range(args.trials)])
    storage = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(str(RUNS / "optuna" / f"{model}.log")))
    study = optuna.create_study(
        study_name=model,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(n_startup_trials=10, multivariate=True, seed=args.worker),
        pruner=optuna.pruners.NopPruner(),
    )
    if args.worker == 0 and len(study.trials) == 0:
        study.enqueue_trial(defaults(model))
    if args.workers == 1:  # a trial still "running" was cut off by a crash: count it as failed
        for t in study.trials:
            if t.state == optuna.trial.TrialState.RUNNING:
                study._storage.set_trial_state_values(t._trial_id, optuna.trial.TrialState.FAIL)

    def objective(trial):
        cfg = {**suggest(trial, model), "stage": "tune", "iteration": trial.number}
        torch.cuda.empty_cache()
        result, _ = run(cfg, data=data, trial=trial, log=None, record=record_path("tune", model, trial.number))
        return result["best_val_macro_f1"]

    ran = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED, optuna.trial.TrialState.RUNNING)
    left = args.trials - len([t for t in study.trials if t.state in ran])
    if left > 0:
        study.optimize(objective, n_trials=max(1, left // args.workers + (args.worker < left % args.workers)))
    print(f"{model}: best val F1 {study.best_value:.3f} with {study.best_params}", flush=True)
