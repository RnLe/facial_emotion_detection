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

Stage 2 (--stage shape): the three strongest baselines again, now with the network's
shape in the search too: width, depth and number of stages (SHAPES). It starts where stage
1 ended. Batch size and class weights stay at the stage 1 best (stage 1 settled them), the
other training settings are searched over the same ranges as before, and the first trial
is the stage 1 best with the default shape, so the gain over it is what the shape adds.
40 trials, the first 12 random: the shape settings interact with each other and with the
learning rate (a wider network usually wants a smaller one), so TPE needs a few more.

Workers can share one study: run the script twice with different --worker numbers.
"""
import argparse
import inspect
import json

import optuna
import torch

from fer.models import MODELS
from fer.runs import RUNS, add_to_plan, record_path
from fer.train import GPUData, config, run

p = argparse.ArgumentParser()
p.add_argument("--stage", choices=["tune", "shape"], default="tune")
p.add_argument("--models", nargs="+", help="default: all, the best baselines first (stage 2: the best three)")
p.add_argument("--trials", type=int, help="default: 30 (stage 2: 40)")
p.add_argument("--worker", type=int, default=0)
p.add_argument("--workers", type=int, default=1)
p.add_argument("--epochs", type=int, default=25)
args = p.parse_args()
shape_stage = args.stage == "shape"
args.trials = args.trials or (40 if shape_stage else 30)

# Stage 2 ranges for the network's shape. ResNet-18 and VGG are already large (11 and 7 M
# parameters), so their width goes mostly down; DenseNet is small (0.8 M), so both ways.
SHAPES = {
    "resnet": {"width": (0.25, 1.5), "depth": (1, 3), "stages": (3, 4)},  # depth: blocks per stage
    "vgg": {"width": (0.25, 1.5), "depth": (1, 3), "stages": (3, 4)},  # depth: convs per stage
    "densenet": {"width": (0.5, 2.0), "depth": (6, 20), "stages": (2, 3)},  # growth 6 to 24, layers per block
}


def by_baseline(models):
    """Models ordered by their mean baseline validation accuracy, best first, so the
    strongest models are tuned first (validation only; the test set plays no part)."""
    acc = {}
    for m in models:
        runs = [json.loads(f.read_text()) for f in (RUNS / "baseline" / m).glob("*.json")]
        done = [r["val"]["accuracy"] for r in runs if r.get("status") == "done"]
        acc[m] = sum(done) / len(done) if done else 0.0
    return sorted(models, key=lambda m: -acc[m])


args.models = args.models or by_baseline([m for m in MODELS if m != "resnet_pretrained"])[: 3 if shape_stage else None]


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


def suggest_shape(trial, model, best):
    d = config(model, epochs=args.epochs)
    cfg = {
        **d,
        "batch_size": best["batch_size"],
        "class_weights": best["class_weights"],
        "lr": trial.suggest_float("lr", d["lr"] / 10, d["lr"] * 10, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 5e-4, 0.5, log=True),
        "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.2),
        "augment": trial.suggest_float("augment", 0.0, 1.5),
        "warmup_epochs": trial.suggest_int("warmup_epochs", 0, 10),
        "dropout": trial.suggest_float("dropout", 0.0, 0.6),
    }
    for k, (lo, hi) in SHAPES[model].items():
        cfg[k] = trial.suggest_float(k, lo, hi, log=True) if k == "width" else trial.suggest_int(k, lo, hi)
    return cfg


def defaults(model):
    d = config(model)
    out = {k: d[k] for k in ("lr", "weight_decay", "batch_size", "label_smoothing", "augment", "class_weights", "warmup_epochs")}
    out["drop_path" if "drop_path" in d else "dropout"] = d.get("drop_path", d.get("dropout"))
    return out


def storage(name):
    return optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(str(RUNS / "optuna" / f"{name}.log")))


def start_shape(model, best):
    """Stage 2's first trial: the stage 1 best with the model's default shape."""
    shape = {k: inspect.signature(MODELS[model]).parameters[k].default for k in SHAPES[model]}
    return {**{k: v for k, v in best.items() if k not in ("batch_size", "class_weights")}, **shape}


torch.cuda.set_per_process_memory_fraction(0.85 / args.workers)
data = GPUData()
(RUNS / "optuna").mkdir(parents=True, exist_ok=True)
for model in args.models:
    name = f"{model}_shape" if shape_stage else model
    best = optuna.load_study(study_name=model, storage=storage(model)).best_params if shape_stage else None
    add_to_plan([{**config(model, epochs=args.epochs), "stage": args.stage, "iteration": k} for k in range(args.trials)])
    study = optuna.create_study(
        study_name=name,
        storage=storage(name),
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(n_startup_trials=12 if shape_stage else 10, multivariate=True, seed=args.worker),
        pruner=optuna.pruners.NopPruner(),
    )
    if args.worker == 0 and len(study.trials) == 0:
        study.enqueue_trial(start_shape(model, best) if shape_stage else defaults(model))
    if args.workers == 1:  # a trial still "running" was cut off by a crash: count it as failed
        for t in study.trials:
            if t.state == optuna.trial.TrialState.RUNNING:
                study._storage.set_trial_state_values(t._trial_id, optuna.trial.TrialState.FAIL)

    def objective(trial):
        cfg = suggest_shape(trial, model, best) if shape_stage else suggest(trial, model)
        cfg = {**cfg, "stage": args.stage, "iteration": trial.number}
        torch.cuda.empty_cache()
        result, _ = run(cfg, data=data, trial=trial, log=None, record=record_path(args.stage, model, trial.number))
        trial.set_user_attr("params", result["params"])
        return result["best_val_macro_f1"]

    ran = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED, optuna.trial.TrialState.RUNNING)
    left = args.trials - len([t for t in study.trials if t.state in ran])
    if left > 0:
        study.optimize(objective, n_trials=max(1, left // args.workers + (args.worker < left % args.workers)))
    print(f"{name}: best val F1 {study.best_value:.3f} with {study.best_params}", flush=True)
