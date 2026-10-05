"""Known methods, one at a time, on the ResNet-18 winner (part two, phase 3).

Each condition changes one thing against the stage 1 runs and trains 3 seeds; decisions
use the validation set. Scratch track (our data only):
    votes          FER+ vote shares as soft targets for the FER2013 faces
    logit_adjust   logit-adjusted loss (Menon et al. 2021, tau 1) instead of class weights
    ema            an average of the weights (decay 0.999, about the last 4 epochs)
    sam            sharpness-aware minimisation (Foret et al. 2021, rho 0.05)
    clean          without the training faces confident learning flagged (scripts/19_errors.py)
    self_distill   distilled from the ensemble of our nine stage runs (T 4, alpha 0.5)
Open track (outside data through the teacher):
    distill_fmae   distilled from FMAE fine-tuned on our data (scripts/16_references.py)
tta: flip test-time augmentation on the existing stage 1 runs (no training).
teacher: caches the nine-run ensemble's log-probabilities for every face.
annotators: every condition judged by the annotators instead of the majority label, on
    the FER2013 faces of validation and test: expected agreement with one random
    annotator (E[p_pred]), cross-entropy against the vote shares, and calibration (ECE).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from fer.metrics import scores
from fer.models import build
from fer.runs import RUNS, add_to_plan, is_done, record_path
from fer.train import GPUData, predict, run

WINNERS = {"vgg": "shape_final", "resnet": "final", "densenet": "shape_final"}
MODEL_ARGS = ("dropout", "drop_path", "width", "depth", "stages")
ENSEMBLE = RUNS / "opt" / "ensemble_logprob.pt"
CONDITIONS = {
    "votes": {"targets": "votes"},
    "logit_adjust": {"logit_adjust": 1.0, "class_weights": "none"},
    "ema": {"ema": 0.999},
    "sam": {"sam": 0.05},
    "clean": {"clean": True},
    "self_distill": {"teacher": str(ENSEMBLE), "kd_alpha": 0.5, "kd_temperature": 4.0},
    "distill_fmae": {"teacher": str(RUNS / "references" / "fmae" / "finetune_logprob.pt"), "kd_alpha": 0.5, "kd_temperature": 4.0},
}


def winner_config(m, seed=0):
    cfg = json.loads(record_path(WINNERS[m], m, seed).read_text())["config"]
    return {k: v for k, v in cfg.items() if k not in ("stage", "iteration")}


def load_winner(m, seed):
    cfg = winner_config(m, seed)
    model = build(m, **{k: cfg[k] for k in MODEL_ARGS if k in cfg}).cuda().to(memory_format=torch.channels_last)
    model.load_state_dict(torch.load(record_path(WINNERS[m], m, seed).with_suffix(".pt"), map_location="cuda"))
    return model


def merge(key, value):
    path = Path("results/optimize.json")
    res = json.loads(path.read_text()) if path.exists() else {}
    res[key] = value
    path.write_text(json.dumps(res, indent=1))


def summary(rows):
    keys = rows[0].keys()
    return {k: {"mean": float(np.mean([r[k] for r in rows])), "std": float(np.std([r[k] for r in rows]))} for k in keys}


def row(r):
    return {"val": r["val"]["accuracy"], "val_macro_f1": r["val"]["macro_f1"], "test": r["test"]["accuracy"],
            "test_macro_f1": r["test"]["macro_f1"], "test_fer": r["test_fer"]["accuracy"], "test_raf": r["test_raf"]["accuracy"]}


def teacher():
    data = GPUData()
    data.idx["all"] = torch.arange(len(data.label), device="cuda")
    probs = []
    for m in WINNERS:
        for seed in range(3):
            probs.append(predict(load_winner(m, seed), data, "all")[0])
    p = torch.stack(probs).mean(0)
    ENSEMBLE.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"dataset": p.clamp_min(1e-8).log().half().cpu()}, ENSEMBLE)
    train = data.idx["train"]
    print("ensemble on its own training faces:", float((p[train].argmax(1) == data.label[train]).float().mean()))


def tta():
    data = GPUData()
    rows = {"plain": [], "tta": []}
    for seed in range(3):
        model = load_winner("resnet", seed)
        for name, flag in [("plain", False), ("tta", True)]:
            out = {}
            for split in ["val", "test"]:
                prob, true = predict(model, data, split, tta=flag)
                out[split] = scores(prob, true)
                src = data.source[data.idx[split]]
                for s_name, k in [("fer", 0), ("raf", 1)]:
                    out[f"{split}_{s_name}"] = scores(prob[src == k], true[src == k])
            rows[name].append(row(out))
    return {k: summary(v) for k, v in rows.items()}


def condition(name, seeds):
    data = GPUData()
    extra = dict(CONDITIONS[name])
    if extra.pop("clean", False):
        flagged = torch.tensor(np.load(RUNS / "errors" / "flagged.npy"), device="cuda")
        data.idx["train"] = data.idx["train"][~torch.isin(data.idx["train"], flagged)]
    add_to_plan([{"stage": "opt", "model": "resnet", "iteration": f"{name}_{s}", "epochs": 60} for s in seeds])
    rows = []
    for seed in seeds:
        rec = record_path("opt", "resnet", f"{name}_{seed}")
        if not is_done("opt", "resnet", f"{name}_{seed}"):
            cfg = {**winner_config("resnet", seed), "seed": seed, **extra, "stage": "opt", "iteration": f"{name}_{seed}"}
            run(cfg, data=data, test=True, record=rec)
        rows.append(row(json.loads(rec.read_text())))
        print(name, seed, rows[-1])
    base = [row(json.loads(record_path("final", "resnet", s).read_text())) for s in seeds]
    return {"runs": rows, "summary": summary(rows), "baseline": summary(base)}


def annotators():
    data = GPUData()
    votes = data.vote_targets()
    out = {}
    for name in ["baseline"] + list(CONDITIONS):
        rows = []
        for seed in range(3):
            rec = record_path("final", "resnet", seed) if name == "baseline" else record_path("opt", "resnet", f"{name}_{seed}")
            if not rec.with_suffix(".pt").exists():
                break
            model = build("resnet", **{k: v for k, v in json.loads(rec.read_text())["config"].items() if k in MODEL_ARGS}).cuda().to(memory_format=torch.channels_last)
            model.load_state_dict(torch.load(rec.with_suffix(".pt"), map_location="cuda"))
            row = {}
            for split in ["val", "test"]:
                prob, true = predict(model, data, split)
                fer = data.source[data.idx[split]] == 0
                p, v = prob[fer], votes[data.idx[split]][fer]
                row[f"{split}_agreement"] = float(v[torch.arange(len(p)), p.argmax(1)].mean())
                row[f"{split}_vote_ce"] = float(-(v * p.clamp_min(1e-8).log()).sum(1).mean())
                row[f"{split}_ece"] = scores(p, true[fer])["ece"]
            rows.append(row)
        if rows:
            out[name] = summary(rows)
            print(name, {k: round(v["mean"], 4) for k, v in out[name].items()})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["teacher", "tta", "annotators"] + list(CONDITIONS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    if a.step == "teacher":
        teacher()
    elif a.step == "tta":
        merge("tta", tta())
    elif a.step == "annotators":
        merge("annotators", annotators())
    else:
        merge(a.step, condition(a.step, a.seeds))
