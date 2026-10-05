"""Where the remaining errors come from (part two, phase 2, first half).

sanity: each family winner overfits 256 training faces with augmentation and every
    regulariser off. Failing to reach 100% would point to a defect, not to the data.
clean: the trained winners scored on the training set without augmentation, against
    validation, per source: how much of the training set is memorised.
sources: the ResNet-18 winner trained on one source only (FER2013 or RAF-DB), with as many
    optimiser steps as the run on both, selected on that source's validation share, then
    scored on both test parts.
curves: the ResNet-18 winner on 12.5, 25 and 50% of the training set (drawn per source
    and class), again with the same number of steps. With the full runs this gives a
    learning curve; err(n) = a n^-b + c estimates what more data of the same kind could do.
inputs: what our input format throws away. The ResNet-18 winner on RAF-DB alone (the
    rafdb.npz split) at its native 100 px, in grayscale and in color, once with a stride-2
    stem (about the compute of the 48 px runs) and once with a stride-1 stem (about 4x).
    Compared with the same network at 48 px grayscale (scripts/17_rafdb.py).

Records: runs/diag_<step>/<model>/<name>.json; summary in results/diagnose.json.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from fer.data import load_rafdb
from fer.metrics import scores
from fer.models import build
from fer.runs import add_to_plan, is_done, record_path
from fer.train import GPUData, predict, run

WINNERS = {"vgg": "shape_final", "resnet": "final", "densenet": "shape_final"}
MODEL_ARGS = ("dropout", "drop_path", "width", "depth", "stages")
FRACTIONS = [0.125, 0.25, 0.5]


def winner_config(m, seed=0):
    cfg = json.loads(record_path(WINNERS[m], m, seed).read_text())["config"]
    return {k: v for k, v in cfg.items() if k not in ("stage", "iteration")}


def subset(data, train=None, val=None):
    d = copy.copy(data)
    d.idx = dict(data.idx)
    if train is not None:
        d.idx["train"] = train
    if val is not None:
        d.idx["val"] = val
    return d


def merge(key, value):
    path = Path("results/diagnose.json")
    res = json.loads(path.read_text()) if path.exists() else {}
    res[key] = value
    path.write_text(json.dumps(res, indent=1))


def train_once(stage, model, name, cfg, data, test=True):
    rec = record_path(stage, model, name)
    if not is_done(stage, model, name):
        run({**cfg, "stage": stage, "iteration": name}, data=data, test=test, record=rec)
    return json.loads(rec.read_text())


def sanity(data):
    g = torch.Generator(device="cuda").manual_seed(0)
    few = data.idx["train"][torch.randperm(len(data.idx["train"]), device="cuda", generator=g)[:256]]
    d = subset(data, train=few)
    out = {}
    for m in WINNERS:
        cfg = {**winner_config(m), "augment": 0.0, "dropout": 0.0, "weight_decay": 0.0, "label_smoothing": 0.0,
               "class_weights": "none", "epochs": 300, "warmup_epochs": 5, "keep_last": True}
        r = train_once("diag_sanity", m, "256", cfg, d, test=False)
        hist = r["history"]
        first = next((h["epoch"] for h in hist if h["train_accuracy"] >= 0.999), None)
        out[m] = {"final_train_accuracy": hist[-1]["train_accuracy"], "first_epoch_at_100": first,
                  "final_train_loss": hist[-1]["train_loss"], "val_accuracy": hist[-1]["val_accuracy"]}
        print(m, out[m])
    return out


def clean(data):
    out = {}
    for m, stage in WINNERS.items():
        rows = []
        for seed in range(3):
            cfg = winner_config(m, seed)
            model = build(m, **{k: cfg[k] for k in MODEL_ARGS if k in cfg}).cuda().to(memory_format=torch.channels_last)
            model.load_state_dict(torch.load(record_path(stage, m, seed).with_suffix(".pt"), map_location="cuda"))
            row = {}
            for split in ["train", "val"]:
                prob, true = predict(model, data, split)
                src = data.source[data.idx[split]]
                row[split] = scores(prob, true)["accuracy"]
                for name, k in [("fer", 0), ("raf", 1)]:
                    row[f"{split}_{name}"] = scores(prob[src == k], true[src == k])["accuracy"]
            rows.append(row)
        out[m] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        print(m, {k: round(v, 4) for k, v in out[m].items()})
    return out


def step_matched(cfg, n_sub, n_all):
    ratio = n_all / n_sub
    return {**cfg, "epochs": round(cfg["epochs"] * ratio), "warmup_epochs": round(cfg["warmup_epochs"] * ratio)}


def sources(data, seeds):
    n_all = len(data.idx["train"])
    src_train, src_val = data.source[data.idx["train"]], data.source[data.idx["val"]]
    out = {}
    for name, k in [("fer_only", 0), ("raf_only", 1)]:
        train, val = data.idx["train"][src_train == k], data.idx["val"][src_val == k]
        rows = []
        for seed in seeds:
            cfg = step_matched({**winner_config("resnet", seed), "seed": seed}, len(train), n_all)
            r = train_once("diag_sources", "resnet", f"{name}_{seed}", cfg, subset(data, train, val))
            rows.append({"test_fer": r["test_fer"]["accuracy"], "test_raf": r["test_raf"]["accuracy"], "val": r["val"]["accuracy"]})
            print(name, seed, rows[-1])
        out[name] = rows
    both = [json.loads(record_path("final", "resnet", s).read_text()) for s in seeds]
    out["both"] = [{"test_fer": r["test_fer"]["accuracy"], "test_raf": r["test_raf"]["accuracy"], "val": r["val"]["accuracy"]} for r in both]
    return out


def inputs(seeds):
    d = np.load("data/processed/rafdb.npz")
    out = {}
    for color in [False, True]:
        images = load_rafdb(color=color)["images"]
        data = GPUData(arrays={"images": images, "label": d["label"], "split": d["split"], "source": d["source"]})
        for stride in [2, 1]:
            name = f"{'rgb' if color else 'gray'}100_stem{stride}"
            rows = []
            for seed in seeds:
                cfg = {**winner_config("resnet", seed), "seed": seed, "in_ch": 3 if color else 1, "stem_stride": stride}
                r = train_once("diag_inputs", "resnet", f"{name}_{seed}", cfg, data)
                rows.append({"val": r["val"]["accuracy"], "test": r["test"]["accuracy"], "seconds": sum(h["seconds"] for h in r["history"])})
                print(name, seed, rows[-1])
            out[name] = rows
    base = [json.loads(record_path("rafdb", "resnet", s).read_text()) for s in seeds]
    out["gray48_stem1"] = [{"val": r["val"]["accuracy"], "test": r["test"]["accuracy"], "seconds": sum(h["seconds"] for h in r["history"])} for r in base]
    return out


def stratified(data, fraction, seed):
    rng = np.random.default_rng(seed)
    idx = data.idx["train"].cpu().numpy()
    key = (data.source[data.idx["train"]] * 10 + data.label[data.idx["train"]]).cpu().numpy()
    keep = []
    for k in np.unique(key):
        members = idx[key == k]
        keep.append(rng.choice(members, max(1, round(fraction * len(members))), replace=False))
    return torch.tensor(np.sort(np.concatenate(keep)), device="cuda")


def curves(data, seeds):
    n_all = len(data.idx["train"])
    out = {"n": [], "val_error": [], "test_error": [], "fraction": [], "seed": []}
    for f in FRACTIONS + [1.0]:
        for seed in seeds:
            if f == 1.0:
                r = json.loads(record_path("final", "resnet", seed).read_text())
                n = n_all
            else:
                train = stratified(data, f, seed)
                n = len(train)
                cfg = step_matched({**winner_config("resnet", seed), "seed": seed}, n, n_all)
                r = train_once("diag_curves", "resnet", f"{f}_{seed}", cfg, subset(data, train))
            for k, v in [("n", n), ("val_error", 1 - r["val"]["accuracy"]), ("test_error", 1 - r["test"]["accuracy"]), ("fraction", f), ("seed", seed)]:
                out[k].append(v)
            print(f, seed, n, round(1 - r["val"]["accuracy"], 4))
    out["fit"] = fit_power_law(np.array(out["n"]), np.array(out["val_error"]))
    return out


def fit_power_law(n, err, boot=1000):
    """err = a n^-b + c by least squares; bootstrap over the runs for an interval on c."""
    from scipy.optimize import curve_fit

    f = lambda n, a, b, c: a * (n / 1e4) ** (-b) + c

    def one(n, err):
        p, _ = curve_fit(f, n, err, p0=[0.05, 0.5, 0.1], bounds=([0, 0, 0], [10, 3, 1]), maxfev=20000)
        return p

    a, b, c = one(n, err)
    rng = np.random.default_rng(0)
    cs = []
    for _ in range(boot):
        i = rng.integers(0, len(n), len(n))
        if len(np.unique(n[i])) < 3:
            continue
        try:
            cs.append(one(n[i], err[i])[2])
        except RuntimeError:
            pass
    return {"a": a, "b": b, "c": c, "c_ci": [float(np.percentile(cs, 2.5)), float(np.percentile(cs, 97.5))],
            "predicted_error_at_2x": float(f(2 * n.max(), a, b, c))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["sanity", "clean", "sources", "curves", "inputs"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    if a.step == "inputs":
        merge("inputs", inputs(a.seeds))
        raise SystemExit
    data = GPUData()
    if a.step == "sanity":
        add_to_plan([{"stage": "diag_sanity", "model": m, "iteration": "256", "epochs": 300} for m in WINNERS])
        merge("sanity", sanity(data))
    elif a.step == "clean":
        merge("clean", clean(data))
    elif a.step == "sources":
        merge("sources", sources(data, a.seeds))
    else:
        merge("curves", curves(data, a.seeds))
