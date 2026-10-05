"""The mirror-invariant ResNet-18 against the ResNet-18 winner (part two, phase 5, H42).

train: the winner's settings with the mirror network, at ResNet-18's parameter count
    (match=params) and at its compute (match=compute), 3 seeds each.
quarter: both networks on a quarter of the training set (the same draw as the learning
    curves in scripts/18_diagnose.py, as many optimiser steps as the full runs), 3 seeds;
    the ResNet-18 runs are the learning-curve runs.
summary: validation and test against the winner with and without flip test-time
    augmentation, recall per class.
"""
import argparse
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np

from fer.data import CLASSES
from fer.runs import add_to_plan, is_done, record_path
from fer.train import GPUData, run

spec = importlib.util.spec_from_file_location("diagnose", Path(__file__).with_name("18_diagnose.py"))
diagnose = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnose)


def base_config(seed):
    cfg = json.loads(record_path("final", "resnet", seed).read_text())["config"]
    return {k: v for k, v in cfg.items() if k not in ("stage", "iteration", "model")}


def train_one(name, cfg, data):
    rec = record_path("mirror", "mirror", name)
    if not is_done("mirror", "mirror", name):
        add_to_plan([{"stage": "mirror", "model": "mirror", "iteration": name, "epochs": cfg["epochs"]}])
        run({**cfg, "model": "mirror", "stage": "mirror", "iteration": name}, data=data, test=True, record=rec)
    return json.loads(rec.read_text())


def train(seeds):
    data = GPUData()
    for match in ["params", "compute"]:
        for seed in seeds:
            r = train_one(f"{match}_{seed}", {**base_config(seed), "seed": seed, "match": match}, data)
            print(match, seed, f"val {r['val']['accuracy']:.4f}  test {r['test']['accuracy']:.4f}")


def quarter(seeds):
    data = GPUData()
    n_all = len(data.idx["train"])
    for seed in seeds:
        sub = diagnose.stratified(data, 0.25, seed)
        d = copy.copy(data)
        d.idx = {**data.idx, "train": sub}
        cfg = diagnose.step_matched({**base_config(seed), "seed": seed, "match": "params"}, len(sub), n_all)
        r = train_one(f"quarter_{seed}", cfg, d)
        print("quarter", seed, f"val {r['val']['accuracy']:.4f}  test {r['test']['accuracy']:.4f}")


def row(r):
    return {"val": r["val"]["accuracy"], "val_macro_f1": r["val"]["macro_f1"], "test": r["test"]["accuracy"],
            "test_macro_f1": r["test"]["macro_f1"], "test_fer": r["test_fer"]["accuracy"], "test_raf": r["test_raf"]["accuracy"],
            **{f"val_recall_{c}": r["val"]["recall"][k] for k, c in enumerate(CLASSES)}}


def mean(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def summary(seeds):
    load = lambda stage, model, name: json.loads(record_path(stage, model, name).read_text())
    out = {
        "resnet": mean([row(load("final", "resnet", s)) for s in seeds]),
        "mirror_params": mean([row(load("mirror", "mirror", f"params_{s}")) for s in seeds]),
        "mirror_compute": mean([row(load("mirror", "mirror", f"compute_{s}")) for s in seeds]),
    }
    tta = json.loads(Path("results/optimize.json").read_text())["tta"]["tta"]
    out["resnet_tta"] = {k: v["mean"] for k, v in tta.items()}
    q = [f"quarter_{s}" for s in seeds if record_path("mirror", "mirror", f"quarter_{s}").exists()]
    if q:
        out["quarter_mirror"] = mean([row(load("mirror", "mirror", n)) for n in q])
        out["quarter_resnet"] = mean([row(load("diag_curves", "resnet", f"0.25_{s}")) for s in seeds])
    for k, v in out.items():
        print(k, {m: round(x, 4) for m, x in v.items() if not m.startswith("val_recall")})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["train", "quarter", "summary"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    if a.step == "train":
        train(a.seeds)
    elif a.step == "quarter":
        quarter(a.seeds)
    else:
        path = Path("results/mirror.json")
        path.write_text(json.dumps(summary(a.seeds), indent=1))
