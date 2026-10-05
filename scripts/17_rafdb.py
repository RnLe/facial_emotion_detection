"""RAF-DB on its own, the benchmark the reference models report.

build: data/processed/rafdb.npz with every RAF-DB image at 48x48 grayscale (the same
    processing as in our dataset), its label, and the split. The protocol of the RAF-DB
    papers: the official train and test sets, nothing removed. Validation is the same 10%
    of the train set that our dataset uses, so selection never sees the test set.
    `in_ours` marks the images our cleaned dataset kept (the test set minus 13 duplicates).
combined: our three family winners, trained on the combined dataset, scored on the full
    official RAF-DB test set.
train: the three family winners trained on RAF-DB alone in our pipeline, 3 seeds.

Family winners, chosen by validation macro-F1 (the study's rule): VGG from the shape
stage, ResNet-18 from stage 1, DenseNet from the shape stage.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from fer.data import load_rafdb
from fer.metrics import scores
from fer.models import build
from fer.runs import RUNS, add_to_plan, is_done, record_path
from fer.train import GPUData, predict, run, save

PATH = Path("data/processed/rafdb.npz")
N_FER = 35887  # RAF-DB images follow FER2013's in our dataset's original_index
WINNERS = {"vgg": "shape_final", "resnet": "final", "densenet": "shape_final"}
MODEL_ARGS = ("dropout", "drop_path", "width", "depth", "stages")


def build_dataset():
    raf = load_rafdb(size=48)
    ours = np.load("data/processed/dataset.npz")
    is_raf = ours["source"] == 1
    raf_idx = ours["original_index"][is_raf] - N_FER
    val = raf_idx[ours["split"][is_raf] == 1]
    split = np.where(raf["usage"] == "train", 0, 2)
    split[val] = 1
    in_ours = np.zeros(len(split), bool)
    in_ours[raf_idx] = True
    np.savez_compressed(
        PATH, images=raf["images"], label=raf["label"], split=split, source=np.ones(len(split), int),
        in_ours=in_ours, name=raf["name"],
    )
    print({s: int((split == k).sum()) for k, s in enumerate(["train", "val", "test"])},
          "test kept in ours:", int(in_ours[split == 2].sum()))


def winner_config(m, seed):
    cfg = json.loads(record_path(WINNERS[m], m, seed).read_text())["config"]
    return {k: v for k, v in cfg.items() if k not in ("stage", "iteration")}


def test_scores(prob, true, keep):
    return {"test": scores(prob, true), "test_dedup": scores(prob[keep], true[keep])}


def combined():
    data = GPUData(PATH)
    keep = torch.tensor(np.load(PATH)["in_ours"], device="cuda")[data.idx["test"]]
    out = {}
    for m, stage in WINNERS.items():
        out[m] = []
        for seed in range(3):
            cfg = winner_config(m, seed)
            model = build(m, **{k: cfg[k] for k in MODEL_ARGS if k in cfg}).cuda().to(memory_format=torch.channels_last)
            model.load_state_dict(torch.load(record_path(stage, m, seed).with_suffix(".pt"), map_location="cuda"))
            prob, true = predict(model, data, "test")
            out[m].append(test_scores(prob, true, keep))
            print(m, seed, f"{out[m][-1]['test']['accuracy']:.4f}")
    return out


def train():
    data = GPUData(PATH)
    keep = torch.tensor(np.load(PATH)["in_ours"], device="cuda")[data.idx["test"]]
    todo = [(m, s) for m in WINNERS for s in range(3)]
    add_to_plan([{"stage": "rafdb", "model": m, "iteration": s, "epochs": 60} for m, s in todo])
    out = {}
    for m, seed in todo:
        rec = record_path("rafdb", m, seed)
        if not is_done("rafdb", m, seed):
            cfg = {**winner_config(m, seed), "stage": "rafdb", "iteration": seed}
            result, model = run(cfg, data=data, test=True, record=rec)
            prob, true = predict(model, data, "test")
            result.update(test_scores(prob, true, keep))
            save({**json.loads(rec.read_text()), **result}, rec)
        r = json.loads(rec.read_text())
        if "test_dedup" not in r:  # a run that stopped between its two saves
            cfg = r["config"]
            model = build(m, **{k: cfg[k] for k in MODEL_ARGS if k in cfg}).cuda().to(memory_format=torch.channels_last)
            model.load_state_dict(torch.load(rec.with_suffix(".pt"), map_location="cuda"))
            r.update(test_scores(*predict(model, data, "test"), keep))
            save(r, rec)
        out.setdefault(m, []).append({k: r[k] for k in ("val", "test", "test_dedup")})
        print(m, seed, f"val {r['val']['accuracy']:.4f}  test {r['test']['accuracy']:.4f}")
    return out


def merge(key, value):
    path = Path("results/rafdb.json")
    res = json.loads(path.read_text()) if path.exists() else {}
    res[key] = value
    path.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["build", "combined", "train"])
    a = ap.parse_args()
    if a.step == "build":
        build_dataset()
    elif a.step == "combined":
        merge("combined_models", combined())
    else:
        merge("rafdb_only", train())
