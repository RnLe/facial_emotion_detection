"""Inside the trained winners (part two, phase 2): what each stage holds, how similar the
families' representations are, and which face regions each class depends on.

probes: features after every stage (average-pooled to 2x2, which keeps the coarse layout
    of the aligned faces), a logistic regression on 8,000 training faces, scored on
    validation: for the expression, and for the source (FER2013 or RAF-DB). Alain and
    Bengio 2016.
cka: linear CKA (Kornblith et al. 2019) between the penultimate features of the families,
    and between seeds of one family as the reference for "the same".
occlusion: brows, eyes, nose or mouth replaced by the mean training face; recall per
    class on validation, against the intact faces. Regions read off the mean face.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fer.data import CLASSES
from fer.models import build
from fer.runs import record_path
from fer.train import MEAN, STD, GPUData

WINNERS = {"vgg": "shape_final", "resnet": "final", "densenet": "shape_final"}
MODEL_ARGS = ("dropout", "drop_path", "width", "depth", "stages")
# rows, columns on the 48x48 aligned faces (from the mean training face)
REGIONS = {"brows": (8, 13, 6, 42), "eyes": (13, 20, 6, 42), "nose": (20, 31, 17, 31), "mouth": (31, 42, 10, 38)}


def load_winner(m, seed=0):
    cfg = json.loads(record_path(WINNERS[m], m, seed).read_text())["config"]
    model = build(m, **{k: cfg[k] for k in MODEL_ARGS if k in cfg}).cuda().to(memory_format=torch.channels_last)
    model.load_state_dict(torch.load(record_path(WINNERS[m], m, seed).with_suffix(".pt"), map_location="cuda"))
    return model.eval()


def stage_modules(m, model):
    if m == "resnet":
        return [("stem", model.stem)] + [(f"stage{k + 1}", s) for k, s in enumerate(model.stages)]
    if m == "vgg":
        pools = [mod for mod in model.features if isinstance(mod, nn.MaxPool2d)]
        return [(f"block{k + 1}", p) for k, p in enumerate(pools)]
    blocks = [mod for mod in model.features if type(mod).__name__ == "DenseBlock"]
    return [("stem", model.features[0])] + [(f"block{k + 1}", b) for k, b in enumerate(blocks)] + [("final", model.features[-1])]


@torch.no_grad()
def features(model, mods, data, idx, pool=2, batch_size=512):
    store = {}
    hooks = [mod.register_forward_hook(lambda _, __, out, name=name: store.__setitem__(name, out)) for name, mod in mods]
    out = {name: [] for name, _ in mods}
    out["penultimate"] = []
    pen = model.head[-1] if hasattr(model, "head") else None
    hp = pen.register_forward_hook(lambda mod, inp, _: store.__setitem__("penultimate", inp[0]))
    for k in range(0, len(idx), batch_size):
        x, _ = data.batch(idx[k:k + batch_size])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(x)
        for name, _ in mods:
            out[name].append(F.adaptive_avg_pool2d(store[name].float(), pool).flatten(1).cpu())
        out["penultimate"].append(store["penultimate"].float().cpu())
    for h in hooks + [hp]:
        h.remove()
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def probe_scores(train_x, train_y, val_x, val_y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    s = StandardScaler().fit(train_x)
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(s.transform(train_x), train_y)
    return float((clf.predict(s.transform(val_x)) == val_y).mean())


def probes(data):
    g = torch.Generator(device="cuda").manual_seed(0)
    train = data.idx["train"][torch.randperm(len(data.idx["train"]), device="cuda", generator=g)[:8000]]
    val = data.idx["val"]
    ys = {k: (data.label[i].cpu().numpy(), data.source[i].cpu().numpy()) for k, i in [("train", train), ("val", val)]}
    out = {}
    for m in WINNERS:
        model = load_winner(m)
        mods = stage_modules(m, model)
        ft, fv = features(model, mods, data, train), features(model, mods, data, val)
        out[m] = {}
        for name in [n for n, _ in mods] + ["penultimate"]:
            out[m][name] = {"dims": int(ft[name].shape[1]),
                            "expression": probe_scores(ft[name], ys["train"][0], fv[name], ys["val"][0]),
                            "source": probe_scores(ft[name], ys["train"][1], fv[name], ys["val"][1])}
            print(m, name, out[m][name])
    return out


def linear_cka(a, b):
    a = a - a.mean(0)
    b = b - b.mean(0)
    hsic = np.linalg.norm(a.T @ b) ** 2
    return float(hsic / (np.linalg.norm(a.T @ a) * np.linalg.norm(b.T @ b)))


def cka(data):
    val = data.idx["val"]
    feats = {}
    for m in WINNERS:
        for seed in range(2):
            model = load_winner(m, seed)
            feats[f"{m}_{seed}"] = features(model, [], data, val)["penultimate"]
    names = list(feats)
    out = {f"{a}|{b}": linear_cka(feats[a], feats[b]) for i, a in enumerate(names) for b in names[i + 1:]}
    print(json.dumps(out, indent=1))
    return out


@torch.no_grad()
def occlusion(data):
    idx = data.idx["val"]
    y = data.label[idx]
    mean_face = data.images[data.idx["train"]].float().mean(0, keepdim=True).div(255)
    out = {}
    for m in WINNERS:
        model = load_winner(m)
        res = {}
        for region, box in [("none", None)] + list(REGIONS.items()):
            preds = []
            for k in range(0, len(idx), 1024):
                x = data.images[idx[k:k + 1024]].float().div(255)
                if box is not None:
                    r0, r1, c0, c1 = box
                    x[:, :, r0:r1, c0:c1] = mean_face[:, :, r0:r1, c0:c1]
                x = ((x - MEAN) / STD).contiguous(memory_format=torch.channels_last)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    preds.append(model(x).argmax(1))
            p = torch.cat(preds)
            res[region] = {"accuracy": float((p == y).float().mean()),
                           "recall": {CLASSES[c]: float((p[y == c] == c).float().mean()) for c in range(7)}}
        out[m] = res
        print(m, {r: round(v["accuracy"], 4) for r, v in res.items()})
    return out


def merge(key, value):
    path = Path("results/probes.json")
    res = json.loads(path.read_text()) if path.exists() else {}
    res[key] = value
    path.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["probes", "cka", "occlusion"])
    a = ap.parse_args()
    data = GPUData()
    merge(a.step, {"probes": probes, "cka": cka, "occlusion": occlusion}[a.step](data))
