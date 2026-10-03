"""Compact data for the case study on the portfolio site: one JSON file per part of the
study, plus a sprite of sample faces (FER2013 only: RAF-DB images may not be
redistributed). Floats are rounded and long curves thinned, so the whole export stays
small. Written to --out (default results/site); the site copies these files.
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import optuna
import torch
from PIL import Image

from fer.compress import output_stats, standardised_pca
from fer.data import CLASSES
from fer.models import build
from fer.runs import RUNS
from fer.train import GPUData

p = argparse.ArgumentParser()
p.add_argument("--out", default="results/site")
args = p.parse_args()
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
MODELS = ["cnn", "vgg", "resnet", "densenet", "vit", "convnext", "cct"]
TRIO = ["vgg", "resnet", "densenet"]


def r(x, d=4):
    if isinstance(x, float):
        return round(x, d)
    if isinstance(x, dict):
        return {k: r(v, d) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [r(v, d) for v in x]
    return x


def write(name, obj):
    (OUT / f"{name}.json").write_text(json.dumps(r(obj), separators=(",", ":")))
    print(name, (OUT / f"{name}.json").stat().st_size // 1024, "KB")


def records(stage, model):
    out = []
    for f in sorted((RUNS / stage / model).glob("*.json")):
        rec = json.loads(f.read_text())
        if rec.get("status") == "done":
            out.append(rec)
    return out


def mean_curve(runs, key):
    n = min(len(x["history"]) for x in runs)
    return [float(np.mean([x["history"][i][key] for x in runs])) for i in range(n)]


def study(name):
    storage = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(str(RUNS / "optuna" / f"{name}.log")))
    return optuna.load_study(study_name=name, storage=storage)


# ------------------------------------------------------------------ data
summary = json.loads(Path("results/data/summary.json").read_text())
write("data", {"classes": CLASSES, **{k: summary[k] for k in ("fer_total", "kept", "dropped_by_reason", "counts",
                                                               "fer_label_changed_by_ferplus", "duplicate_groups")}})

d = np.load("data/processed/dataset.npz")
rng = np.random.default_rng(1)
train = d["split"] == 0
rows = []
for k in range(7):
    pick = rng.choice(np.where(train & (d["source"] == 0) & (d["label"] == k))[0], 8, replace=False)
    rows.append(np.concatenate([d["images"][i] for i in pick], axis=1))
Image.fromarray(np.concatenate(rows, axis=0)).save(OUT / "samples.png")
# thumbnail: one column per emotion, four faces each, 16:9-ish, upscaled twice
cols = [np.concatenate([d["images"][i] for i in rng.choice(np.where(train & (d["source"] == 0) & (d["label"] == k))[0], 4, replace=False)], axis=0)
        for k in range(7)]
Image.fromarray(np.concatenate(cols, axis=1)).resize((672, 384), Image.LANCZOS).save(OUT / "thumb.webp", quality=88)

# --------------------------------------------- baselines and stage 1 finals
base, final = {}, {}
for m in MODELS + ["resnet_pretrained"]:
    runs = records("baseline", m)
    if not runs:
        continue
    base[m] = {"params": runs[0]["params"], "test": float(np.mean([x["test"]["accuracy"] for x in runs])),
               "test_f1": float(np.mean([x["test"]["macro_f1"] for x in runs])),
               "seconds": float(np.median([e["seconds"] for x in runs for e in x["history"]])),
               **{k: mean_curve(runs, k) for k in ("train_loss", "val_loss", "train_accuracy", "val_accuracy", "val_macro_f1")}}
for m in MODELS:
    runs = records("final", m)
    conf = np.mean([np.array(x["test"]["confusion"], dtype=float) / np.array(x["test"]["confusion"]).sum(1, keepdims=True) for x in runs], 0)
    final[m] = {"test": float(np.mean([x["test"]["accuracy"] for x in runs])), "test_f1": float(np.mean([x["test"]["macro_f1"] for x in runs])),
                "test_std": float(np.std([x["test"]["accuracy"] for x in runs], ddof=1)), "recall": np.mean([x["test"]["recall"] for x in runs], 0).tolist(),
                "confusion": conf.tolist(), "params": runs[0]["params"]}
write("models", {"baseline": base, "final": final})

# ------------------------------------------------------- tuning, stage 1 and 2
tuning = {}
for m in MODELS:
    s = study(m)
    trials = [t for t in s.trials if t.state.name in ("COMPLETE", "PRUNED")]
    done = [t for t in trials if t.state.name == "COMPLETE"]
    imp = optuna.importance.get_param_importances(s, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=0))
    last = lambda t: t.value if t.value is not None else (t.intermediate_values[max(t.intermediate_values)] if t.intermediate_values else None)
    tuning[m] = {"trials": [[t.number, last(t), t.state.name == "PRUNED"] for t in trials],
                 "default": done[0].value, "best": s.best_value, "importance": imp}
shape = {}
for m in TRIO:
    s = study(f"{m}_shape")
    done = [t for t in s.trials if t.state.name == "COMPLETE"]
    shape[m] = {"trials": [[t.number, t.user_attrs.get("params"), t.value] for t in done], "start": done[0].value, "best": s.best_value,
                "importance": optuna.importance.get_param_importances(s, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=0))}
shape_final = json.loads(Path("results/shape_final.json").read_text())
write("tuning", {"stage1": tuning, "stage2": shape, "stage2_test": shape_final})

# ----------------------------------------------------------------- long runs
long = {}
for m in TRIO:
    for v in ("tuned", "larger", "memorize"):
        rec = json.loads((RUNS / "long" / m / f"{v}.json").read_text())
        cool = json.loads((RUNS / "cooldown" / m / f"{v}_e100.json").read_text())
        h = rec["history"]
        long[f"{m}/{v}"] = {"params": rec["params"], "test": cool["test"]["accuracy"], "val_after": cool["val"]["accuracy"],
                            **{k: [e[k] for e in h] for k in ("val_accuracy", "train_clean_accuracy", "val_loss", "train_clean_loss", "weight_rank", "nc1")}}
ref = {m: {"tuned": float(np.mean([x["test"]["accuracy"] for x in records("final", m)])),
           "larger": float(np.mean([x["test"]["accuracy"] for x in records("shape_final", m)]))} for m in TRIO}
vgg_bn = json.loads((RUNS / "long" / "vgg" / "grok.json").read_text())["history"]
vgg_bn_cool = json.loads((RUNS / "cooldown" / "vgg" / "grok_e1110.json").read_text())
write("long", {"runs": long, "final60": ref,
               "vgg_bn_grok": {"epoch": [e["epoch"] for e in vgg_bn], "val_accuracy": [e["val_accuracy"] for e in vgg_bn],
                               "train_loss": [e["train_loss"] for e in vgg_bn], "train_accuracy": [e["train_accuracy"] for e in vgg_bn],
                               "test_after": vgg_bn_cool["test"]["accuracy"], "start_test": json.loads((RUNS / "cooldown" / "vgg" / "memorize_e100.json").read_text())["test"]["accuracy"]}})

# ------------------------------------------------------------------- grokking
grok = {}
for f in sorted((RUNS / "grok").glob("*/*.json")):
    rec = json.loads(f.read_text())
    h = [e for e in rec["history"] if e["step"] <= 100_000]
    steps = np.array([e["step"] for e in h])
    keep = sorted({int(np.argmin(np.abs(steps - s))) for s in np.unique(np.geomspace(1, steps[-1], 160).round())} | {len(h) - 1})
    hh = [h[i] for i in keep]
    c = rec["config"]
    full = [e for e in rec["history"] if e["step"] <= 100_000]
    every = 1 if c["model"] == "vgg" else 4  # the VGG sawtooth needs every logged step
    grok[f"{c['model']}/{f.stem}"] = {"alpha": c["alpha"], "weight_decay": c["weight_decay"],
                                      **{k: [e[k] for e in hh] for k in ("step", "train_accuracy", "val_accuracy", "nc1", "weight_norm_rel", "repr_entropy")},
                                      "dense_step": [e["step"] for e in full if e["step"] >= 1000][::every],
                                      "dense_weight_norm": [e["weight_norm_rel"] for e in full if e["step"] >= 1000][::every],
                                      "dense_val": [e["val_accuracy"] for e in full if e["step"] >= 1000][::every]}
write("grok", grok)

# ---------------------------------------------------------------- compression
sweep = json.loads(Path("results/compress.json").read_text())
rep, quant, scratch = {}, {}, {}
for m in TRIO:
    for f in sorted((RUNS / "repair" / m).glob("*.json")):
        if f.name.endswith("_sensitivity.json") or "_lr" in f.stem:
            continue
        x = json.loads(f.read_text())
        if x.get("status") == "done":
            rep[f"{m}/{f.stem}"] = {"method": x["config"]["method"], "budget": x["config"]["budget"], "params": x["params"], "macs": x["macs"],
                                   "bytes": x["bytes"], "zero_shot": x["zero_shot"]["accuracy"], "val": x["val"]["accuracy"],
                                   "test": x["test"]["accuracy"], "test_f1": x["test"]["macro_f1"], "speed": x["speed"]}
    for f in sorted((RUNS / "quant" / m).glob("*.json")):
        x = json.loads(f.read_text())
        if x.get("status") == "done" and "_lr" not in f.stem:
            quant[f"{m}/{f.stem}"] = {"bytes": x["bytes"], "val": x["val"]["accuracy"], "test": x["test"]["accuracy"], "speed": x["speed"]}
    for x in records("scratch", m):
        scratch[f"{m}/{x['config']['iteration']}"] = {"params": x["params"], "val": x["val"]["accuracy"], "test": x["test"]["accuracy"]}
for m in TRIO:
    sweep[m]["test"] = json.loads((RUNS / "cooldown" / m / "larger_e100.json").read_text())["test"]["accuracy"]

# spectra: VGG, an early and the last conv layer, weights against outputs
cfg = json.loads((RUNS / "cooldown" / "vgg" / "larger_e100.json").read_text())["config"]
net = build("vgg", **{k: cfg[k] for k in ("dropout", "width", "depth", "stages")})
net.load_state_dict(torch.load(RUNS / "cooldown" / "vgg" / "larger_e100.ckpt", weights_only=False, map_location="cpu")["model"])
net = net.cuda().eval()
data = GPUData()
idx = data.idx["train"][torch.randperm(len(data.idx["train"]), generator=torch.Generator().manual_seed(0))[:2048].to(data.idx["train"].device)]
stats = output_stats(net, data.batch(idx)[0].contiguous())
spectra = {}
for name in ("features.3", "features.36"):
    w = dict(net.named_modules())[name].weight.detach().double()
    sw = torch.linalg.svdvals(w.flatten(1)) ** 2
    ev = standardised_pca(*stats[name])[0]
    spectra[name] = {"channels": w.shape[0], "weights": (sw.cumsum(0) / sw.sum()).tolist(), "outputs": (ev.cumsum(0) / ev.sum()).tolist()}
write("compress", {"sweep": sweep, "repair": rep, "quant": quant, "scratch": scratch, "spectra": spectra})
