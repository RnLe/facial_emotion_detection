"""The published references on RAF-DB and on our data (part two, phase 1).

reproduce: FMAE and POSTER++ with their released RAF-DB weights on the official RAF-DB
    test set: native 100 px color (their own setting), the same faces in grayscale, and
    the same faces at 48 px grayscale (our format). The class order of their outputs is
    checked on RAF-DB training images first.
applied: every reference on our validation and test sets, as released (CLIP zero-shot
    from text prompts).
probe: CLIP's frozen image features with a logistic-regression head, regularisation
    chosen on validation.
finetune: a few epochs on our training set (fixed length, last weights). The fine-tuned
    model's logits for every image of both datasets are kept for distillation later.

Records go to runs/references/<model>/<step>.json, the summary to results/references.json.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from fer.data import load_rafdb
from fer.finetune import evaluate, finetune, predict
from fer.metrics import scores
from fer.reference import RAF_ORDER, load
from fer.runs import RUNS
from fer.train import GPUData, save

OUT = RUNS / "references"
RAFDB = Path("data/processed/rafdb.npz")
CONFIGS = {
    # FMAE's own recipe uses layer decay 0.65 and drop path 0.1; a lower peak learning rate
    # since it starts from the RAF-DB fine-tuned weights, not the pretrained backbone.
    "fmae": dict(epochs=4, batch_size=32, lr=5e-5, layer_decay=0.65, depth=24, weight_decay=0.05, warmup_epochs=0.5),
    # POSTER++'s own optimiser: Adam at 3.5e-5 without weight decay (SAM left out: twice the cost).
    "poster": dict(epochs=10, batch_size=64, lr=3.5e-5, weight_decay=0.0, warmup_epochs=1),
    # CLIP: the usual fine-tuning setting for ViT-B/16, head from the linear probe (LP-FT).
    "clip": dict(epochs=5, batch_size=64, lr=3e-5, layer_decay=0.65, depth=12, weight_decay=0.05, warmup_epochs=0.5),
}
COMMON = dict(label_smoothing=0.1, class_weights="sqrt", augment=1.0, seed=0)


def record(model, step, value):
    save(value, OUT / model / f"{step}.json")
    path = Path("results/references.json")
    res = json.loads(path.read_text()) if path.exists() else {}
    res.setdefault(model, {})[step] = {k: v for k, v in value.items() if k != "history"}
    path.write_text(json.dumps(res, indent=1))


def on_gpu(images):
    t = torch.tensor(images, device="cuda")
    return t.permute(0, 3, 1, 2).contiguous() if t.ndim == 4 else t[:, None]


def rafdb_variants():
    """The official RAF-DB test set three ways, plus 1,000 training faces for the order check."""
    color = load_rafdb(color=True)
    gray = load_rafdb()
    d = np.load(RAFDB)
    test, train = d["split"] == 2, np.where(d["split"] == 0)[0][:1000]
    return {
        "check": (on_gpu(color["images"][train]), d["label"][train]),
        "native": on_gpu(color["images"][test]),
        "gray": on_gpu(gray["images"][test]),
        "gray48": on_gpu(d["images"][test]),
        "label": torch.tensor(d["label"][test], device="cuda"),
        "keep": torch.tensor(d["in_ours"][test], device="cuda"),
    }


def throughput(model, spec, n=256):
    x = torch.zeros(n, 1, 48, 48, dtype=torch.uint8, device="cuda")
    predict(model, spec, x)
    torch.cuda.synchronize()
    t = time.perf_counter()
    predict(model, spec, x)
    torch.cuda.synchronize()
    return n / (time.perf_counter() - t)


def reproduce(name, raf):
    model, spec = load(name)
    model = model.cuda()
    x, y = raf["check"]
    prob = predict(model, spec, x)
    as_is = prob[:, RAF_ORDER]  # the raw outputs, read as if already in CLASSES order
    check = {"raf_order": float((prob.argmax(1).cpu().numpy() == y).mean()),
             "classes_order": float((as_is.argmax(1).cpu().numpy() == y).mean())}
    out = {"order_check_train_1000": check}
    for variant in ["native", "gray", "gray48"]:
        prob = predict(model, spec, raf[variant])
        out[variant] = {"test": scores(prob, raf["label"]), "test_dedup": scores(prob[raf["keep"]], raf["label"][raf["keep"]])}
    out["params"] = sum(p.numel() for p in model.parameters())
    out["images_per_second"] = throughput(model, spec)
    record(name, "reproduce", out)
    print(name, check, {v: round(out[v]["test"]["accuracy"], 4) for v in ["native", "gray", "gray48"]})


def applied(name, data):
    model, spec = load(name)
    model = model.cuda()
    out = {}
    for split in ["val", "test"]:
        out.update(evaluate(model, spec, data, split)[0])
    record(name, "applied", out)
    print(name, {k: round(v["accuracy"], 4) for k, v in out.items()})


@torch.no_grad()
def clip_features(model, spec, images, batch_size=512):
    from fer.finetune import prepare

    model.eval()
    out = []
    for k in range(0, len(images), batch_size):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model.features(prepare(images[k:k + batch_size].float().div(255), spec)).float())
    return torch.cat(out)


def probe(data):
    from sklearn.linear_model import LogisticRegression

    model, spec = load("clip")
    model = model.cuda()
    feats = {s: clip_features(model, spec, data.images[data.idx[s]]).cpu().numpy() for s in ["train", "val", "test"]}
    labels = {s: data.label[data.idx[s]].cpu().numpy() for s in ["train", "val", "test"]}
    counts = np.bincount(labels["train"], minlength=7)
    w = (counts.sum() / counts) ** 0.5
    weight = dict(enumerate(w / (w * counts).sum() * counts.sum()))
    best = None
    for c in [0.1, 0.3, 1, 3, 10, 30]:
        clf = LogisticRegression(C=c, class_weight=weight, max_iter=3000).fit(feats["train"], labels["train"])
        acc = (clf.predict(feats["val"]) == labels["val"]).mean()
        print(f"C {c}: val {acc:.4f}")
        if best is None or acc > best[0]:
            best = (acc, c, clf)
    _, c, clf = best
    torch.save({"weight": torch.tensor(clf.coef_, dtype=torch.float32), "bias": torch.tensor(clf.intercept_, dtype=torch.float32)},
               OUT / "clip" / "probe_head.pt")
    out = {"C": c}
    for s in ["val", "test"]:
        prob = torch.tensor(clf.predict_proba(feats[s]))
        true = torch.tensor(labels[s])
        out[s] = scores(prob, true)
        src = data.source[data.idx[s]].cpu()
        for name, k in [("fer", 0), ("raf", 1)]:
            out[f"{s}_{name}"] = scores(prob[src == k], true[src == k])
    record("clip", "probe", out)
    print("clip probe", {k: round(v["accuracy"], 4) for k, v in out.items() if isinstance(v, dict)})


def tune(name, data, raf):
    cfg = {**COMMON, **CONFIGS[name]}
    model, spec = load(name)
    if name == "clip":
        head = torch.load(OUT / "clip" / "probe_head.pt")
        with torch.no_grad():
            model.head.weight.copy_(head["weight"])
            model.head.bias.copy_(head["bias"])
        model.visual.set_grad_checkpointing(True)
    elif name == "fmae":
        model.set_grad_checkpointing(True)
    model = model.cuda()
    state = {"config": cfg, "status": "running", "started": time.time()}
    rec = OUT / name / "finetune.json"
    history = finetune(model, spec, data, cfg, on_epoch=lambda h: save({**state, "history": h}, rec))
    out = {"config": cfg, "history": history}
    for split in ["val", "test"]:
        out.update(evaluate(model, spec, data, split)[0])
    prob = predict(model, spec, raf["gray48"])
    out["rafdb_gray48"] = {"test": scores(prob, raf["label"]), "test_dedup": scores(prob[raf["keep"]], raf["label"][raf["keep"]])}
    # logits for distillation: every image of our dataset and of RAF-DB at 48 px
    logits = {}
    for key, images in [("dataset", data.images), ("rafdb", GPUData(RAFDB).images)]:
        p = predict(model, spec, images)
        logits[key] = p.clamp_min(1e-8).log().half().cpu()
    torch.save(logits, OUT / name / "finetune_logprob.pt")
    torch.save({k: v.to(torch.bfloat16) for k, v in model.state_dict().items()}, OUT / name / "finetune_bf16.pt")
    record(name, "finetune", {**out, "status": "done", "finished": time.time()})
    print(name, "finetuned", {k: round(v["accuracy"], 4) for k, v in out.items() if isinstance(v, dict) and "accuracy" in v})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["reproduce", "applied", "probe", "finetune"])
    ap.add_argument("--models", nargs="+", default=["fmae", "poster", "clip"])
    a = ap.parse_args()
    for m in a.models:
        (OUT / m).mkdir(parents=True, exist_ok=True)
    if a.step == "reproduce":
        raf = rafdb_variants()
        for m in a.models:
            if m != "clip":
                reproduce(m, raf)
    elif a.step == "applied":
        data = GPUData()
        for m in a.models:
            applied(m, data)
    elif a.step == "probe":
        probe(GPUData())
    else:
        data, raf = GPUData(), rafdb_variants()
        for m in a.models:
            tune(m, data, raf)
