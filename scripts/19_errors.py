"""Per-image evidence: shared errors, the annotators' ceiling, label errors (part two,
phase 2, second half).

table: class probabilities of the nine final runs (three family winners, three seeds)
    for every validation and test face, with label, source, and the FER+ votes.
analyze: per-model and ensemble accuracy, agreement between families, the oracle (some
    family right), and on the FER2013 part the annotators' view:
    - accuracy by agreement (5 to 10 of 10 votes for the label);
    - expected agreement with one random annotator, E[p_pred], against the best possible,
      E[max_c p_c] (Ishida et al. 2023 for the binary case), bracketed from the votes;
    - agreement between the majorities of two random halves of the annotators;
    - rank correlation of the model's entropy with the votes' entropy.
cl: out-of-fold probabilities for the training set (5 folds, ResNet-18 winner), then
    confident learning (Northcutt et al. 2021): a face is a candidate label error when its
    probability for another class clears that class's average self-confidence.
retrain: the ResNet-18 winner trained without the candidates, 3 seeds.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from fer.data import CLASSES, FERPLUS_EMOTIONS, FERPLUS_TO_CLASS, load_fer2013
from fer.models import build
from fer.runs import RUNS, is_done, record_path
from fer.train import GPUData, predict, run

WINNERS = {"vgg": "shape_final", "resnet": "final", "densenet": "shape_final"}
MODEL_ARGS = ("dropout", "drop_path", "width", "depth", "stages")
OUT = RUNS / "errors"
TABLE = OUT / "table.npz"


def winner_config(m, seed=0):
    cfg = json.loads(record_path(WINNERS[m], m, seed).read_text())["config"]
    return {k: v for k, v in cfg.items() if k not in ("stage", "iteration")}


def load_winner(m, seed):
    cfg = winner_config(m, seed)
    model = build(m, **{k: cfg[k] for k in MODEL_ARGS if k in cfg}).cuda().to(memory_format=torch.channels_last)
    model.load_state_dict(torch.load(record_path(WINNERS[m], m, seed).with_suffix(".pt"), map_location="cuda"))
    return model


def merge(key, value):
    path = Path("results/errors.json")
    res = json.loads(path.read_text()) if path.exists() else {}
    res[key] = value
    path.write_text(json.dumps(res, indent=1))


def votes_for(dataset):
    """FER+ votes per face: (N, 10) with all ten columns, and (N, 7) in CLASSES order;
    zeros for RAF-DB faces."""
    v = load_fer2013()["votes"]
    cols = [FERPLUS_EMOTIONS.index(next(e for e, c in FERPLUS_TO_CLASS.items() if c == k)) for k in range(len(CLASSES))]
    all10 = np.zeros((len(dataset["label"]), v.shape[1]), int)
    fer = dataset["source"] == 0
    all10[fer] = v[dataset["original_index"][fer]]
    return all10, all10[:, cols]


def table():
    data = GPUData()
    d = np.load("data/processed/dataset.npz")
    idx = torch.cat([data.idx["val"], data.idx["test"]])
    names, probs = [], []
    for m in WINNERS:
        for seed in range(3):
            model = load_winner(m, seed)
            p = []
            for split in ["val", "test"]:
                p.append(predict(model, data, split)[0])
            probs.append(torch.cat(p).cpu().numpy().astype(np.float32))
            names.append(f"{m}_{seed}")
    i = idx.cpu().numpy()
    all10, v7 = votes_for(d)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(TABLE, index=i, split=d["split"][i], label=d["label"][i], source=d["source"][i],
                        agreement=d["agreement"][i], votes10=all10[i], votes=v7[i], probs=np.stack(probs), names=names)
    print("table:", np.stack(probs).shape)


def acc(prob, label):
    return float((prob.argmax(-1) == label).mean())


def ceiling(votes, pred_probs, rng, draws=2000):
    """The best possible agreement with one random annotator, E[max_c p_c], bracketed from
    the seven-class votes. Upper: the plug-in max of the vote shares (too high with ten
    votes, since the max of noisy shares is biased up). Lower: one annotator against the
    majority of the others, an unbiased estimate of what a panel of n - 1 achieves, which
    the best classifier can only beat. Between them a bootstrap bias correction of the
    plug-in. The model's expected agreement with a random annotator is E[p_pred]."""
    n = votes.sum(1)
    p_hat = votes / n[:, None]
    plug_in = p_hat.max(1)
    boot = np.array([rng.multinomial(k, p, size=draws).max(1).mean() / k for k, p in zip(n, p_hat)])
    loo = []
    for v in votes:
        credit = 0.0
        for c in np.nonzero(v)[0]:
            rest = v.copy()
            rest[c] -= 1
            top = np.flatnonzero(rest == rest.max())
            credit += v[c] * (c in top) / len(top)
        loo.append(credit / v.sum())
    return {"upper_plug_in": float(plug_in.mean()), "bias_corrected": float((2 * plug_in - boot).mean()),
            "lower_panel_of_others": float(np.mean(loo)),
            "model": {k: float(p_hat[np.arange(len(p_hat)), p.argmax(1)].mean()) for k, p in pred_probs.items()}}


def human_baselines(votes, label, rng, splits=200):
    """The majorities of two random halves of the annotators agreeing (ties broken at
    random): how stable a five-person majority label is."""
    halves = []
    for _ in range(splits):
        agree = []
        for v in votes:
            pool = np.repeat(np.arange(len(v)), v)
            rng.shuffle(pool)
            h = len(pool) // 2
            a, b = np.bincount(pool[:h], minlength=len(v)), np.bincount(pool[h:2 * h], minlength=len(v))
            ma, mb = rng.choice(np.flatnonzero(a == a.max())), rng.choice(np.flatnonzero(b == b.max()))
            agree.append(ma == mb)
        halves.append(np.mean(agree))
    return {"half_vs_half": float(np.mean(halves))}


def entropy(p):
    p = np.clip(p, 1e-12, 1)
    return -(p * np.log(p)).sum(-1)


def analyze():
    t = np.load(TABLE)
    probs, names, label, split, source = t["probs"], list(t["names"]), t["label"], t["split"], t["source"]
    fam = {m: [names.index(f"{m}_{s}") for s in range(3)] for m in WINNERS}
    rng = np.random.default_rng(0)
    out = {}
    for sname, sk in [("val", 1), ("test", 2)]:
        s = split == sk
        P, y = probs[:, s], label[s]
        single = {n: acc(P[i], y) for i, n in enumerate(names)}
        ens = {
            "seeds_" + m: acc(P[fam[m]].mean(0), y) for m in WINNERS
        }
        ens["families_one_seed"] = float(np.mean([acc(P[[fam[m][k] for m in WINNERS]].mean(0), y) for k in range(3)]))
        ens["all_nine"] = acc(P.mean(0), y)
        right = np.stack([P[fam[m][0]].argmax(-1) == y for m in WINNERS])
        pred = {m: P[fam[m][0]].argmax(-1) for m in WINNERS}
        pairs = {f"{a}_{b}": float((pred[a] == pred[b]).mean()) for a, b in [("vgg", "resnet"), ("vgg", "densenet"), ("resnet", "densenet")]}
        row = {"single": single, "single_mean_by_family": {m: float(np.mean([single[names[i]] for i in fam[m]])) for m in WINNERS},
               "ensembles": ens, "oracle_any_family": float(right.any(0).mean()), "all_families_right": float(right.all(0).mean()),
               "agreement": pairs}
        for src_name, k in [("fer", 0), ("raf", 1)]:
            m = source[s] == k
            row[f"ensemble_all_nine_{src_name}"] = acc(P.mean(0)[m], y[m])
            row[f"oracle_{src_name}"] = float(right[:, m].any(0).mean())
        # the annotators' view, FER2013 part
        f = source[s] == 0
        votes, agreement = t["votes"][s][f], t["agreement"][s][f]
        Pf, yf = P[:, f], y[f]
        by_agreement = {}
        for a in range(5, 11):
            m = agreement == a
            if m.any():
                by_agreement[a] = {"n": int(m.sum()), "resnet": float(np.mean([acc(Pf[i][m], yf[m]) for i in fam["resnet"]])),
                                   "ensemble": acc(Pf.mean(0)[m], yf[m])}
        row["fer_by_agreement"] = by_agreement
        row["fer_ceiling"] = ceiling(votes, {"resnet_0": Pf[fam["resnet"][0]], "ensemble_all_nine": Pf.mean(0)}, rng)
        row["fer_humans"] = human_baselines(votes, yf, rng)
        row["fer_entropy_spearman"] = float(spearmanr(entropy(Pf.mean(0)), entropy(votes / votes.sum(1, keepdims=True)))[0])
        out[sname] = row
        print(sname, json.dumps({k: row[k] for k in ["single_mean_by_family", "ensembles", "oracle_any_family", "fer_ceiling", "fer_humans"]}, indent=None)[:900])
    return out


FOLDS = 5


def cl():
    """Out-of-fold probabilities, then confident learning on the training set."""
    import copy

    data = GPUData()
    train = data.idx["train"]
    g = torch.Generator(device="cuda").manual_seed(0)
    perm = train[torch.randperm(len(train), device="cuda", generator=g)]
    folds = perm.chunk(FOLDS)
    path = OUT / "oof.pt"
    oof = torch.load(path) if path.exists() else {}
    for k in range(FOLDS):
        if k in oof:
            continue
        d = copy.copy(data)
        d.idx = {**data.idx, "train": torch.cat([folds[j] for j in range(FOLDS) if j != k])}
        cfg = {**winner_config("resnet", 0), "stage": "errors_cl", "iteration": k}
        _, model = run(cfg, data=d, record=record_path("errors_cl", "resnet", k))
        d.idx["held_out"] = folds[k]
        prob, _ = predict(model, d, "held_out")
        oof[k] = (folds[k].cpu(), prob.cpu())
        OUT.mkdir(parents=True, exist_ok=True)
        torch.save(oof, path)
    idx = torch.cat([oof[k][0] for k in range(FOLDS)])
    prob = torch.cat([oof[k][1] for k in range(FOLDS)]).numpy()
    label = data.label[idx.cuda()].cpu().numpy()
    source = data.source[idx.cuda()].cpu().numpy()
    thresholds = np.array([prob[label == c, c].mean() for c in range(7)])
    above = prob >= thresholds
    masked = np.where(above, prob, -1)
    guess = masked.argmax(1)
    confident = above.any(1)
    flagged = confident & (guess != label)
    order = np.argsort(prob[np.arange(len(label)), label])  # least self-confident first
    flagged_idx = idx.numpy()[order][flagged[order]]
    np.save(OUT / "flagged.npy", flagged_idx)
    joint = np.zeros((7, 7), int)
    for y, j in zip(label[confident], guess[confident]):
        joint[y, j] += 1
    out = {"oof_accuracy": float((prob.argmax(1) == label).mean()), "thresholds": thresholds.tolist(),
           "flagged": int(flagged.sum()), "flagged_share": float(flagged.mean()),
           "flagged_share_fer": float(flagged[source == 0].mean()), "flagged_share_raf": float(flagged[source == 1].mean()),
           "confident_joint": joint.tolist(),
           "flagged_by_label": {CLASSES[c]: int((flagged & (label == c)).sum()) for c in range(7)}}
    print(json.dumps(out, indent=None)[:600])
    return out


def retrain(seeds):
    import copy

    data = GPUData()
    flagged = torch.tensor(np.load(OUT / "flagged.npy"), device="cuda")
    keep = data.idx["train"][~torch.isin(data.idx["train"], flagged)]
    d = copy.copy(data)
    d.idx = {**data.idx, "train": keep}
    rows = []
    for seed in seeds:
        rec = record_path("errors_clean", "resnet", seed)
        if not is_done("errors_clean", "resnet", seed):
            run({**winner_config("resnet", seed), "seed": seed, "stage": "errors_clean", "iteration": seed}, data=d, record=rec, test=True)
        r = json.loads(rec.read_text())
        base = json.loads(record_path("final", "resnet", seed).read_text())
        rows.append({"val": r["val"]["accuracy"], "val_macro_f1": r["val"]["macro_f1"],
                     "base_val": base["val"]["accuracy"], "base_val_macro_f1": base["val"]["macro_f1"],
                     "test": r["test"]["accuracy"], "base_test": base["test"]["accuracy"]})
        print(seed, rows[-1])
    return {"removed": int(len(data.idx["train"]) - len(keep)), "runs": rows}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["table", "analyze", "cl", "retrain"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    if a.step == "table":
        table()
    elif a.step == "analyze":
        merge("analysis", analyze())
    elif a.step == "cl":
        merge("confident_learning", cl())
    else:
        merge("retrain_without_flagged", retrain(a.seeds))
