"""Figures and tables for the report, from data/processed and runs/. Rerun any time;
parts whose runs are not finished yet are skipped."""
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from fer.data import CLASSES
from fer.models import MODELS, NAMES

matplotlib.use("Agg")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
OUT = Path("report/figures")
OUT.mkdir(parents=True, exist_ok=True)
FER, RAF = "#3a78c9", "#e0a33a"
COLORS = {"cnn": "#8c8c8c", "vgg": "#e0a33a", "resnet": "#3a78c9", "densenet": "#2e9d6a", "vit": "#c9493a",
          "convnext": "#8a5cc9", "cct": "#d1609a", "resnet_pretrained": "#1d3557"}
TABLES = Path("report/tables")
TABLES.mkdir(parents=True, exist_ok=True)


def load(stage):
    """Finished runs of a stage, per model."""
    runs = {}
    for f in sorted(Path("runs", stage).glob("*/*.json")):
        r = json.loads(f.read_text())
        if r.get("status") == "done":
            runs.setdefault(r["config"]["model"], []).append(r)
    return runs


def mean_std(values):
    v = np.array(values) * 100
    return f"{v.mean():.1f} ± {v.std(ddof=1) if len(v) > 1 else 0:.1f}"


def baseline_figures():
    runs = load("baseline")
    if not runs:
        return
    rows = [["Model", "Seeds", "Accuracy", "Macro-F1", "Balanced acc.", "FER+ acc.", "RAF-DB acc.", "ECE", "Params (M)", "s / epoch"]]
    for m in [m for m in MODELS if m in runs]:
        rs = runs[m]
        rows.append([
            NAMES[m], str(len(rs)),
            mean_std([r["test"]["accuracy"] for r in rs]),
            mean_std([r["test"]["macro_f1"] for r in rs]),
            mean_std([r["test"]["balanced_accuracy"] for r in rs]),
            mean_std([r["test_fer"]["accuracy"] for r in rs]),
            mean_std([r["test_raf"]["accuracy"] for r in rs]),
            mean_std([r["test"]["ece"] for r in rs]),
            f"{rs[0]['params'] / 1e6:.2f}",
            f"{np.mean([np.mean([h['seconds'] for h in r['history'][1:]]) for r in rs]):.1f}",
        ])
    keep = ("accuracy", "macro_f1", "balanced_accuracy", "ece", "recall")
    compact = {m: [{"seed": r["config"]["seed"], "best_val_macro_f1": r["best_val_macro_f1"], "params": r["params"],
                    "seconds_per_epoch": float(np.mean([h["seconds"] for h in r["history"][1:]])),
                    **{split: {k: r[split][k] for k in keep} for split in ("val", "test", "test_fer", "test_raf")}}
                   for r in rs] for m, rs in runs.items()}
    Path("results/baselines.json").write_text(json.dumps(compact, indent=1))
    with open(TABLES / "baselines.csv", "w") as f:
        f.writelines(",".join(f'"{c}"' for c in row) + "\n" for row in rows)

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6), sharex=True)
    for m in [m for m in MODELS if m in runs]:
        for ax, key in zip(axes, ["val_macro_f1", "val_accuracy"]):
            curves = np.array([[h[key] for h in r["history"]] for r in runs[m]]) * 100
            ep = np.arange(1, curves.shape[1] + 1)
            ax.plot(ep, curves.mean(0), color=COLORS[m], lw=1.3, label=NAMES[m])
            ax.fill_between(ep, curves.min(0), curves.max(0), color=COLORS[m], alpha=0.15, lw=0)
    axes[0].set_ylabel("validation macro-F1 (%)")
    axes[1].set_ylabel("validation accuracy (%)")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.set_ylim(40, None)
    axes[1].legend(frameon=False, fontsize=6.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "baseline_curves.svg")
    plt.close(fig)

    names = [m for m in MODELS if m in runs]
    recall = np.array([np.mean([r["test"]["recall"] for r in runs[m]], 0) for m in names]) * 100
    fig, ax = plt.subplots(figsize=(6.2, 0.35 * len(names) + 0.8))
    ax.imshow(recall, cmap="Blues", vmin=30, vmax=100, aspect="auto")
    ax.set_xticks(range(7), CLASSES)
    ax.set_yticks(range(len(names)), [NAMES[m] for m in names])
    for i in range(len(names)):
        for j in range(7):
            ax.text(j, i, f"{recall[i, j]:.0f}", ha="center", va="center", fontsize=7, color="white" if recall[i, j] > 75 else "black")
    ax.tick_params(length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "baseline_recall.svg")
    plt.close(fig)


def data_figures():
    d = np.load("data/processed/dataset.npz")
    train = d["split"] == 0
    fig, ax = plt.subplots(figsize=(6.2, 2.4))
    x = np.arange(len(CLASSES))
    fer = [((d["label"] == k) & train & (d["source"] == 0)).sum() for k in range(7)]
    raf = [((d["label"] == k) & train & (d["source"] == 1)).sum() for k in range(7)]
    ax.bar(x, fer, color=FER, label="FER2013 (FER+ labels)")
    ax.bar(x, raf, bottom=fer, color=RAF, label="RAF-DB")
    for k in range(7):
        ax.text(k, fer[k] + raf[k] + 150, f"{fer[k] + raf[k]:,}", ha="center", fontsize=7)
    ax.set_xticks(x, CLASSES)
    ax.set_ylabel("training images")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "class_counts.svg")
    plt.close(fig)

    s = json.loads(Path("results/data/summary.json").read_text())["dropped_by_reason"]
    order = sorted(s, key=s.get)
    fig, ax = plt.subplots(figsize=(6.2, 2.0))
    ax.barh(order, [s[r] for r in order], color="#8c8c8c")
    for k, r in enumerate(order):
        ax.text(s[r] + 40, k, f"{s[r]:,}", va="center", fontsize=7)
    ax.set_xlabel("images dropped")
    fig.tight_layout()
    fig.savefig(OUT / "dropped.svg")
    plt.close(fig)

    # Sample faces: FER2013 only (RAF-DB images may not be redistributed).
    rng = np.random.default_rng(3)
    fig, axes = plt.subplots(7, 8, figsize=(6.2, 5.8))
    for k, c in enumerate(CLASSES):
        pick = rng.choice(np.where(train & (d["source"] == 0) & (d["label"] == k))[0], 8, replace=False)
        for j, i in enumerate(pick):
            axes[k, j].imshow(d["images"][i], cmap="gray", vmin=0, vmax=255)
            axes[k, j].set_xticks([]), axes[k, j].set_yticks([])
            for s_ in axes[k, j].spines.values():
                s_.set_visible(False)
        axes[k, 0].set_ylabel(c, rotation=0, ha="right", va="center", fontsize=8)
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / "samples.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    data_figures()
    baseline_figures()
    print(sorted(p.name for p in OUT.iterdir()))
