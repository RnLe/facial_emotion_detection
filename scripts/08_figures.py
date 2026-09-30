"""Figures and tables for the report, from data/processed and runs/. Rerun any time;
parts whose runs are not finished yet are skipped."""
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from fer.data import CLASSES

matplotlib.use("Agg")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
OUT = Path("report/figures")
OUT.mkdir(parents=True, exist_ok=True)
FER, RAF = "#3a78c9", "#e0a33a"


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
    print(sorted(p.name for p in OUT.iterdir()))
