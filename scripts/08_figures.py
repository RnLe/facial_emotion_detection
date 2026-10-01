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


PARAMS = ["lr", "weight_decay", "batch_size", "label_smoothing", "augment", "class_weights", "warmup_epochs", "dropout", "drop_path"]
PARAM_NAMES = {"lr": "learning rate", "weight_decay": "weight decay", "batch_size": "batch size", "label_smoothing": "label smoothing",
               "augment": "augmentation", "class_weights": "class weights", "warmup_epochs": "warmup", "dropout": "dropout", "drop_path": "drop path"}


def studies(suffix=""):
    """Finished trials of every tuning study, per model (suffix "_shape": stage 2)."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    out = {}
    for m in MODELS:
        path = Path("runs/optuna") / f"{m}{suffix}.log"
        if not path.exists():
            continue
        storage = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(str(path)))
        try:
            study = optuna.load_study(study_name=m + suffix, storage=storage)
        except KeyError:  # an empty journal: the study was never started
            continue
        done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if len(done) >= 3:
            out[m] = (study, done)
    return out


def random_phase(done, startup):
    """Numbers of the trials TPE did not choose. Optuna samples at random until `startup`
    trials have finished (failed ones do not count); trial 0 is the enqueued start."""
    return [t.number for t in sorted(done, key=lambda t: t.number)[1:startup]]


def importance_figure(importance, params, path):
    """fANOVA importances as a grid: models down, settings across, in percent."""
    names = [m for m in MODELS if m in importance]
    params = [p for p in params if any(p in importance[m] for m in names)]
    grid = np.array([[importance[m].get(p, np.nan) * 100 for p in params] for m in names])
    fig, ax = plt.subplots(figsize=(6.4, 0.38 * len(names) + 1.0))
    ax.imshow(np.nan_to_num(grid), cmap="Oranges", vmin=0, vmax=60, aspect="auto")
    for i in range(len(names)):
        for j in range(len(params)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.0f}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(params)), [PARAM_NAMES[p] for p in params], rotation=25, ha="right")
    ax.set_yticks(range(len(names)), [NAMES[m] for m in names])
    ax.tick_params(length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def tuning_figures():
    import optuna

    found = studies()
    if not found:
        return
    base = load("baseline")
    rows = [["Model", "Default", "Best", "Gain", "Default, 60 ep.", "Random median", "Random IQR", "Within 1 pt", "Most important"]]
    importance = {}
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    for k, m in enumerate([m for m in MODELS if m in found]):
        study, done = found[m]
        values = np.array([t.value for t in done]) * 100
        numbers = np.array([t.number for t in done])
        default = values[numbers == 0][0] if (numbers == 0).any() else np.nan
        is_random = np.isin(numbers, random_phase(done, 10))
        random_values = values[is_random]
        best = values.max()
        imp = {}
        if len(done) >= 10:
            imp = optuna.importance.get_param_importances(study, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=0))
            importance[m] = imp
        q1, q3 = (np.percentile(random_values, [25, 75]) if len(random_values) else (np.nan, np.nan))
        long = 100 * np.mean([r["best_val_macro_f1"] for r in base[m]]) if m in base else np.nan
        rows.append([
            NAMES[m], f"{default:.1f}", f"{best:.1f}", f"{best - default:+.1f}", f"{long:.1f}",
            f"{np.median(random_values):.1f}" if len(random_values) else "-", f"{q3 - q1:.1f}" if len(random_values) else "-",
            f"{100 * np.mean(values >= best - 1):.0f}%",
            ", ".join(PARAM_NAMES[p] for p in list(imp)[:2]) if imp else "-",
        ])
        jitter = np.random.default_rng(k).uniform(-0.18, 0.18, len(values))
        is_tpe = ~is_random & (numbers > 0)
        ax.scatter(k + jitter[is_random], values[is_random], s=9, color=COLORS[m], alpha=0.45, lw=0)
        ax.scatter(k + jitter[is_tpe], values[is_tpe], s=9, color=COLORS[m], alpha=1, lw=0)
        ax.scatter([k], [default], marker="_", s=260, color="black", lw=1.5)
    names = [m for m in MODELS if m in found]
    ax.set_xticks(range(len(names)), [NAMES[m] for m in names], rotation=20, ha="right")
    ax.set_ylabel("validation macro-F1 (%)")
    ax.set_title("each dot a trial (pale: random, solid: TPE); bar: default settings", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "tuning_trials.svg")
    plt.close(fig)
    with open(TABLES / "tuning.csv", "w") as f:
        f.writelines(",".join(f'"{c}"' for c in row) + "\n" for row in rows)

    if importance:
        importance_figure(importance, PARAMS, OUT / "tuning_importance.svg")

    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    for m in [m for m in MODELS if m in found]:
        study, done = found[m]
        order = sorted(done, key=lambda t: t.number)
        best = np.maximum.accumulate([t.value * 100 for t in order])
        ax.plot([t.number for t in order], best, color=COLORS[m], lw=1.3, label=NAMES[m])
    ax.axvline(9.5, color="#bbb", lw=0.8, ls="--")
    ax.set_xlabel("trial")
    ax.set_ylabel("best validation macro-F1 so far (%)")
    ax.legend(frameon=False, fontsize=6.5, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "tuning_history.svg")
    plt.close(fig)


def bootstrap(pred_sets, true, n=2000, seed=0):
    """Accuracy and macro-F1 averaged over seeds, with a 95% interval from resampling the
    test images (the same resample for every seed)."""
    rng = np.random.default_rng(seed)
    preds = np.array(pred_sets)

    def score(idx):
        accs, f1s = [], []
        for p in preds:
            t, q = true[idx], p[idx]
            accs.append((q == t).mean())
            f = []
            for k in range(7):
                tp = ((q == k) & (t == k)).sum()
                prec, rec = tp / max(1, (q == k).sum()), tp / max(1, (t == k).sum())
                f.append(0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
            f1s.append(np.mean(f))
        return np.mean(accs), np.mean(f1s)

    full = score(np.arange(len(true)))
    draws = np.array([score(rng.integers(0, len(true), len(true))) for _ in range(n)])
    return full, np.percentile(draws, [2.5, 97.5], axis=0)


def final_figures():
    finals, base = load("final"), load("baseline")
    finals = {m: rs for m, rs in finals.items() if all("test_pred" in r for r in rs)}
    if not finals:
        return
    d = np.load("data/processed/dataset.npz")
    true = d["label"][d["split"] == 2]
    rows = [["Model", "Baseline acc.", "Tuned acc.", "Gain", "Baseline F1", "Tuned F1", "Gain F1", "Tuned acc. 95% CI"]]
    summary = {}
    for m in [m for m in MODELS if m in finals]:
        (acc, f1), ci = bootstrap([r["test_pred"] for r in finals[m]], true)
        b_acc = np.mean([r["test"]["accuracy"] for r in base.get(m, [])]) if m in base else np.nan
        b_f1 = np.mean([r["test"]["macro_f1"] for r in base.get(m, [])]) if m in base else np.nan
        summary[m] = {"accuracy": acc, "macro_f1": f1, "ci": ci.tolist(), "baseline_accuracy": b_acc, "baseline_macro_f1": b_f1}
        rows.append([NAMES[m], f"{100 * b_acc:.1f}", f"{100 * acc:.1f}", f"{100 * (acc - b_acc):+.1f}",
                     f"{100 * b_f1:.1f}", f"{100 * f1:.1f}", f"{100 * (f1 - b_f1):+.1f}", f"{100 * ci[0, 0]:.1f} to {100 * ci[1, 0]:.1f}"])
    with open(TABLES / "final.csv", "w") as f:
        f.writelines(",".join(f'"{c}"' for c in row) + "\n" for row in rows)
    Path("results/final.json").write_text(json.dumps(summary, indent=1))

    # Are neighbours in the ranking really different? Paired bootstrap of the accuracy gap.
    ranked = sorted(summary, key=lambda m: -summary[m]["accuracy"])
    rng = np.random.default_rng(1)
    gaps = []
    for a, b in zip(ranked, ranked[1:]):
        pa, pb = np.array([r["test_pred"] for r in finals[a]]), np.array([r["test_pred"] for r in finals[b]])
        diff = []
        for _ in range(2000):
            idx = rng.integers(0, len(true), len(true))
            diff.append((pa[:, idx] == true[idx]).mean() - (pb[:, idx] == true[idx]).mean())
        lo, hi = np.percentile(diff, [2.5, 97.5])
        gaps.append([NAMES[a], NAMES[b], f"{100 * (summary[a]['accuracy'] - summary[b]['accuracy']):+.1f}", f"{100 * lo:+.1f} to {100 * hi:+.1f}", "yes" if lo > 0 else "no"])
    with open(TABLES / "final_gaps.csv", "w") as f:
        f.write('"Better","Next","Gap","95% interval","Clear?"\n')
        f.writelines(",".join(f'"{c}"' for c in row) + "\n" for row in gaps)

    names = [m for m in MODELS if m in summary]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6), sharey=True)
    for ax, key, label in [(axes[0], "accuracy", "test accuracy (%)"), (axes[1], "macro_f1", "test macro-F1 (%)")]:
        for i, m in enumerate(names):
            b, t = 100 * summary[m][f"baseline_{key}"], 100 * summary[m][key]
            ax.plot([b, t], [i, i], color=COLORS[m], lw=1.5)
            ax.scatter([b], [i], color="white", edgecolor=COLORS[m], s=28, zorder=3)
            ax.scatter([t], [i], color=COLORS[m], s=28, zorder=3)
        ax.set_xlabel(label)
    axes[0].set_yticks(range(len(names)), [NAMES[m] for m in names])
    axes[0].invert_yaxis()
    axes[1].set_title("open: default settings, filled: tuned", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "final_gain.svg")
    plt.close(fig)


SHAPE_PARAMS = ["lr", "weight_decay", "label_smoothing", "augment", "warmup_epochs", "dropout", "width", "depth", "stages"]
PARAM_NAMES.update(width="width", depth="depth", stages="stages")


def shape_figures():
    """Stage 2: what the shape adds over the stage 1 best (trial 0), how much of the spread
    it explains, and accuracy against size."""
    import optuna

    found = studies("_shape")
    if not found:
        return
    rows = [["Model", "Trials", "Start", "Best", "Gain", "Random: IQR", "Within 1 pt of best", "Shape share", "Best shape", "Smallest within 1 pt"]]
    importance, everything = {}, []
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for m in [m for m in MODELS if m in found]:
        study, done = found[m]
        values = np.array([t.value for t in done]) * 100
        numbers = np.array([t.number for t in done])
        size = np.array([t.user_attrs["params"] for t in done]) / 1e6
        start = values[numbers == 0][0] if (numbers == 0).any() else np.nan
        random_values = values[np.isin(numbers, random_phase(done, 12))]
        best = done[int(values.argmax())]
        small = min([t for t in done if 100 * t.value >= values.max() - 1], key=lambda t: (t.user_attrs["params"], -t.value))
        imp = {}
        if len(done) >= 12:
            imp = optuna.importance.get_param_importances(study, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=0))
            importance[m] = imp
        share = sum(imp.get(k, 0) for k in ("width", "depth", "stages"))
        q1, q3 = np.percentile(random_values, [25, 75]) if len(random_values) else (np.nan, np.nan)
        rows.append([
            NAMES[m], str(len(done)), f"{start:.1f}", f"{values.max():.1f}", f"{values.max() - start:+.1f}",
            f"{q3 - q1:.1f}" if len(random_values) else "-", f"{100 * np.mean(values >= values.max() - 1):.0f}%",
            f"{100 * share:.0f}%" if imp else "-",
            f"width {best.params['width']:.2f}, depth {best.params['depth']}, {best.params['stages']} stages ({best.user_attrs['params'] / 1e6:.2f} M)",
            f"{100 * small.value:.1f} with {small.user_attrs['params'] / 1e6:.2f} M",
        ])
        ax.scatter(size, values, s=10, color=COLORS[m], alpha=0.75, lw=0, label=NAMES[m])
        ax.scatter(size[numbers == 0], values[numbers == 0], s=50, facecolor="none", edgecolor=COLORS[m], lw=1.3)
        everything += list(values)
    low = max(min(everything), max(everything) - 12)
    hidden = sum(v < low for v in everything)
    ax.set_ylim(low - 0.5, max(everything) + 0.8)
    ax.set_xscale("log")
    ax.set_xlabel("parameters (millions)")
    ax.set_ylabel("validation macro-F1 (%)")
    ax.set_title("each dot a trial; ring: stage 1 best with the default shape"
                 + (f"; {hidden} trials below {low:.0f}% not shown" if hidden else ""), fontsize=8)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "shape_size.svg")
    plt.close(fig)
    with open(TABLES / "shape.csv", "w") as f:
        f.writelines(",".join(f'"{c}"' for c in row) + "\n" for row in rows)
    if importance:
        importance_figure(importance, SHAPE_PARAMS, OUT / "shape_importance.svg")


def shape_final_figures():
    """Stage 2 final runs against the stage 1 final runs, on the same test images: the
    accuracy gap with a paired bootstrap interval."""
    stage1, picks = load("final"), {"best": load("shape_final"), "smallest": load("shape_compact")}
    models = [m for m in MODELS if m in picks["best"] and m in stage1]
    if not models:
        return
    d = np.load("data/processed/dataset.npz")
    true = d["label"][d["split"] == 2]
    rng = np.random.default_rng(2)
    rows = [["Model", "Network", "Parameters", "Test acc.", "Test F1", "Acc. vs stage 1", "95% interval"]]
    summary = {}
    for m in models:
        p1 = np.array([r["test_pred"] for r in stage1[m]])
        mean = lambda runs, key: np.mean([r["test"][key] for r in runs])
        rows.append([NAMES[m], "stage 1 (default shape)", f"{stage1[m][0]['params'] / 1e6:.2f} M",
                     f"{100 * mean(stage1[m], 'accuracy'):.1f}", f"{100 * mean(stage1[m], 'macro_f1'):.1f}", "", ""])
        summary[m] = {}
        for pick, found in picks.items():
            if m not in found:
                rows.append([NAMES[m], f"stage 2 {pick}", "same as best", "", "", "", ""])
                continue
            runs = found[m]
            p2 = np.array([r["test_pred"] for r in runs])
            gap = []
            for _ in range(2000):
                idx = rng.integers(0, len(true), len(true))
                gap.append((p2[:, idx] == true[idx]).mean() - (p1[:, idx] == true[idx]).mean())
            lo, hi = np.percentile(gap, [2.5, 97.5])
            acc, f1 = mean(runs, "accuracy"), mean(runs, "macro_f1")
            summary[m][pick] = {"params": runs[0]["params"], "accuracy": acc, "macro_f1": f1, "gap": acc - mean(stage1[m], "accuracy"),
                                "gap_ci": [lo, hi], "seeds": len(runs)}
            rows.append([NAMES[m], f"stage 2 {pick}", f"{runs[0]['params'] / 1e6:.2f} M", f"{100 * acc:.1f}", f"{100 * f1:.1f}",
                         f"{100 * summary[m][pick]['gap']:+.1f}", f"{100 * lo:+.1f} to {100 * hi:+.1f}"])
    with open(TABLES / "shape_final.csv", "w") as f:
        f.writelines(",".join(f'"{c}"' for c in row) + "\n" for row in rows)
    Path("results/shape_final.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    data_figures()
    baseline_figures()
    tuning_figures()
    final_figures()
    shape_figures()
    shape_final_figures()
    print(sorted(p.name for p in OUT.iterdir()))
