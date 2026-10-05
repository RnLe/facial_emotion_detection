"""Part two, phase 6: the candidates against the targets, scored once.

For every candidate (3 seeds each, FMAE once): test accuracy and macro-F1 with the paired
bootstrap interval of the accuracy difference to the ResNet-18 winner (the same resampled
test faces for both, averaged over seeds), the parts, the official RAF-DB test set at
48 px grayscale (rafdb.npz), and size, compute and speed (one face and 256 faces on the
GPU, one face on one CPU thread).
"""
import json
from pathlib import Path

import numpy as np
import torch

from fer.compress import macs, speed
from fer.models.mirror import MirrorConv
from fer.finetune import predict as predict_reference
from fer.models import build
from fer.reference import load as load_reference
from fer.runs import RUNS, record_path
from fer.train import GPUData, predict

MODEL_ARGS = ("dropout", "drop_path", "width", "depth", "stages", "readout", "match")
CANDIDATES = {
    # name: (track, stage, model, iteration names)
    "resnet": ("scratch", "final", "resnet", ["0", "1", "2"]),
    "masked": ("scratch", "masked", "resnet", ["finetune_0", "finetune_1", "finetune_2"]),
    "mirror_compute": ("scratch", "mirror", "mirror", ["compute_0", "compute_1", "compute_2"]),
    "mirror": ("scratch", "mirror", "mirror", ["params_0", "params_1", "params_2"]),
    "mirror_self_distill": ("scratch", "mirror", "mirror", ["self_distill_0", "self_distill_1", "self_distill_2"]),
    "resnet_distill_fmae": ("open", "opt", "resnet", ["distill_fmae_0", "distill_fmae_1", "distill_fmae_2"]),
    "mirror_distill_fmae": ("open", "mirror", "mirror", ["distill_fmae_0", "distill_fmae_1", "distill_fmae_2"]),
}


def load_model(stage, model, name):
    rec = record_path(stage, model, name)
    cfg = json.loads(rec.read_text())["config"]
    net = build(cfg["model"], **{k: cfg[k] for k in MODEL_ARGS if k in cfg}).cuda().to(memory_format=torch.channels_last)
    net.load_state_dict(torch.load(rec.with_suffix(".pt"), map_location="cuda"))
    return net.eval()


def count_macs(net, x):
    """fer.compress.macs plus the mirror network's convs (which call conv2d themselves)."""
    extra = [0]

    def hook(m, i, o):
        w = m.full_weight()
        extra[0] += o[0].numel() * w.shape[1] * w.shape[2] * w.shape[3]

    hooks = [m.register_forward_hook(hook) for m in net.modules() if isinstance(m, MirrorConv)]
    total = macs(net, x)
    for h in hooks:
        h.remove()
    return total + extra[0]


def paired_bootstrap(preds, base, true, n=2000, seed=0):
    """Mean accuracy difference over seeds with a 95% interval from resampled test faces."""
    rng = np.random.default_rng(seed)
    a, b = np.array(preds) == true, np.array(base) == true
    diff = lambda i: a[:, i].mean() - b[:, i].mean()
    draws = [diff(rng.integers(0, len(true), len(true))) for _ in range(n)]
    return float(diff(np.arange(len(true)))), [float(x) for x in np.percentile(draws, [2.5, 97.5])]


def main():
    d = np.load("data/processed/dataset.npz")
    true = d["label"][d["split"] == 2]
    raf = GPUData("data/processed/rafdb.npz")
    keep = torch.tensor(np.load("data/processed/rafdb.npz")["in_ours"], device="cuda")[raf.idx["test"]]
    x1, x256 = torch.randn(1, 1, 48, 48), torch.randn(256, 1, 48, 48)
    base_preds = [json.loads(record_path("final", "resnet", n).read_text())["test_pred"] for n in ["0", "1", "2"]]
    out = {}
    for name, (track, stage, model, names) in CANDIDATES.items():
        recs = [json.loads(record_path(stage, model, n).read_text()) for n in names]
        row = {"track": track, "seeds": len(recs)}
        for k in ["test", "test_fer", "test_raf"]:
            accs = [r[k]["accuracy"] for r in recs]
            row[k] = float(np.mean(accs))
            row[f"{k}_std"] = float(np.std(accs))
        row["test_macro_f1"] = float(np.mean([r["test"]["macro_f1"] for r in recs]))
        row["val"] = float(np.mean([r["val"]["accuracy"] for r in recs]))
        row["gain_vs_resnet"], row["gain_ci"] = paired_bootstrap([r["test_pred"] for r in recs], base_preds, true)
        raf_acc, raf_dedup = [], []
        for n in names:
            prob, t = predict(load_model(stage, model, n), raf, "test")
            raf_acc.append(float((prob.argmax(1) == t).float().mean()))
            raf_dedup.append(float((prob.argmax(1)[keep] == t[keep]).float().mean()))
        row["rafdb_official_48px"], row["rafdb_official_48px_dedup"] = float(np.mean(raf_acc)), float(np.mean(raf_dedup))
        net = load_model(stage, model, names[0]).float().to(memory_format=torch.contiguous_format)
        row["params"] = sum(p.numel() for p in net.parameters())
        row["macs"] = count_macs(net.cpu(), x1)
        row.update(speed(net, x1, x256))
        out[name] = row
        print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
    # FMAE fine-tuned on our data: the open-track target, for size and speed next to the rest
    fm, spec = load_reference("fmae")
    state = torch.load(RUNS / "references" / "fmae" / "finetune_bf16.pt", map_location="cpu")
    fm.load_state_dict({k: v.float() for k, v in state.items()})
    fm = fm.cuda().eval()
    ref = json.loads(Path("results/references.json").read_text())["fmae"]["finetune"]
    xs = (torch.rand(256, 1, 48, 48, device="cuda") * 255).to(torch.uint8)
    import time
    for _ in range(3):
        predict_reference(fm, spec, xs)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(5):
        predict_reference(fm, spec, xs)
    torch.cuda.synchronize()
    out["fmae_finetuned"] = {"track": "open", "test": ref["test"]["accuracy"], "test_fer": ref["test_fer"]["accuracy"],
                             "test_raf": ref["test_raf"]["accuracy"], "test_macro_f1": ref["test"]["macro_f1"],
                             "rafdb_official_48px": ref["rafdb_gray48"]["test"]["accuracy"],
                             "params": sum(p.numel() for p in fm.parameters()), "macs": 61.6e9,  # timm's count for ViT-L/16 at 224 px
                             "gpu_256_ms_bf16": 1e3 * (time.perf_counter() - t) / 5}
    print("fmae_finetuned", out["fmae_finetuned"])
    Path("results/final_part2.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
