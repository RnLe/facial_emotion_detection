"""What limits the architectures (part two, phase 4). Controlled changes to the input,
scored on validation, for our three winners and for FMAE fine-tuned on our data: what does
the stronger model rely on that ours do not?

perturb:
    flip       the mirrored face: how often the prediction changes, per class
    shuffle    the face cut into a k x k grid of tiles, tiles shuffled (k = 2, 3, 4, 6):
               the parts stay, their arrangement goes
    blur       low-pass, Gaussian sigma 1, 2, 3 px (at 48 px)
    highpass   the face minus its sigma 2 blur, mean added back
readout: the ResNet-18 winner with a position-aware head instead of global average
    pooling ("spatial": 4 learned weightings of the 6x6 grid; "flatten": a weight per
    position), 3 seeds. Each is then scored on validation faces shifted by up to 4 px:
    if the gain comes from position, misalignment should cost it more than the baseline.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from fer.data import CLASSES
from fer.finetune import predict as predict_reference
from fer.models import build
from fer.reference import load as load_reference
from fer.metrics import scores
from fer.runs import RUNS, add_to_plan, is_done, record_path
from fer.train import MEAN, STD, GPUData, run

WINNERS = {"vgg": "shape_final", "resnet": "final", "densenet": "shape_final"}
MODEL_ARGS = ("dropout", "drop_path", "width", "depth", "stages")


def load_winner(m, seed=0):
    cfg = json.loads(record_path(WINNERS[m], m, seed).read_text())["config"]
    model = build(m, **{k: cfg[k] for k in MODEL_ARGS if k in cfg}).cuda().to(memory_format=torch.channels_last)
    model.load_state_dict(torch.load(record_path(WINNERS[m], m, seed).with_suffix(".pt"), map_location="cuda"))
    return model.eval()


def fmae_finetuned():
    model, spec = load_reference("fmae")
    state = torch.load(RUNS / "references" / "fmae" / "finetune_bf16.pt", map_location="cpu")
    model.load_state_dict({k: v.float() for k, v in state.items()})
    return model.cuda().eval(), spec


def gaussian(x, sigma):
    r = int(3 * sigma)
    t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-t**2 / (2 * sigma**2))
    k = k / k.sum()
    c = x.shape[1]
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)


def shuffle_tiles(x, k, generator):
    b, c, h, w = x.shape
    th, tw = h // k, w // k
    x = x[:, :, :th * k, :tw * k]
    tiles = x.reshape(b, c, k, th, k, tw).permute(0, 2, 4, 1, 3, 5).reshape(b, k * k, c, th, tw)
    order = torch.argsort(torch.rand(b, k * k, device=x.device, generator=generator), 1)
    tiles = tiles[torch.arange(b, device=x.device)[:, None], order]
    out = tiles.reshape(b, k, k, c, th, tw).permute(0, 3, 1, 4, 2, 5).reshape(b, c, th * k, tw * k)
    return F.interpolate(out, size=(h, w), mode="nearest") if (th * k, tw * k) != (h, w) else out


def variants(x01, generator):
    yield "intact", x01
    yield "flip", x01.flip(-1)
    for k in [2, 3, 4, 6]:
        yield f"shuffle{k}", shuffle_tiles(x01, k, generator)
    for s in [1, 2, 3]:
        yield f"blur{s}", gaussian(x01, s)
    yield "highpass2", (x01 - gaussian(x01, 2) + x01.mean((2, 3), keepdim=True)).clamp(0, 1)


@torch.no_grad()
def predict_ours(model, x01):
    out = []
    for k in range(0, len(x01), 1024):
        x = ((x01[k:k + 1024] - MEAN) / STD).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model(x).float().softmax(1))
    return torch.cat(out)


def load_record(stage, model, name):
    cfg = json.loads(record_path(stage, model, name).read_text())["config"]
    args = {k: cfg[k] for k in MODEL_ARGS + ("readout", "match") if k in cfg}
    net = build(cfg["model"], **args).cuda().to(memory_format=torch.channels_last)
    net.load_state_dict(torch.load(record_path(stage, model, name).with_suffix(".pt"), map_location="cuda"))
    return net.eval()


# later models, seed 0 of each (perturb_more)
MORE = {"mirror_params": ("mirror", "mirror", "params_0"), "mirror_compute": ("mirror", "mirror", "compute_0"),
        "masked": ("masked", "resnet", "finetune_0"), "distill_fmae": ("opt", "resnet", "distill_fmae_0"),
        "self_distill": ("opt", "resnet", "self_distill_0")}


def perturb(more=False):
    data = GPUData()
    idx = data.idx["val"]
    y = data.label[idx]
    x01 = data.images[idx].float().div(255)
    if more:
        models = {k: (lambda x, model=load_record(*v): predict_ours(model, x)) for k, v in MORE.items()
                  if record_path(*v).with_suffix(".pt").exists()}
    else:
        models = {m: (lambda x, model=load_winner(m): predict_ours(model, x)) for m in WINNERS}
        fm, spec = fmae_finetuned()
        models["fmae_finetuned"] = lambda x: predict_reference(fm, spec, (x * 255).round().to(torch.uint8))
    out = {}
    for name, f in models.items():
        g = torch.Generator(device="cuda").manual_seed(0)
        res, base = {}, None
        for v, xv in variants(x01, g):
            p = f(xv).argmax(1)
            res[v] = {"accuracy": float((p == y).float().mean())}
            if v == "intact":
                base = p
            if v == "flip":
                changed = p != base
                res[v]["changed"] = float(changed.float().mean())
                res[v]["changed_by_class"] = {CLASSES[c]: float(changed[y == c].float().mean()) for c in range(7)}
        out[name] = res
        print(name, {v: round(r["accuracy"], 4) for v, r in res.items()}, "flip changed", round(res["flip"]["changed"], 4))
    return out


def shifted(x01, pixels, generator):
    """Each face moved by a random whole-pixel offset up to pixels in each direction."""
    b, _, h, w = x01.shape
    pad = F.pad(x01, (pixels, pixels, pixels, pixels), mode="replicate")
    dy = torch.randint(0, 2 * pixels + 1, (b,), device=x01.device, generator=generator)
    dx = torch.randint(0, 2 * pixels + 1, (b,), device=x01.device, generator=generator)
    rows = (dy[:, None] + torch.arange(h, device=x01.device))[:, :, None]
    cols = (dx[:, None] + torch.arange(w, device=x01.device))[:, None, :]
    return pad[torch.arange(b, device=x01.device)[:, None, None], 0, rows, cols][:, None]


def readout(seeds):
    data = GPUData()
    idx = data.idx["val"]
    y = data.label[idx]
    x01 = data.images[idx].float().div(255)
    out = {}
    for head in ["gap", "spatial", "flatten"]:
        rows = []
        for seed in seeds:
            if head == "gap":
                rec, cfg = record_path("final", "resnet", seed), None
            else:
                rec = record_path("bottleneck", "resnet", f"{head}_{seed}")
                base = json.loads(record_path("final", "resnet", seed).read_text())["config"]
                cfg = {**{k: v for k, v in base.items() if k not in ("stage", "iteration")}, "seed": seed, "readout": head,
                       "stage": "bottleneck", "iteration": f"{head}_{seed}"}
                if not is_done("bottleneck", "resnet", f"{head}_{seed}"):
                    add_to_plan([{"stage": "bottleneck", "model": "resnet", "iteration": f"{head}_{seed}", "epochs": cfg["epochs"]}])
                    run(cfg, data=data, test=True, record=rec)
            r = json.loads(rec.read_text())
            model = build("resnet", **{k: r["config"][k] for k in MODEL_ARGS + ("readout",) if k in r["config"]}).cuda().to(memory_format=torch.channels_last)
            model.load_state_dict(torch.load(rec.with_suffix(".pt"), map_location="cuda"))
            model.eval()
            g = torch.Generator(device="cuda").manual_seed(0)
            row = {"val": r["val"]["accuracy"], "val_macro_f1": r["val"]["macro_f1"], "test": r["test"]["accuracy"]}
            for px in [2, 4]:
                p = predict_ours(model, shifted(x01, px, g)).argmax(1)
                row[f"val_shift{px}"] = float((p == y).float().mean())
            rows.append(row)
            print(head, seed, row)
        out[head] = {"runs": rows, "mean": {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}}
    return out


def merge(key, value):
    path = Path("results/bottleneck.json")
    res = json.loads(path.read_text()) if path.exists() else {}
    res[key] = value
    path.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["perturb", "perturb_more", "readout"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    if a.step == "readout":
        merge(a.step, readout(a.seeds))
    else:
        merge(a.step, perturb(more=a.step == "perturb_more"))
