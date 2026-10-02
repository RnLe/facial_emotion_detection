"""Grokking test on the simple CNN, in the setting where grokking was shown on real images
(Omnigrok, Liu et al. 2023, MNIST): a small training set, weights initialised larger than
usual, AdamW with weight decay, MSE loss on one-hot targets, no augmentation, 1e5 steps.

The simple CNN has no batch norm, so the size of its weights matters and weight decay can
pull it towards a simpler solution (in VGG, batch norm makes the weight size irrelevant,
and the long grok run only produced loss spikes).

- 1,000 training faces, every class in its share of the full training set.
- All conv and dense weights multiplied by alpha after the usual initialisation. Omnigrok
  used alpha 8 on a 3-layer MLP (outputs scaled by 8^3 = 512). The CNN has 5 weight
  layers: alpha 8 scales its outputs by 8^5 = 33,000, and the first steps kill every ReLU
  (constant output, the majority class). Alpha 3 (3^5 = 243) is the closest working match.
- Grid: alpha 3 and 1 times weight decay 0.1, 0, 0.01 (in Omnigrok the time to generalise
  scales with 1 / weight decay, so 0.01 may need more than 1e5 steps). Grokking: with alpha 3
  and weight decay, training accuracy reaches 100% early and validation accuracy rises
  much later, towards the alpha 1 level. Without weight decay it should not rise.
- Logged at log-spaced steps early and every 250 steps after: accuracy and loss on the
  1,000 training faces and on validation, and the measures from fer/measures.py.
- Full float32 (no bfloat16, no TF32): grokking is sensitive to numerical precision.

--model vgg runs the same test on VGG without batch norm (--bf16 for speed; with MSE loss
there is no softmax whose precision could matter). With PyTorch's default initialisation
the signal fades through its 10 weight layers until the output no longer depends on the
face, so alpha is chosen to give the same spread of outputs across faces at step 0 as
the CNN's alpha 3 (3.2): alpha 2.8 gives 3.5, with no dead layer. A checkpoint every
2,500 steps; a larger --steps continues a run.
"""
import os
import argparse
import json
import time

import torch
import torch.nn.functional as F
from torch import nn

from fer.measures import neural_collapse, spectral_entropy, weight_measures
from fer.metrics import scores
from fer.models import build
from fer.runs import add_to_plan, record_path
from fer.train import GPUData, save

p = argparse.ArgumentParser()
p.add_argument("--model", default="cnn", choices=["cnn", "vgg"])
p.add_argument("--bf16", action="store_true")
p.add_argument("--compile", action="store_true")
p.add_argument("--alphas", nargs="+", type=float, default=[3, 1])
p.add_argument("--wds", nargs="+", type=float, default=[0.1, 0.0, 0.01])
p.add_argument("--steps", type=int, default=100_000)
p.add_argument("--n", type=int, default=1000, help="training faces")
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--batch", type=int, default=200)
p.add_argument("--every", type=int, default=250)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False


def subset(data, n, seed):
    """n training faces, every class in its share of the full training set."""
    idx = data.idx["train"]
    lab = data.label[idx]
    g = torch.Generator().manual_seed(seed)
    picks = []
    for c in range(7):
        members = idx[lab == c]
        k = round(n * len(members) / len(idx))
        picks.append(members[torch.randperm(len(members), generator=g)[:k].to(members.device)])
    return torch.cat(picks)


@torch.no_grad()
def outputs(model, x, batch_size=1024):
    """Raw outputs and the input of the last dense layer."""
    head = [m for m in model.modules() if isinstance(m, nn.Linear)][-1]
    feats, outs = [], []
    hook = head.register_forward_pre_hook(lambda m, inp: feats.append(inp[0]))
    model.eval()
    for k in range(0, len(x), batch_size):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
            outs.append(model(x[k:k + batch_size]).float())
    hook.remove()
    model.train()
    return torch.cat(outs), torch.cat(feats).float()


def log_steps(total, every):
    early = {1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200}
    return early | set(range(every, total + 1, every))


def run(alpha, wd, data, train_idx):
    name = f"a{alpha:g}_wd{wd:g}"
    record = record_path("grok", args.model, name)
    checkpoint = record.with_suffix(".ckpt")
    done = json.loads(record.read_text())["history"] if record.exists() else []
    if done and done[-1]["step"] >= args.steps:
        return
    cfg = {"model": args.model, "stage": "grok", "iteration": name, "alpha": alpha, "weight_decay": wd, "lr": args.lr,
           "batch_size": args.batch, "steps": args.steps, "train_faces": len(train_idx), "loss": "mse one-hot",
           "augment": 0.0, "dropout": 0.0, "seed": args.seed, "epochs": args.steps * args.batch // len(train_idx),
           "precision": "bf16" if args.bf16 else "fp32"}
    kwargs = {"dropout": 0.0} if args.model == "cnn" else {"dropout": 0.0, "norm": False}
    cfg.update({k: v for k, v in kwargs.items() if k != "dropout"})
    torch.manual_seed(args.seed)
    model = build(args.model, **kwargs).cuda()
    w0 = weight_measures(model)["weight_norm"]
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                m.weight.mul_(alpha)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=wd)
    fast = torch.compile(model) if args.compile else model

    x_train, y_train = data.batch(train_idx)
    x_val, y_val = data.batch(data.idx["val"])
    target = F.one_hot(y_train, 7).float()
    g = torch.Generator(device="cuda").manual_seed(args.seed)
    first, history = 1, []
    if checkpoint.exists():
        c = torch.load(checkpoint, weights_only=False)
        model.load_state_dict(c["model"])
        opt.load_state_dict(c["opt"])
        g.set_state(c["g"])
        first, history = c["step"] + 1, c["history"]
        print(f"{name}: continuing from step {c['step']}", flush=True)
    state = {"config": cfg, "status": "running", "started": time.time(), "history": history,
             "params": sum(p.numel() for p in model.parameters())}
    save(state, record)
    when = log_steps(args.steps, args.every)
    t0 = time.perf_counter()
    for step in range(first, args.steps + 1):
        b = torch.randint(len(train_idx), (args.batch,), device="cuda", generator=g)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
            out = fast(x_train[b])
        loss = F.mse_loss(out.float(), target[b])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step not in when:
            continue
        out_t, feat_t = outputs(model, x_train)
        out_v, feat_v = outputs(model, x_val)
        trn, val = scores(out_t.softmax(1), y_train), scores(out_v.softmax(1), y_val)
        w = weight_measures(model)
        state["history"].append({
            "epoch": step * args.batch / len(train_idx), "step": step,
            "train_loss": F.mse_loss(out_t, target).item(), "train_accuracy": trn["accuracy"],
            "val_loss": F.mse_loss(out_v, F.one_hot(y_val, 7).float()).item(),
            "val_accuracy": val["accuracy"], "val_macro_f1": val["macro_f1"],
            "gap": trn["accuracy"] - val["accuracy"],
            **w, "weight_norm_rel": w["weight_norm"] / w0,
            "repr_entropy": spectral_entropy(feat_v), "nc1": neural_collapse(feat_t, y_train),
            "lr": args.lr, "seconds": time.perf_counter() - t0,
        })
        t0 = time.perf_counter()
        save(state, record)
        if step % 2500 == 0 or step == args.steps:
            tmp = checkpoint.with_suffix(".tmp")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "g": g.get_state(), "step": step,
                        "history": state["history"]}, tmp)
            os.replace(tmp, checkpoint)
    state["status"], state["finished"] = "done", time.time()
    state["val"] = val
    save(state, record)
    h = state["history"]
    hit = next((x["step"] for x in h if x["train_accuracy"] >= 0.999), None)
    peak = max(h, key=lambda x: x["val_accuracy"])
    print(f"{name}: train 100% at step {hit}, val {100 * h[-1]['val_accuracy']:.1f} at the end, "
          f"peak {100 * peak['val_accuracy']:.1f} at step {peak['step']}, weight norm x{h[-1]['weight_norm_rel']:.2f}", flush=True)


data = GPUData()
train_idx = subset(data, args.n, args.seed)
print(f"{len(train_idx)} training faces, per class {torch.bincount(data.label[train_idx], minlength=7).tolist()}", flush=True)
grid = [(a, wd) for a in args.alphas for wd in args.wds]
add_to_plan([{"model": args.model, "stage": "grok", "iteration": f"a{a:g}_wd{wd:g}", "epochs": args.steps * args.batch // args.n} for a, wd in grid])
for alpha, wd in grid:
    run(alpha, wd, data, train_idx)
