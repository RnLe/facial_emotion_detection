"""The seven winners for the browser demo on the portfolio site.

models: each winner (the best seed by validation macro-F1; VGG and DenseNet from the shape
    stage, ResNet-18 from stage 1) becomes one file: a short program of operations, in the
    order the network computes them, and its weights. Conv and linear weights are int8 with
    one scale per output channel (test accuracy within 0.12 points of float for every
    model); batch norm is folded into the conv before it where there is one. A reference
    interpreter in numpy runs the same programs and must match PyTorch; the vectors written
    here check the Rust engine in web/fer-wasm.
samples: FER2013 test faces for the demo (RAF-DB images may not be redistributed), about
    20 per emotion, with every model's probabilities (from the int8 weights, as in the
    browser) and the FER+ votes for the label.
thumb: the animated card images: DenseNet's probabilities on five neutral, happy and
    surprise faces, each followed by its occlusion map over the face (where greying out a
    patch lowers its confidence most), in a 16:9 layout (the landing list and the case
    study's lead) and a taller one (the index card).
site: the layout the profile serves (runs/web/site/): each model file cut into parts
    under 2.5 MiB (the site's checks cap a page asset at 3 MiB), models.json listing the
    parts in the order the browser loads them (smallest first), and the samples.

Output in runs/web/; copied into the profile by hand (see its SOURCE.md there).
"""
import argparse
import json
import math
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from fer.data import CLASSES
from fer.models import build
from fer.models.convnext import ConvNeXtBlock, LayerNorm2d
from fer.models.densenet import DenseBlock
from fer.models.resnet import BasicBlock
from fer.models.transformer import Block
from fer.runs import record_path
from fer.train import MEAN, STD, GPUData

OUT = Path("runs/web")
STAGES = {"cnn": "final", "vgg": "shape_final", "resnet": "final", "densenet": "shape_final",
          "vit": "final", "convnext": "final", "cct": "final"}
ARGS = ("dropout", "drop_path", "width", "depth", "stages")
NONE = 0xFFFFFFFF
OPS = ["conv", "affine", "relu", "gelu", "maxpool", "avgpool", "gap", "flatten", "linear", "ln",
       "save", "load", "add", "cat", "cls", "pos", "mha", "take0", "seqpool"]
OP = {name: k for k, name in enumerate(OPS)}


def winner(m):
    stage = STAGES[m]
    seed = max(range(3), key=lambda s: json.loads(record_path(stage, m, s).read_text())["val"]["macro_f1"])
    rec = record_path(stage, m, seed)
    cfg = json.loads(rec.read_text())["config"]
    net = build(m, **{k: cfg[k] for k in ARGS if k in cfg}).eval()
    net.load_state_dict(torch.load(rec.with_suffix(".pt"), map_location="cpu"))
    return net, {"stage": stage, "seed": seed}


class Program:
    """Tensors (f32, or int8 rows with one scale each) and operations on them."""

    def __init__(self):
        self.tensors, self.ops = [], []

    def f32(self, a):
        self.tensors.append(("f32", np.ascontiguousarray(a, dtype=np.float32)))
        return len(self.tensors) - 1

    def i8(self, w):
        w = np.asarray(w, dtype=np.float32)
        rows = w.reshape(w.shape[0], -1)
        scale = np.maximum(np.abs(rows).max(1), 1e-12) / 127
        q = np.clip(np.round(rows / scale[:, None]), -127, 127).astype(np.int8).reshape(w.shape)
        self.tensors.append(("i8", q, scale.astype(np.float32)))
        return len(self.tensors) - 1

    def op(self, name, *args):
        self.ops.append((name, [int(a) for a in args]))

    # building blocks
    def conv(self, conv, bn=None):
        w = conv.weight.detach().numpy()
        b = conv.bias.detach().numpy() if conv.bias is not None else None
        if bn is not None:
            s = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach().numpy()
            w = w * s[:, None, None, None]
            b = (bn.bias - bn.running_mean * bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach().numpy() + (0 if b is None else b * s)
        self.op("conv", self.i8(w), NONE if b is None else self.f32(b), conv.stride[0], conv.padding[0], conv.groups)

    def affine(self, bn):
        s = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach().numpy()
        self.op("affine", self.f32(s), self.f32(bn.bias.detach().numpy() - bn.running_mean.numpy() * s))

    def linear(self, lin):
        self.op("linear", self.i8(lin.weight.detach().numpy()), NONE if lin.bias is None else self.f32(lin.bias.detach().numpy()))

    def ln(self, norm):
        self.op("ln", self.f32(norm.weight.detach().numpy()), self.f32(norm.bias.detach().numpy()), struct.unpack("<I", struct.pack("<f", norm.eps))[0])

    def sequential(self, mods):
        mods = [m for m in mods if not isinstance(m, (nn.Dropout, nn.Identity))]
        i = 0
        while i < len(mods):
            m = mods[i]
            nxt = mods[i + 1] if i + 1 < len(mods) else None
            if isinstance(m, nn.Conv2d):
                if isinstance(nxt, nn.BatchNorm2d):
                    self.conv(m, nxt)
                    i += 1
                else:
                    self.conv(m)
            elif isinstance(m, nn.BatchNorm2d):
                self.affine(m)
            elif isinstance(m, nn.ReLU):
                self.op("relu")
            elif isinstance(m, nn.GELU):
                self.op("gelu")
            elif isinstance(m, nn.MaxPool2d):
                k, s, p = (m.kernel_size, m.stride, m.padding)
                self.op("maxpool", k, s, p)
            elif isinstance(m, nn.AvgPool2d):
                self.op("avgpool", m.kernel_size)
            elif isinstance(m, nn.AdaptiveAvgPool2d):
                assert m.output_size in (1, (1, 1))
                self.op("gap")
            elif isinstance(m, nn.Flatten):
                self.op("flatten")
            elif isinstance(m, nn.Linear):
                self.linear(m)
            elif isinstance(m, LayerNorm2d) or isinstance(m, nn.LayerNorm):
                self.ln(m)
            elif isinstance(m, BasicBlock):
                self.basic_block(m)
            elif isinstance(m, DenseBlock):
                self.dense_block(m)
            elif isinstance(m, ConvNeXtBlock):
                self.convnext_block(m)
            elif isinstance(m, nn.Sequential):
                self.sequential(list(m))
            else:
                raise TypeError(f"no translation for {type(m).__name__}")
            i += 1

    def basic_block(self, b):
        self.op("save", 0)
        self.conv(b.conv1, b.bn1)
        self.op("relu")
        self.conv(b.conv2, b.bn2)
        if isinstance(b.skip, nn.Identity):
            self.op("add", 0)
        else:
            self.op("save", 1)
            self.op("load", 0)
            self.conv(b.skip[0], b.skip[1])
            self.op("add", 1)
        self.op("relu")

    def dense_block(self, block):
        self.op("save", 0)
        for layer in block.layers:
            self.op("load", 0)
            self.affine(layer.norm1)
            self.op("relu")
            self.conv(layer.conv1)
            self.affine(layer.norm2)
            self.op("relu")
            self.conv(layer.conv2)
            self.op("cat", 0)

    def convnext_block(self, b):
        self.op("save", 0)
        self.conv(b.dw)
        self.ln(b.norm)
        self.linear(b.mlp[0])
        self.op("gelu")
        self.linear(b.mlp[2])
        self.op("affine", self.f32(b.gamma.detach().numpy()), NONE)
        self.op("add", 0)

    def encoder(self, enc):
        for blk in enc.blocks:
            assert isinstance(blk, Block)
            a = blk.attn
            self.op("save", 0)
            self.ln(blk.norm1)
            self.op("mha", self.i8(a.in_proj_weight.detach().numpy()), self.f32(a.in_proj_bias.detach().numpy()),
                    self.i8(a.out_proj.weight.detach().numpy()), self.f32(a.out_proj.bias.detach().numpy()), a.num_heads)
            self.op("add", 0)
            self.op("save", 0)
            self.ln(blk.norm2)
            self.linear(blk.mlp[0])
            self.op("gelu")
            self.linear(blk.mlp[3])
            self.op("add", 0)
        self.ln(enc.norm)


def program(name, net):
    p = Program()
    if name in ("cnn", "vgg", "densenet", "convnext"):
        p.sequential(list(net.features) + list(net.head))
    elif name == "resnet":
        p.sequential(list(net.stem) + list(net.stages) + list(net.head))
    elif name == "vit":
        p.conv(net.patch)
        p.op("cls", p.f32(net.cls.detach().numpy().reshape(-1)))
        p.op("pos", p.f32(net.pos.detach().numpy()[0].T))
        p.encoder(net.encoder)
        p.op("take0")
        p.linear(net.head)
    elif name == "cct":
        p.sequential(list(net.tokenizer))
        p.op("pos", p.f32(net.pos.detach().numpy()[0].T))
        p.encoder(net.encoder)
        p.op("seqpool", p.f32(net.pool.weight.detach().numpy().reshape(-1)), p.f32(net.pool.bias.detach().numpy()))
        p.linear(net.head)
    else:
        raise ValueError(name)
    return p


# ---- the file format (little endian), read by web/fer-wasm/src/model.rs ----
# "FERM" u32 version | f32 mean, std | u16 name length, name | u32 tensor count, then per tensor:
# u8 kind (0 f32, 1 int8 rows), u8 ndim, u32 dims..., u32 data offset (int8: then u32 scale
# offset) | u32 op count, then per op: u8 opcode, u8 argument count, u32 arguments... | padding
# to 16 bytes | data. Offsets count from the start of the data.

def write(p, name, path):
    blob = bytearray()

    def put(arr):
        while len(blob) % 16:
            blob.append(0)
        off = len(blob)
        blob.extend(arr.tobytes())
        return off

    table = []
    for t in p.tensors:
        if t[0] == "f32":
            table.append((0, t[1].shape, put(t[1]), None))
        else:
            off = put(t[1])
            table.append((1, t[1].shape, off, put(t[2])))
    head = bytearray(b"FERM") + struct.pack("<I", 1) + struct.pack("<ff", MEAN, STD)
    nb = name.encode()
    head += struct.pack("<H", len(nb)) + nb + struct.pack("<I", len(table))
    for kind, shape, off, soff in table:
        head += struct.pack("<BB", kind, len(shape)) + struct.pack(f"<{len(shape)}I", *shape) + struct.pack("<I", off)
        if kind == 1:
            head += struct.pack("<I", soff)
    head += struct.pack("<I", len(p.ops))
    for opname, args in p.ops:
        head += struct.pack("<BB", OP[opname], len(args)) + struct.pack(f"<{len(args)}I", *args)
    while len(head) % 16:
        head.append(0)
    path.write_bytes(bytes(head + blob))
    return len(head) + len(blob)


# ---- reference interpreter: the semantics the Rust engine implements ----

def weight(p, k):
    t = p.tensors[k]
    return t[1] if t[0] == "f32" else t[1].astype(np.float32) * t[2].reshape(-1, *[1] * (t[1].ndim - 1))


def conv2d(x, w, b, stride, pad, groups):
    c, h, wd = x.shape
    o, ig, kh, kw = w.shape
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    oh, ow = (h + 2 * pad - kh) // stride + 1, (wd + 2 * pad - kw) // stride + 1
    out = np.zeros((o, oh, ow), np.float32)
    og = o // groups
    for g in range(groups):
        xs = xp[g * ig:(g + 1) * ig]
        cols = np.empty((ig, kh, kw, oh, ow), np.float32)
        for i in range(kh):
            for j in range(kw):
                cols[:, i, j] = xs[:, i:i + stride * oh:stride, j:j + stride * ow:stride]
        out[g * og:(g + 1) * og] = (w[g * og:(g + 1) * og].reshape(og, -1) @ cols.reshape(-1, oh * ow)).reshape(og, oh, ow)
    return out + (b[:, None, None] if b is not None else 0)


def run_reference(p, pixels):
    x = ((pixels.astype(np.float32) / 255 - MEAN) / STD)[None]  # (C, H, W)
    slots = {}
    erf = np.vectorize(math.erf)
    for name, a in p.ops:
        if name == "conv":
            x = conv2d(x, weight(p, a[0]), None if a[1] == NONE else weight(p, a[1]), a[2], a[3], a[4])
        elif name == "affine":
            x = x * weight(p, a[0]).reshape(-1, 1, 1) + (0 if a[1] == NONE else weight(p, a[1]).reshape(-1, 1, 1))
        elif name == "relu":
            x = np.maximum(x, 0)
        elif name == "gelu":
            x = (0.5 * x * (1 + erf(x / math.sqrt(2)))).astype(np.float32)
        elif name == "maxpool":
            k, s, pd = a
            c, h, w = x.shape
            xp = np.pad(x, ((0, 0), (pd, pd), (pd, pd)), constant_values=-np.inf)
            oh, ow = (h + 2 * pd - k) // s + 1, (w + 2 * pd - k) // s + 1
            x = np.max(np.stack([xp[:, i:i + s * oh:s, j:j + s * ow:s] for i in range(k) for j in range(k)]), 0)
        elif name == "avgpool":
            k = a[0]
            c, h, w = x.shape
            x = x[:, :h // k * k, :w // k * k].reshape(c, h // k, k, w // k, k).mean((2, 4))
        elif name == "gap":
            x = x.mean((1, 2), keepdims=True)
        elif name == "flatten":
            x = x.reshape(-1, 1, 1)
        elif name == "linear":
            c, h, w = x.shape
            y = weight(p, a[0]) @ x.reshape(c, -1)
            if a[1] != NONE:
                y = y + weight(p, a[1])[:, None]
            x = y.reshape(-1, h, w)
        elif name == "ln":
            eps = struct.unpack("<f", struct.pack("<I", a[2]))[0]
            mu, var = x.mean(0, keepdims=True), x.var(0, keepdims=True)
            x = (x - mu) / np.sqrt(var + eps) * weight(p, a[0]).reshape(-1, 1, 1) + weight(p, a[1]).reshape(-1, 1, 1)
        elif name == "save":
            slots[a[0]] = x.copy()
        elif name == "load":
            x = slots[a[0]].copy()
        elif name == "add":
            x = x + slots[a[0]]
        elif name == "cat":
            slots[a[0]] = np.concatenate([slots[a[0]], x], 0)
            x = slots[a[0]].copy()
        elif name == "cls":
            c, h, w = x.shape
            x = np.concatenate([weight(p, a[0]).reshape(c, 1), x.reshape(c, -1)], 1)[:, None]
        elif name == "pos":
            c = x.shape[0]
            x = x.reshape(c, -1)[:, None] + weight(p, a[0])[:, None]
        elif name == "mha":
            c = x.shape[0]
            t = x.reshape(c, -1)
            qkv = weight(p, a[0]) @ t + weight(p, a[1])[:, None]
            heads, dh = a[4], c // a[4]
            out = np.empty_like(t)
            for hd in range(heads):
                q, k, v = (qkv[i * c + hd * dh:i * c + (hd + 1) * dh] for i in range(3))
                s = q.T @ k / math.sqrt(dh)
                s = np.exp(s - s.max(1, keepdims=True))
                s /= s.sum(1, keepdims=True)
                out[hd * dh:(hd + 1) * dh] = v @ s.T
            x = (weight(p, a[2]) @ out + weight(p, a[3])[:, None])[:, None]
        elif name == "take0":
            x = x.reshape(x.shape[0], -1)[:, :1, None]
        elif name == "seqpool":
            c = x.shape[0]
            t = x.reshape(c, -1)
            z = weight(p, a[0]) @ t + weight(p, a[1])[0]
            z = np.exp(z - z.max())
            z /= z.sum()
            x = (t @ z).reshape(c, 1, 1)
        else:
            raise ValueError(name)
    return x.reshape(-1)


def quantised_torch(net):
    """The PyTorch model with every conv and linear weight replaced by its int8 rounding,
    the reference for the interpreter (and for the demo's precomputed probabilities)."""
    import copy

    q = copy.deepcopy(net)
    with torch.no_grad():
        for mod in q.modules():
            ws = []
            if isinstance(mod, (nn.Conv2d, nn.Linear)):
                ws.append(mod.weight)
            if isinstance(mod, nn.MultiheadAttention):
                ws += [mod.in_proj_weight]
            for w in ws:
                s = w.abs().flatten(1).amax(1).clamp_min(1e-12) / 127
                s = s.view(-1, *[1] * (w.dim() - 1))
                w.copy_((w / s).round().clamp(-127, 127) * s)
    return q


def models():
    data = GPUData(device="cpu")
    test = data.idx["test"][:64]
    x = data.images[test].numpy()  # (64, 1, 48, 48) uint8
    (OUT / "models").mkdir(parents=True, exist_ok=True)
    (OUT / "vectors").mkdir(parents=True, exist_ok=True)
    summary = {}
    for m in STAGES:
        net, info = winner(m)
        p = program(m, net)
        size = write(p, m, OUT / "models" / f"{m}.ferm")
        q = quantised_torch(net)
        with torch.no_grad():
            ref = q(((torch.tensor(x[:16]).float() / 255 - MEAN) / STD)).numpy()
        ours = np.stack([run_reference(p, x[i, 0]) for i in range(16)])
        err = float(np.abs(ours - ref).max())
        assert err < 2e-3, (m, err)
        np.savez(OUT / "vectors" / f"{m}.npz", pixels=x[:16, 0], logits=ours)
        (OUT / "vectors" / f"{m}.bin").write_bytes(x[:16, 0].tobytes() + ours.astype(np.float32).tobytes())
        summary[m] = {**info, "bytes": size, "ops": len(p.ops), "tensors": len(p.tensors),
                      "params": sum(t.numel() for t in net.parameters()), "max_logit_error_vs_torch": err}
        print(m, summary[m])
    (OUT / "models.json").write_text(json.dumps(summary, indent=1))


def probabilities(x):
    """Every model's class probabilities (int8 weights, as in the browser) for uint8 faces."""
    out = {}
    xt = (torch.tensor(x).float()[:, None] / 255 - MEAN) / STD
    for m in STAGES:
        q = quantised_torch(winner(m)[0])
        with torch.no_grad():
            out[m] = torch.cat([q(xt[k:k + 256]).softmax(1) for k in range(0, len(xt), 256)]).numpy()
    return out


def samples(per_class=20, seed=0):
    d = np.load("data/processed/dataset.npz")
    rng = np.random.default_rng(seed)
    test_fer = np.where((d["split"] == 2) & (d["source"] == 0))[0]
    pick = []
    for c in range(len(CLASSES)):
        members = test_fer[d["label"][test_fer] == c]
        pick += list(rng.choice(members, min(per_class, len(members)), replace=False))
    pick = np.array(sorted(pick))
    x = d["images"][pick]
    probs = probabilities(x)
    (OUT / "samples.u8").write_bytes(x.astype(np.uint8).tobytes())
    meta = {
        "size": 48, "count": int(len(pick)), "classes": CLASSES, "models": list(STAGES),
        "label": d["label"][pick].tolist(), "votes": d["agreement"][pick].tolist(),
        # probabilities in thousandths, per face: model by class
        "probs": np.round(np.stack([probs[m] for m in STAGES], 1) * 1000).astype(int).tolist(),
    }
    (OUT / "samples.json").write_text(json.dumps(meta, separators=(",", ":")))
    acc = {m: float((probs[m].argmax(1) == d["label"][pick]).mean()) for m in STAGES}
    print(len(pick), "faces; accuracy on them", {m: round(v, 3) for m, v in acc.items()})


# FER2013 test faces (dataset indices) for the card image, chosen by eye (no watermarks,
# no well-known faces) among faces that 8 or more of 10 annotators gave the label and that
# DenseNet gets right at 60 to 90%, so the runner-up shows too: happy, surprise, neutral,
# happy, surprise.
THUMB_FACES = [27666, 26992, 27131, 26680, 27744]


def occlusion_maps(faces, size=6):
    """Where DenseNet's answer depends on the face (occlusion, Zeiler & Fergus 2014): each
    6 x 6 patch in turn set to the dataset's mean grey, and the drop in the probability of
    the class DenseNet predicts. Each pixel gets the mean drop over the patches that cover
    it, scaled to 0..1 per face; a rise counts as 0. Int8 weights, as in the browser."""
    net = quantised_torch(winner("densenet")[0])
    x = (torch.tensor(faces).float()[:, None] / 255 - MEAN) / STD
    n = x.shape[-1]
    drop, cover = torch.zeros(len(x), n, n), torch.zeros(n, n)
    with torch.no_grad():
        p = net(x).softmax(1)
        pred = p.argmax(1, keepdim=True)
        p0 = p.gather(1, pred)[:, 0]
        for i in range(n - size + 1):
            for j in range(n - size + 1):
                xo = x.clone()
                xo[..., i:i + size, j:j + size] = 0  # the mean, after normalisation
                drop[:, i:i + size, j:j + size] += (p0 - net(xo).softmax(1).gather(1, pred)[:, 0])[:, None, None]
                cover[i:i + size, j:j + size] += 1
    m = (drop / cover).clamp_min(0)
    return (m / m.amax((1, 2), keepdim=True).clamp_min(1e-9)).numpy()


def thumb(fps=30, move=0.6, hold=2.2, fade=0.4, heat=1.6):
    """DenseNet's class probabilities on five faces, one after another: the face changes
    and the bars move to the new values and hold; then the occlusion map (occlusion_maps,
    in viridis) fades in over the face and stays until the next face. The face and the map
    keep their 48 x 48 pixels, each drawn as a square block. Two layouts, since a card
    shows the image small: the landing list's 16:9 frame (face beside the bars, only the
    leading emotion named; also the case study's lead) and the index card's taller column
    (face above the bars, every emotion named). Drawn at twice the size and halved per
    frame, on the site's paper colour, in Inter."""
    from matplotlib import colormaps
    from PIL import Image, ImageDraw, ImageFont

    d = np.load("data/processed/dataset.npz")
    faces = d["images"][THUMB_FACES]
    probs = probabilities(faces)["densenet"]
    assert (probs.argmax(1) == d["label"][THUMB_FACES]).all()
    maps = occlusion_maps(faces)
    # the face in grey under the map in viridis, pixel by pixel
    gray = faces[..., None] / 255
    heated = np.uint8(np.round(255 * (0.42 * gray + 0.58 * colormaps["viridis"](maps)[..., :3])))
    paper, line, ink, muted = (247, 246, 242), (213, 220, 225), (24, 33, 41), (82, 96, 107)
    green = (46, 138, 95)
    pale = tuple(round(g * 0.38 + p * 0.62) for g, p in zip(green, paper))
    font = lambda weight, size: ImageFont.truetype(f"/usr/share/fonts/opentype/inter/Inter-{weight}.otf", size)
    S = 2
    layouts = {
        # name: (width, height, face box (x, y, side: a multiple of 48), chart box (x0, top, x1, base), labels)
        "thumb": (960, 540, (40, 54, 432), (520, 150, 920, 480), False),
        "thumb_card": (720, 800, (168, 34, 384), (60, 486, 660, 712), True),
    }
    ease = lambda t: 4 * t ** 3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2
    for name, (width, height, (fx, fy, side), (x0, top, x1, base), labels) in layouts.items():
        W, H = width * S, height * S
        fx, fy, side, x0, top, x1, base = (v * S for v in (fx, fy, side, x0, top, x1, base))
        block = lambda a: Image.fromarray(a).convert("RGB").resize((side, side), Image.NEAREST)
        big = [block(f) for f in faces]
        heat_faces = [block(h) for h in heated]
        mask = Image.new("L", (side, side), 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, side - 1, side - 1), radius=16 * S, fill=255)
        n = len(CLASSES)
        gap = (12 if labels else 16) * S
        bw = (x1 - x0 - gap * (n - 1)) / n
        label_font, strong_font = font("Medium", 21 * S), font("SemiBold", 21 * S)
        value_font = font("SemiBold", (30 if labels else 34) * S)
        title_font = font("SemiBold", 40 * S)

        def frame(face, p, winner):
            im = Image.new("RGB", (W, H), paper)
            im.paste(face, (fx, fy), mask)
            dr = ImageDraw.Draw(im)
            dr.line((x0, base, x1, base), fill=line, width=2 * S)
            for k, c in enumerate(CLASSES):
                bx = x0 + k * (bw + gap)
                h = (base - top) * float(p[k])
                dr.rounded_rectangle((bx, base - max(h, 3 * S), bx + bw, base), radius=6 * S, fill=green if k == winner else pale)
                if labels:
                    f = strong_font if k == winner else label_font
                    tw = dr.textlength(c, font=f)
                    dr.text((bx + bw / 2 - tw / 2, base + 14 * S), c, font=f, fill=ink if k == winner else muted)
                if k == winner:
                    txt = f"{round(float(p[k]) * 100)}%"
                    tw = dr.textlength(txt, font=value_font)
                    dr.text((bx + bw / 2 - tw / 2, base - h - 46 * S), txt, font=value_font, fill=green)
            if not labels:
                # the 16:9 frame names only the leading emotion, large, above the bars
                dr.text((x0, fy + 6 * S), CLASSES[winner], font=title_font, fill=ink)
            return im.reduce(S)

        frames, durations = [], []
        tick = round(1000 / fps)
        steps, fade_steps = round(move * fps), round(fade * fps)
        for i in range(len(faces)):
            a, b = (i - 1) % len(faces), i
            # from the last face under its map to the next face; the bars move with it
            for st in range(1, steps + 1):
                t = ease(st / steps)
                p = probs[a] * (1 - t) + probs[b] * t
                frames.append(frame(Image.blend(heat_faces[a], big[b], t), p, int(p.argmax())))  # the highlight follows the tallest bar
                durations.append(tick)
            durations[-1] = round(hold * 1000)
            # the map fades in and stays; the bars stay
            for st in range(1, fade_steps + 1):
                frames.append(frame(Image.blend(big[b], heat_faces[b], ease(st / fade_steps)), probs[b], int(probs[b].argmax())))
                durations.append(tick)
            durations[-1] = round(heat * 1000)
        path = OUT / f"{name}.webp"
        # lossless: the pixel blocks stay sharp, and the file is no larger than at quality 86
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0, lossless=True, quality=100, method=6)
        frames[steps - 1].save(OUT / f"{name}_still.png")
        print(path, path.stat().st_size, "bytes,", len(frames), "frames,", round(sum(durations) / 1000, 1), "s")


PART = 2_500_000
COLORS_ORDER = ["cnn", "vgg", "resnet", "densenet", "vit", "convnext", "cct"]  # the study's order


def site():
    import shutil

    out = OUT / "site"
    if out.exists():
        shutil.rmtree(out)
    (out / "models").mkdir(parents=True)
    info = json.loads((OUT / "models.json").read_text())
    models = []
    for m in COLORS_ORDER:
        blob = (OUT / "models" / f"{m}.ferm").read_bytes()
        parts = []
        for k in range(0, len(blob), PART):
            name = f"models/{m}.ferm.{k // PART}"
            (out / name).write_bytes(blob[k:k + PART])
            parts.append(name)
        models.append({"name": m, "bytes": len(blob), "parts": parts, "stage": info[m]["stage"], "seed": info[m]["seed"]})
    load_order = sorted(models, key=lambda r: r["bytes"])
    (out / "models.json").write_text(json.dumps({"order": COLORS_ORDER, "models": load_order}, indent=1))
    shutil.copy(OUT / "samples.u8", out / "samples.u8")
    shutil.copy(OUT / "samples.json", out / "samples.json")
    total = sum(r["bytes"] for r in models)
    print(f"{sum(len(r['parts']) for r in models)} parts, {total / 1e6:.1f} MB of models")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["models", "samples", "thumb", "site"])
    a = ap.parse_args()
    {"models": models, "samples": samples, "thumb": thumb, "site": site}[a.step]()
