"""Ways to make a trained network smaller or faster, each on its own.

- low_rank: output-based low rank (Zhang et al. 2016, the linear version). A layer's outputs
  on training faces are projected on their top principal directions: the layer with r
  output channels, then a 1x1 conv (or dense layer) back to the full width. The outputs are
  scaled to unit variance per channel first, which is what the batch norm after the layer
  (or a later one, in DenseNet) sees.
- tucker2: Tucker-2 from the weights alone (Kim et al. 2016): a 1x1 conv into R_in channels,
  the k x k core, a 1x1 conv out of R_out channels. Factors from a truncated HOSVD.
- spatial: a k x k conv as a k x 1 conv into R channels and a 1 x k conv out (Jaderberg et
  al. 2014, scheme 2), from the SVD of the weights reshaped to (C_in k) x (C_out k), which
  is the best such split in the squared-error sense (Tai et al. 2016).
- int8: post-training static quantisation for the CPU (PyTorch FX, x86 backend), calibrated
  on training faces. Conv, batch norm and ReLU are fused first. PyTorch gives every input of
  a concatenation one shared scale; DenseNet concatenates up to 18 feature groups of very
  different size, so keep_float=(torch.cat,) leaves the concatenations in float.

The ranks come from one knob, tau: the share of the squared singular values (or of the
output variance) a layer keeps. A layer is only replaced when that saves parameters.
"""
import copy
import io

import torch
from torch import nn


def energy_rank(ev, tau):
    """Smallest rank whose leading values (sorted, descending) hold a share tau of the sum."""
    c = ev.cumsum(0) / ev.sum()
    return min(int((c < tau).sum()) + 1, len(ev))


def layers(model):
    """Every conv and dense layer, with its parent module and attribute name for replacing it."""
    found = []
    for parent_name, parent in model.named_modules():
        for attr, child in parent.named_children():
            if isinstance(child, (nn.Conv2d, nn.Linear)):
                found.append((f"{parent_name}.{attr}".lstrip("."), parent, attr, child))
    return found


@torch.no_grad()
def output_stats(model, x, batch_size=128):
    """Mean and covariance of every conv and dense layer's outputs (spatial positions count as
    samples), in float64."""
    sums = {}

    def hook(name):
        def f(m, i, o):
            z = o.detach().double()
            z = z.permute(0, 2, 3, 1).reshape(-1, z.shape[1]) if z.dim() == 4 else z
            s = sums.setdefault(name, [0, 0, 0])
            s[0], s[1], s[2] = s[0] + z.T @ z, s[1] + z.sum(0), s[2] + len(z)
        return f

    hooks = [m.register_forward_hook(hook(name)) for name, _, _, m in layers(model)]
    model.eval()
    for k in range(0, len(x), batch_size):
        model(x[k:k + batch_size])
    for h in hooks:
        h.remove()
    stats = {}
    for name, (S, s, n) in sums.items():
        mu = s / n
        stats[name] = (mu, S / n - torch.outer(mu, mu))
    return stats


def standardised_pca(mu, cov):
    """Principal directions of the outputs after scaling each channel to unit variance."""
    sd = cov.diagonal().clamp_min(0).sqrt()
    sd = sd.clamp_min(1e-3 * sd.max())  # dead channels would otherwise divide by ~0
    ev, q = torch.linalg.eigh(cov / torch.outer(sd, sd))
    return ev.flip(0).clamp_min(0), q.flip(1), sd


def low_rank_layer(m, mu, cov, r):
    """y = Wx + b becomes y' = D^-1 Q Q^T D (y - mu) + mu, with D = diag(1 / sd) and Q the
    top r directions: a layer with r outputs, then one back to the full width."""
    _, q, sd = standardised_pca(mu, cov)
    q = q[:, :r]
    down, up = q.T / sd, sd[:, None] * q
    w = m.weight.detach().double()
    co = w.shape[0]
    b = m.bias.detach().double() if m.bias is not None else torch.zeros(co, dtype=w.dtype, device=w.device)
    if isinstance(m, nn.Conv2d):
        assert m.groups == 1
        first = nn.Conv2d(m.in_channels, r, m.kernel_size, m.stride, m.padding, m.dilation)
        second = nn.Conv2d(r, co, 1)
        first.weight.data = (down @ w.flatten(1)).reshape(r, *w.shape[1:]).float()
        second.weight.data = up.reshape(co, r, 1, 1).float()
    else:
        first, second = nn.Linear(m.in_features, r), nn.Linear(r, co)
        first.weight.data, second.weight.data = (down @ w).float(), up.float()
    first.bias.data, second.bias.data = (down @ (b - mu)).float(), mu.float()
    return nn.Sequential(first, second).to(m.weight.device)


def tucker2_layer(m, r_in, r_out):
    w = m.weight.detach().double()
    co, ci, kh, kw = w.shape
    u_out = torch.linalg.svd(w.reshape(co, -1), full_matrices=False)[0][:, :r_out]
    u_in = torch.linalg.svd(w.transpose(0, 1).reshape(ci, -1), full_matrices=False)[0][:, :r_in]
    core = torch.einsum("oikl,or,is->rskl", w, u_out, u_in)
    first = nn.Conv2d(ci, r_in, 1, bias=False)
    mid = nn.Conv2d(r_in, r_out, (kh, kw), m.stride, m.padding, m.dilation, bias=False)
    last = nn.Conv2d(r_out, co, 1, bias=m.bias is not None)
    first.weight.data = u_in.T.reshape(r_in, ci, 1, 1).float()
    mid.weight.data = core.float()
    last.weight.data = u_out.reshape(co, r_out, 1, 1).float()
    if m.bias is not None:
        last.bias.data = m.bias.detach().clone()
    return nn.Sequential(first, mid, last).to(m.weight.device)


def tucker2_ranks(m, tau):
    w = m.weight.detach().double()
    co, ci = w.shape[:2]
    s_out = torch.linalg.svdvals(w.reshape(co, -1))
    s_in = torch.linalg.svdvals(w.transpose(0, 1).reshape(ci, -1))
    return energy_rank(s_in**2, tau), energy_rank(s_out**2, tau)


def spatial_matrix(w):
    """W[o, i, y, x] as a matrix M[(i, y), (o, x)]: a rank-R factorisation of M is a k x 1
    conv into R channels followed by a 1 x k conv."""
    co, ci, kh, kw = w.shape
    return w.permute(1, 2, 0, 3).reshape(ci * kh, co * kw)


def spatial_layer(m, r):
    w = m.weight.detach().double()
    co, ci, kh, kw = w.shape
    u, s, vh = torch.linalg.svd(spatial_matrix(w), full_matrices=False)
    root = s[:r].sqrt()
    vert = nn.Conv2d(ci, r, (kh, 1), (m.stride[0], 1), (m.padding[0], 0), bias=False)
    hori = nn.Conv2d(r, co, (1, kw), (1, m.stride[1]), (0, m.padding[1]), bias=m.bias is not None)
    vert.weight.data = (u[:, :r] * root).T.reshape(r, ci, kh, 1).float()
    hori.weight.data = (vh[:r].T * root).reshape(co, kw, r).permute(0, 2, 1).reshape(co, r, 1, kw).float()
    if m.bias is not None:
        hori.bias.data = m.bias.detach().clone()
    return nn.Sequential(vert, hori).to(m.weight.device)


def compress(model, method, tau, stats=None):
    """A copy of model with every layer replaced where `method` at strength tau saves
    parameters. Returns the copy and the number of layers replaced."""
    model = copy.deepcopy(model)
    replaced = 0
    for name, parent, attr, m in layers(model):
        w = m.weight
        is_kxk = isinstance(m, nn.Conv2d) and m.kernel_size[0] > 1 and m.kernel_size[1] > 1
        if method == "low_rank":
            mu, cov = stats[name]
            r = energy_rank(standardised_pca(mu, cov)[0], tau)
            cost, new = r * (w[0].numel() + w.shape[0]), lambda: low_rank_layer(m, mu, cov, r)
        elif method == "tucker2" and is_kxk:
            ri, ro = tucker2_ranks(m, tau)
            cost, new = w.shape[1] * ri + ri * ro * w[0, 0].numel() + ro * w.shape[0], lambda: tucker2_layer(m, ri, ro)
        elif method == "spatial" and is_kxk:
            r = energy_rank(torch.linalg.svdvals(spatial_matrix(w.detach().double())) ** 2, tau)
            co, ci, kh, kw = w.shape
            cost, new = r * (ci * kh + co * kw), lambda: spatial_layer(m, r)
        else:
            continue
        if cost < w.numel():
            setattr(parent, attr, new())
            replaced += 1
    return model, replaced


def int8(model, x_calib, keep_float=(), batch_size=256):
    """Post-training static int8 quantisation for the CPU."""
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

    torch.backends.quantized.engine = "x86"
    m = copy.deepcopy(model).cpu().eval()
    x_calib = x_calib.cpu().contiguous()
    mapping = get_default_qconfig_mapping("x86")
    for op in keep_float:
        mapping.set_object_type(op, None)
    prepared = prepare_fx(m, mapping, (x_calib[:1],))
    with torch.no_grad():
        for k in range(0, len(x_calib), batch_size):
            prepared(x_calib[k:k + batch_size])
    return convert_fx(prepared)


def size_bytes(model):
    """Bytes of the saved weights (works for int8 models too)."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes


def macs(model, x):
    """Multiply-accumulates of the conv and dense layers for one input."""
    total = [0]

    def hook(m, i, o):
        if isinstance(m, nn.Conv2d):
            total[0] += o[0].numel() * m.in_channels // m.groups * m.kernel_size[0] * m.kernel_size[1]
        else:
            total[0] += m.in_features * m.out_features

    hooks = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    with torch.no_grad():
        model.eval()(x[:1])
    for h in hooks:
        h.remove()
    return total[0]
