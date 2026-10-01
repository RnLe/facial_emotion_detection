"""What a network does during long training, beyond loss and accuracy.

- weight_norm: L2 norm of all conv and dense weights (Omnigrok, Liu et al. 2023).
- weight_entropy: absolute weight entropy, -|w| log|w|, averaged over all conv and dense
  weights (Golechha 2024: drops sharply when a network starts to generalise late).
- weight_entropy_rel: the same idea without the scale: the entropy of each layer's |w|
  normalised to sum 1, divided by its maximum, averaged over layers. With batch norm the
  overall size of a layer's weights does not change what it computes, so the absolute
  weight entropy mostly follows the weight norm; this one only sees how evenly the weight
  is spread. 1: all weights equally large; lower: a few weights carry most of it.
- weight_rank: spectral entropy of each conv and dense weight matrix (singular values
  normalised to sum 1), divided by its maximum, averaged over layers. 1: every direction
  is used equally; lower: the layer is closer to low rank, and easier to compress.
- repr_entropy: the same for the covariance of the penultimate features of the validation
  faces (Khanh et al. 2026: it collapses shortly before grokking).
- nc1: neural collapse, the spread of the penultimate features within a class against the
  spread between classes, on training faces (Papyan et al. 2020). Falls towards 0 when
  each class collapses to a point, which happens when training runs far past fitting.
- pred_entropy: mean entropy of the predicted class probabilities (how sure the model is).
"""
import math

import torch
from torch import nn


def entropy(p, dim=-1):
    return -(p * torch.log(p + 1e-12)).sum(dim)


@torch.no_grad()
def weight_measures(model):
    weights = [m.weight.detach().float() for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    flat = torch.cat([w.flatten() for w in weights]).abs()
    ranks, spreads = [], []
    for w in weights:
        s = torch.linalg.svdvals(w.flatten(1))  # a conv kernel as (out, in x k x k)
        ranks.append((entropy(s / s.sum()) / math.log(len(s))).item())
        a = w.abs().flatten()
        spreads.append((entropy(a / a.sum()) / math.log(len(a))).item())
    return {
        "weight_norm": flat.square().sum().sqrt().item(),
        "weight_entropy": (-(flat * torch.log(flat + 1e-12))).mean().item(),
        "weight_entropy_rel": sum(spreads) / len(spreads),
        "weight_rank": sum(ranks) / len(ranks),
    }


def spectral_entropy(features):
    z = features.double() - features.double().mean(0)
    lam = torch.linalg.eigvalsh(z.T @ z / len(z)).clamp_min(0)
    return (entropy(lam / lam.sum()) / math.log(len(lam))).item()


def neural_collapse(features, labels, classes=7):
    f = features.double()
    mu = torch.stack([f[labels == c].mean(0) for c in range(classes)])
    within = (f - mu[labels]).T @ (f - mu[labels]) / len(f)
    between = (mu - f.mean(0)).T @ (mu - f.mean(0)) / classes
    return (torch.trace(within @ torch.linalg.pinv(between)) / classes).item()


@torch.no_grad()
def predict_with_features(model, data, idx, batch_size=1024):
    """Class probabilities, and the input of the last dense layer, for the faces in idx."""
    head = [m for m in model.modules() if isinstance(m, nn.Linear)][-1]
    feats, probs = [], []
    hook = head.register_forward_pre_hook(lambda m, inp: feats.append(inp[0].float()))
    model.eval()
    for k in range(0, len(idx), batch_size):
        x, _ = data.batch(idx[k:k + batch_size])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            probs.append(model(x).float().softmax(1))
    hook.remove()
    return torch.cat(probs), torch.cat(feats)
