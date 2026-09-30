"""Every model maps a batch of 48x48 grayscale faces to seven scores, and has about the
size its reference configuration says (a wrong layer count shows up here first)."""
import pytest
import torch

from fer.models import MODELS, build

# (min, max) parameters in millions, from the reference configurations
SIZES = {
    "cnn": (1.0, 1.5), "vgg": (6, 8), "resnet": (10.5, 11.5), "densenet": (0.7, 0.9),
    "vit": (3.5, 4.0), "convnext": (3.2, 3.8), "cct": (3.6, 4.2), "resnet_pretrained": (10.5, 11.5),
}


@pytest.mark.parametrize("name", list(MODELS))
def test_shape_and_size(name):
    model = build(name).eval()
    assert model(torch.randn(2, 1, 48, 48)).shape == (2, 7)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    lo, hi = SIZES[name]
    assert lo <= params <= hi, f"{name}: {params:.2f}M parameters"
