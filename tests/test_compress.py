"""At full rank every decomposition must give back the original layer."""
import torch
from torch import nn

from fer.compress import compress, low_rank_layer, output_stats, spatial_layer, tucker2_layer
from fer.models import build


def test_full_rank_is_exact():
    torch.manual_seed(0)
    x = torch.randn(4, 6, 9, 9, dtype=torch.float64)
    for stride in (1, 2):
        conv = nn.Conv2d(6, 5, 3, stride, 1).double()
        y = conv(x)
        torch.testing.assert_close(tucker2_layer(conv, 6, 5).double()(x), y)
        torch.testing.assert_close(spatial_layer(conv, 15).double()(x), y)
        z = y.permute(0, 2, 3, 1).reshape(-1, 5)
        mu, cov = z.mean(0), torch.cov(z.T, correction=0)
        torch.testing.assert_close(low_rank_layer(conv, mu, cov, 5).double()(x), y)
    fc = nn.Linear(6, 4).double()
    v = torch.randn(32, 6, dtype=torch.float64)
    y = fc(v)
    torch.testing.assert_close(low_rank_layer(fc, y.mean(0), torch.cov(y.T, correction=0), 4).double()(v), y)


def test_compress_keeps_shapes():
    torch.manual_seed(0)
    model = build("vgg", dropout=0.0, width=0.25).eval()
    x = torch.randn(8, 1, 48, 48)
    stats = output_stats(model, x)
    for method in ("low_rank", "tucker2", "spatial"):
        small, replaced = compress(model, method, 0.9, stats)
        assert replaced > 0 and small(x).shape == (8, 7)
        assert sum(p.numel() for p in small.parameters()) < sum(p.numel() for p in model.parameters())
