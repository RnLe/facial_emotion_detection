"""At full rank every decomposition must give back the original layer."""
import torch
from torch import nn

from fer.compress import allocate, compress, low_rank_layer, output_stats, spatial_layer, tucker2_layer
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


def test_allocate_meets_the_budget_where_it_costs_least():
    # layer a is sensitive, layer b is not: the cut should come from b
    table = {"a": {"params": 100, "options": [(1, 10, 1.0), (5, 50, 0.1)]},
             "b": {"params": 100, "options": [(1, 10, 0.01), (5, 50, 0.001)]}}
    ranks, params = allocate(table, total=210, target=130)
    assert params <= 130 and ranks == {"b": 1}
    assert allocate(table, total=210, target=10) is None
