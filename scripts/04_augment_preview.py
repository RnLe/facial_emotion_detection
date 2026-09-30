"""Show what augmentation does to a few training faces, and how fast it runs."""
import time

import numpy as np
import torch

from fer.augment import augment
from fer.data import CLASSES, sheet

d = np.load("data/processed/dataset.npz")
train = np.where(d["split"] == 0)[0]
# FER2013 faces only: RAF-DB images may not be redistributed
fer_train = train[d["source"][train] == 0]
rng = np.random.default_rng(1)
pick = rng.choice(fer_train, 6, replace=False)
x = torch.tensor(d["images"][pick], device="cuda").float().div(255)[:, None]
g = torch.Generator(device="cuda").manual_seed(0)
rows = [x] + [augment(x, 1.0, g) for _ in range(7)]
grid = torch.stack(rows, 1).reshape(-1, 48, 48).mul(255).byte().cpu().numpy()
titles = [(CLASSES[d["label"][i]] if k == 0 else "") for i in pick for k in range(8)]
sheet(grid, titles, "results/data/augment_preview.png", cols=8)

big = torch.tensor(d["images"][train], device="cuda").float().div(255)[:, None]
torch.cuda.synchronize()
t = time.perf_counter()
for _ in range(10):
    augment(big, 1.0)
torch.cuda.synchronize()
print(f"augmenting all {len(train)} training images: {(time.perf_counter() - t) / 10 * 1000:.1f} ms")
