"""The one data property the results depend on: no image in training has a
near copy (or a mirrored copy) in validation or test."""
from pathlib import Path

import numpy as np
import pytest

from fer.data import near_pairs, phash

DATASET = Path("data/processed/dataset.npz")


@pytest.mark.skipif(not DATASET.exists(), reason="run scripts/03_build_dataset.py first")
def test_no_duplicates_across_splits():
    d = np.load(DATASET)
    bits = phash(d["images"])
    pairs = np.concatenate([near_pairs(bits, max_dist=5), near_pairs(bits, phash(d["images"][:, :, ::-1]), max_dist=5)])
    split = d["split"]
    assert not (split[pairs[:, 0]] != split[pairs[:, 1]]).any()


@pytest.mark.skipif(not DATASET.exists(), reason="run scripts/03_build_dataset.py first")
def test_every_class_in_every_split():
    d = np.load(DATASET)
    for s in range(3):
        assert set(d["label"][d["split"] == s]) == set(range(7))
