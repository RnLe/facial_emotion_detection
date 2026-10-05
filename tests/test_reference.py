"""The reference models' outputs follow RAF-DB's label order, and the RAF-DB benchmark
file keeps its validation faces out of the official test set."""
from pathlib import Path

import numpy as np
import pytest
import torch

from fer.data import CLASSES
from fer.reference import RAF_ORDER, to_ours

RAFDB = Path("data/processed/rafdb.npz")


def test_raf_order_maps_names():
    raf_names = ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"]  # RAF-DB labels 1..7
    assert [CLASSES[c] for c in RAF_ORDER] == raf_names


def test_to_ours_moves_each_output():
    logits = torch.eye(7)
    out = to_ours(logits, RAF_ORDER)
    assert out.argmax(1).tolist() == RAF_ORDER


@pytest.mark.skipif(not RAFDB.exists(), reason="run scripts/17_rafdb.py build first")
def test_rafdb_splits():
    d = np.load(RAFDB)
    official_test = np.char.startswith(d["name"].astype(str), "test_")
    assert (official_test == (d["split"] == 2)).all()
    assert official_test.sum() == 3068
    ours = np.load("data/processed/dataset.npz")
    raf_val = ours["original_index"][(ours["source"] == 1) & (ours["split"] == 1)] - 35887
    assert set(np.where(d["split"] == 1)[0]) == set(raf_val)
