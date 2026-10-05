"""Loading, cleaning and unifying the raw datasets.

Everything ends up as 48x48 grayscale uint8 images with one of seven labels
(in CLASSES order), a source tag, and a split.
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

RAW = Path("data/raw")
CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
SIZE = 48

# FER2013's own labels: 0 angry, 1 disgust, 2 fear, 3 happy, 4 sad, 5 surprise, 6 neutral
FER_TO_CLASS = [0, 1, 2, 3, 5, 6, 4]
# FER+ vote columns, in the file's order
FERPLUS_EMOTIONS = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"]
FERPLUS_TO_CLASS = {"neutral": 4, "happiness": 3, "surprise": 6, "sadness": 5, "anger": 0, "disgust": 1, "fear": 2}
# RAF-DB labels: 1 surprise, 2 fear, 3 disgust, 4 happiness, 5 sadness, 6 anger, 7 neutral
RAF_TO_CLASS = {1: 6, 2: 2, 3: 1, 4: 3, 5: 5, 6: 0, 7: 4}


def load_fer2013():
    """All 35,887 FER2013 images, their original labels (mapped), and FER+ votes."""
    df = pd.read_csv(RAW / "fer2013/fer2013.csv")
    images = np.stack([np.array(p.split(), dtype=np.uint8).reshape(SIZE, SIZE) for p in df.pixels])
    votes = pd.read_csv(RAW / "FERPlus/fer2013new.csv")
    assert (votes.Usage.values == df.Usage.values).all(), "FER+ rows must line up with FER2013"
    return {
        "images": images,
        "fer_label": np.array(FER_TO_CLASS)[df.emotion.values],
        "usage": df.Usage.values,
        "votes": votes[FERPLUS_EMOTIONS + ["unknown", "NF"]].to_numpy(),
        "ferplus_removed": votes["Image name"].isna().values,
    }


def load_rafdb(size=None, color=False):
    """RAF-DB basic set (aligned 100x100 RGB) as grayscale (or RGB), optionally resized."""
    out = {"images": [], "label": [], "usage": [], "name": []}
    for usage in ["train", "test"]:
        labels = pd.read_csv(RAW / f"rafdb/{usage}_labels.csv")
        for name, label in zip(labels.image, labels.label):
            path = str(RAW / f"rafdb/DATASET/{usage}/{label}/{name}")
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) if color else cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if size:
                img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
            out["images"].append(img)
            out["label"].append(RAF_TO_CLASS[label])
            out["usage"].append(usage)
            out["name"].append(name)
    return {k: np.array(v) if k != "images" else np.stack(v) for k, v in out.items()}


def phash(images):
    """64-bit perceptual hash per image (DCT of a 32x32 copy, 8x8 low frequencies vs their median)."""
    bits = []
    for img in images:
        small = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
        low = cv2.dct(small)[:8, :8].flatten()
        bits.append(low > np.median(low[1:]))
    return np.array(bits)


def hamming(a, b):
    """All pairwise Hamming distances between two sets of bit hashes, on the GPU."""
    import torch

    x = torch.tensor(a, device="cuda", dtype=torch.float16) * 2 - 1
    y = torch.tensor(b, device="cuda", dtype=torch.float16) * 2 - 1
    return ((x.shape[1] - x @ y.T) / 2).round().to(torch.int16)


def near_pairs(a, b=None, max_dist=6, chunk=4096):
    """Pairs (i, j, distance), i < j, where hash a[i] and hash b[j] differ in at most
    max_dist bits. With b the mirrored hashes of the same images, this finds mirrored
    copies; an image is never paired with itself."""
    import torch

    b = a if b is None else b
    found = []
    for start in range(0, len(a), chunk):
        d = hamming(a[start:start + chunk], b)
        i, j = torch.nonzero(d <= max_dist, as_tuple=True)
        dist = d[i, j].long()
        i = i + start
        keep = i != j
        found.append(torch.stack([torch.minimum(i, j), torch.maximum(i, j), dist], 1)[keep].cpu())
    pairs = torch.cat(found).numpy()
    # keep the closest distance once per unordered pair
    order = np.lexsort((pairs[:, 2], pairs[:, 1], pairs[:, 0]))
    pairs = pairs[order]
    first = np.ones(len(pairs), bool)
    first[1:] = (pairs[1:, :2] != pairs[:-1, :2]).any(1)
    return pairs[first]


def sheet(images, titles, path, cols=10, scale=1.1):
    """Save a grid of grayscale images with small titles."""
    import matplotlib.pyplot as plt

    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * scale, rows * scale * 1.18))
    for ax in np.array(axes).flat:
        ax.axis("off")
    for ax, img, title in zip(np.array(axes).flat, images, titles):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=6)
    fig.tight_layout(pad=0.2)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
