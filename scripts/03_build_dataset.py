"""Clean and unify FER2013 (with FER+ labels) and RAF-DB into one dataset.

Rules, each chosen after looking at the sheets in results/inspect/:
- FER2013 labels come from the FER+ votes. An image stays only if at least 5
  of its 10 annotators agree on one of the seven emotions. This drops
  "not a face", "unknown", contempt, and ambiguous faces.
- Near-constant images (black or grey frames) go.
- Near duplicates (perceptual hash within 5 bits, mirrored copies included)
  form groups. A group whose members disagree on the label goes entirely.
  Otherwise one image stays, preferring test over validation over training,
  so no copy of a test image is left in training.
- FER2013 faces are framed looser than the aligned RAF-DB faces. FER2013 is
  cropped to 90%, centred 2 px lower: the crop that best matches the two
  average faces (results/data/framing.png).
- Splits: FER2013 Training / PublicTest / PrivateTest are train / val / test.
  RAF-DB's train set gives 10% (stratified) to validation; its test set stays.

Writes data/processed/dataset.npz and results/data/summary.json.
"""
import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from fer.data import CLASSES, FERPLUS_EMOTIONS, FERPLUS_TO_CLASS, load_fer2013, load_rafdb, near_pairs, phash

MIN_VOTES = 5
MAX_HASH_DIST = 5
CROP, SHIFT = 0.90, 2.0
SPLITS = {"train": 0, "val": 1, "test": 2}


def crop_fer(img):
    half, centre = 24 * CROP, 24 + SHIFT
    m = np.float32([[1 / CROP, 0, -(24 - half) / CROP], [0, 1 / CROP, -(centre - half) / CROP]])
    return cv2.warpAffine(img, m, (48, 48), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REFLECT)


fer = load_fer2013()
raf = load_rafdb(size=48)
log = {}

# FER2013: FER+ majority label, kept when at least MIN_VOTES agree.
votes = fer["votes"][:, :7]  # the seven basic emotions (contempt, unknown, NF left out)
top = fer["votes"].argmax(1)
agree = fer["votes"].max(1)
keep = (~fer["ferplus_removed"]) & (top < 7) & (agree >= MIN_VOTES)
fer_label = np.array([FERPLUS_TO_CLASS[FERPLUS_EMOTIONS[k]] if k < 7 else -1 for k in top])
log["fer_total"] = len(top)
log["fer_dropped_votes"] = int((~keep).sum())
log["fer_label_changed_by_ferplus"] = int((fer_label[keep] != fer["fer_label"][keep]).sum())

std = fer["images"].reshape(len(top), -1).std(1)
log["fer_dropped_flat"] = int((keep & (std < 3)).sum())
keep &= std >= 3

fer_split = np.array([{"Training": 0, "PublicTest": 1, "PrivateTest": 2}[u] for u in fer["usage"]])
raf_split = np.where(raf["usage"] == "train", 0, 2)
train_idx = np.where(raf_split == 0)[0]
_, val_idx = train_test_split(train_idx, test_size=0.1, stratify=raf["label"][train_idx], random_state=0)
raf_split[val_idx] = 1

images = np.concatenate([np.stack([crop_fer(im) for im in fer["images"]]), raf["images"]])
label = np.concatenate([fer_label, raf["label"]])
split = np.concatenate([fer_split, raf_split])
source = np.concatenate([np.zeros(len(top), int), np.ones(len(raf["label"]), int)])
agreement = np.concatenate([agree, np.full(len(raf["label"]), -1)])
alive = np.concatenate([keep, np.ones(len(raf["label"]), bool)])

# Near duplicates among the surviving images, mirrored copies included.
idx = np.where(alive)[0]
bits = phash(images[idx])
pairs = np.concatenate([near_pairs(bits, max_dist=MAX_HASH_DIST), near_pairs(bits, phash(images[idx][:, :, ::-1]), max_dist=MAX_HASH_DIST)])
parent = np.arange(len(idx))


def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


for a, b, _ in pairs:
    parent[find(a)] = find(b)
groups = {}
for k in range(len(idx)):
    groups.setdefault(find(k), []).append(idx[k])
dup_groups = [g for g in groups.values() if len(g) > 1]
log["duplicate_groups"] = len(dup_groups)
log["images_in_duplicate_groups"] = sum(len(g) for g in dup_groups)
log["duplicate_groups_across_splits"] = sum(len(set(split[g])) > 1 for g in dup_groups)
log["duplicate_groups_label_conflict"] = sum(len(set(label[g])) > 1 for g in dup_groups)
for g in dup_groups:
    alive[g] = False
    if len(set(label[g])) == 1:
        alive[max(g, key=lambda i: (split[i], -i))] = True  # keep the test copy if there is one
log["dropped_duplicates"] = log["images_in_duplicate_groups"] - (log["duplicate_groups"] - log["duplicate_groups_label_conflict"])

sel = np.where(alive)[0]
out = dict(images=images[sel], label=label[sel], split=split[sel], source=source[sel], agreement=agreement[sel], original_index=sel)
Path("data/processed").mkdir(parents=True, exist_ok=True)
np.savez_compressed("data/processed/dataset.npz", **out)

counts = {}
for s_name, s in SPLITS.items():
    for src_name, src in [("fer", 0), ("raf", 1)]:
        m = (out["split"] == s) & (out["source"] == src)
        counts[f"{s_name}_{src_name}"] = {c: int((out["label"][m] == k).sum()) for k, c in enumerate(CLASSES)}
train = out["images"][out["split"] == 0].astype(np.float64) / 255
log["kept"] = len(sel)
log["counts"] = counts
log["train_mean"], log["train_std"] = float(train.mean()), float(train.std())
Path("results/data").mkdir(parents=True, exist_ok=True)
Path("results/data/summary.json").write_text(json.dumps(log, indent=2))
print(json.dumps({k: v for k, v in log.items() if k != "counts"}, indent=2))
for k, v in counts.items():
    print(f"{k:10s}", sum(v.values()), v)
