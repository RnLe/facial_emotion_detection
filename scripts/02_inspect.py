"""Look at the raw data before deciding how to clean it.

Writes sample sheets to results/inspect/ and prints the numbers that go into
the journal: vote agreement in FER+, non-faces, near-constant images, near
duplicates, and the average face of each source.
"""
import numpy as np

from fer.data import CLASSES, FERPLUS_EMOTIONS, load_fer2013, load_rafdb, near_pairs, phash, sheet

OUT = "results/inspect"
rng = np.random.default_rng(0)
fer = load_fer2013()
raf = load_rafdb(size=48)
votes = fer["votes"]
names = FERPLUS_EMOTIONS + ["unknown", "NF"]
top = votes.argmax(1)
agree = votes.max(1)


def fer_sheet(idx, path, title=lambda i: f"{names[top[i]]} {agree[i]}/10"):
    sheet(fer["images"][idx], [title(i) for i in idx], f"{OUT}/{path}")


print("FER2013:", len(fer["images"]), "| RAF-DB:", len(raf["images"]))
print("FER+ removed (no name):", int(fer["ferplus_removed"].sum()))
print("FER+ top vote:", {names[k]: int((top == k).sum()) for k in range(len(names))})
print("FER+ votes for the top class:", {int(k): int((agree == k).sum()) for k in range(1, 11)})
print("any NF vote:", int((votes[:, -1] > 0).sum()), "| NF top:", int((top == 9).sum()))
std = fer["images"].reshape(len(fer["images"]), -1).std(1)
print("FER near-constant (std < 3):", int((std < 3).sum()))

fer_sheet(rng.choice(len(votes), 60, replace=False), "fer_random.png")
fer_sheet(np.where(votes[:, -1] >= 3)[0][:60], "fer_notface.png", lambda i: f"NF {votes[i, -1]}/10")
fer_sheet(np.where(fer["ferplus_removed"])[0][:60], "fer_removed.png", lambda i: f"#{i}")
fer_sheet(np.where(top == 8)[0][:60], "fer_unknown.png")
fer_sheet(rng.choice(np.where((agree <= 4) & (top < 8))[0], 60, replace=False), "fer_low_agreement.png")
fer_sheet(np.argsort(std)[:40], "fer_flattest.png", lambda i: f"std {std[i]:.1f}")
pick = rng.choice(len(raf["images"]), 60, replace=False)
sheet(raf["images"][pick], [CLASSES[raf["label"][i]] for i in pick], f"{OUT}/raf_random.png")

# Average faces: where eyes and mouth sit on average in each source.
mean_fer = fer["images"][~fer["ferplus_removed"]].mean(0)
sheet(np.stack([mean_fer, raf["images"].mean(0)]), ["FER2013 mean", "RAF-DB mean"], f"{OUT}/mean_faces.png", cols=2, scale=2.2)

# Near duplicates across both sources; a mirrored copy counts as a copy.
images = np.concatenate([fer["images"], raf["images"]])
bits, bits_flip = phash(images), phash(images[:, :, ::-1])
direct, mirrored = near_pairs(bits, max_dist=12), near_pairs(bits, bits_flip, max_dist=12)
print("near pairs <= 12 bits: direct", len(direct), "| mirrored", len(mirrored))
for lo, hi in [(0, 0), (1, 3), (4, 6), (7, 9), (10, 12)]:
    sel = direct[(direct[:, 2] >= lo) & (direct[:, 2] <= hi)]
    print(f"  distance {lo}-{hi}: {len(sel)} pairs")
    if len(sel):
        s = sel[rng.choice(len(sel), min(15, len(sel)), replace=False)]
        imgs = np.stack([images[k] for p in s for k in p[:2]])
        sheet(imgs, [f"{p[0]} d{p[2]}" if n == 0 else f"{p[1]}" for p in s for n in range(2)], f"{OUT}/dup_d{lo}-{hi}.png", cols=10)
np.savez(f"{OUT}/hashes.npz", bits=bits, bits_flip=bits_flip, direct=direct, mirrored=mirrored)
