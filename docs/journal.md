# Journal

Short notes on every step and decision, in order. The report is written from these.

## 2026-09-30: why start over

A 2024 course project trained a DenseNet on FER2013 and reported 57% against 34% for a
plain CNN. A review of that code showed that neither number measures test performance:

- The DenseNet was trained on the test folder and scored on it. `dataloader.load()`
  returns `(test, train)`, the notebook unpacked it as `(train, test)`.
- The hyperparameter search therefore minimised a training loss. That explains why it
  kept picking the weakest regularisation.
- The CNN was scored on images that were not scaled to [0, 1] like its training data.
  During training its held-out accuracy was about 61%.
- The two models used different versions of the dataset.

So the study starts again: one cleaned dataset, a clean split, the same pipeline for
every model, and the test set touched once at the end.

## 2026-09-30: data sources

- **FER2013** (Kaggle, original challenge CSV): 35,887 grayscale faces, 48x48, split into
  Training (28,709), PublicTest (3,589) and PrivateTest (3,589).
- **FER+** (Microsoft): the same images relabelled by 10 annotators each, with the options
  "unknown" and "not a face". Rows line up with FER2013; 173 images were removed by FER+.
- **RAF-DB** (basic set, Kaggle mirror): 15,339 aligned 100x100 colour faces, 12,271 train
  and 3,068 test, labels from about 40 annotators each. More "disgust" and "fear" than
  FER2013 has.

`scripts/01_download.py` fetches all three.

## 2026-09-30: looking at the raw data

`scripts/02_inspect.py` writes sample sheets (kept local, `results/inspect/`).

- Most FER2013 images are real faces. Stock-photo watermarks are common; they stay, a
  watermark does not change the expression.
- FER+ "not a face" and the removed images are the same 173 to 177 images: error pages with
  text, warning icons, black frames, and a few faces lying sideways.
- 12 images are completely black, one is flat grey. A broken-download icon (a crossed-out
  camera) appears several times.
- FER+ agreement varies: 8,120 images are unanimous, 3,875 have 4 or fewer votes for the
  top class. Low-agreement faces are genuinely ambiguous (grimaces, yawns, hands on faces).
- Near duplicates: a perceptual hash (64 bits, from a DCT) finds 2,632 exact pairs.
  Sheets at increasing distance show real copies up to 6 bits and mostly different faces
  from 7 bits on. At 6 bits, chains of false matches merge into one group of 189 images;
  at 4 or 5 bits the largest group has 24. Threshold: 5 bits. Many copies cross FER2013's
  own splits (for example #26487 in Training and #33042 in PrivateTest).
- Framing differs: RAF-DB faces are aligned and tightly cropped, FER2013 faces are looser
  and sit a little higher. Matching the two average faces (edge maps, correlation) over
  crop and shift gives the best match when FER2013 is cropped to 90% with its centre 2 px
  lower (correlation 0.24 before, 0.60 after; `results/data/framing.png`).

## 2026-09-30: cleaning and unifying

`scripts/03_build_dataset.py`, rules in its docstring.

- FER2013 labels come from the FER+ majority. An image stays only if at least 5 of 10
  annotators agree on one of the seven basic emotions: 4,271 dropped. FER+ changes the
  original FER2013 label of 10,266 kept images, about a third. The original labels are noisy.
- Contempt (FER+ only) is not one of the seven classes and is dropped with the rest.
- Duplicates among the survivors (5 bits, mirrored copies included): 1,852 groups with 4,424
  images, 731 of them across splits and 87 with conflicting labels. Conflicting groups go
  entirely; otherwise one copy stays, the test copy if there is one. 2,659 images dropped.
- All images: 48x48 grayscale (RAF-DB resized with area averaging).
- Splits: FER2013 Training / PublicTest / PrivateTest as train / val / test. RAF-DB gives
  10% of its train set (stratified) to validation and keeps its test set.

Result: 44,296 images. Train 33,827 (FER 22,923, RAF 10,904), val 4,316, test 6,153
(FER 3,098, RAF 3,055). Disgust is the rarest class (718 in train, 637 of them from
RAF-DB), then fear (628). Train pixel mean 0.502, std 0.240.

`tests/test_data.py` checks that no training image has a near copy in val or test.

## 2026-09-30: augmentation

`fer/augment.py`, on the GPU, a batch at a time: mirror, small rotation / zoom / shift,
brightness and contrast, random erasing. One `strength` knob scales everything, so the
tuning can choose it per model. Augmenting the whole training set takes about 60 ms on the
GPU, so it costs nothing per epoch. Augmentation is online (new variants every epoch)
rather than a fixed enlarged copy of the data; mirroring online has the same effect as
storing mirrored copies, without the memory.
