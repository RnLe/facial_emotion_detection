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

## 2026-09-30: models

Seven architectures plus a pretrained reference, each a step on from the one before
(`fer/models/`, hypotheses in `docs/hypotheses.md`, written before training):
simple CNN, VGG-style, ResNet-18, DenseNet-BC-100, ViT-Lite-7/4, ConvNeXt-Atto,
CCT-7/3x2, and ResNet-18 with ImageNet weights. All take a 48x48 grayscale face.

| Model | Parameters | GMACs per image |
|---|---|---|
| Simple CNN | 1.27 M | 0.02 |
| VGG-style | 7.05 M | 0.47 |
| ResNet-18 | 11.2 M | 1.25 |
| DenseNet-BC | 0.77 M | 0.16 (stride-2 stem) |
| ViT | 3.73 M | 0.53 |
| ConvNeXt | 3.38 M | 0.10 |
| CCT | 3.88 M | 0.62 |
| ResNet-18, ImageNet | 11.2 M | 0.33 (96x96 input) |

The architectures stay fixed; tuning changes only training settings. The question is how
much a given architecture depends on its settings, not which size wins.

## 2026-09-30: training setup

- The whole dataset lives on the GPU (100 MB as uint8). Each epoch, the shuffled training set
  is augmented in one call; batches are slices of it. No data loader, no CPU work per step.
- bfloat16 autocast, channels-last memory, fused AdamW, cuDNN autotuning.
- One optimiser family for all (AdamW, warmup then cosine decay), so that differences come
  from the architectures and not from the optimiser.
- Label smoothing 0.1 and square-root class weights by default (disgust and fear are about
  2% of the training images each).
- Model selection by validation macro-F1 after every epoch. Accuracy would reward ignoring
  the rare classes.

DenseNet at full 48x48 resolution took 57 s per epoch, six times ResNet-18, although it
needs half the multiply-adds. Profiling showed why: most of its time goes into copying the
growing stack of features and re-normalising it in every layer; the convolutions take 7%.
Memory-efficient checkpointing and other memory layouts did not help. DenseNet now starts
with one stride-2 convolution (blocks at 24, 12, 6 px): 12.5 s per epoch. Changed before any
real training run.

## 2026-09-30: tuning design

First plan: 40 trials of 60 epochs with successive halving (stop weak trials after 6 and
18 epochs). Changed before the tuning started: with pruning, most trials end after 6
epochs, and spread or importance computed from 6-epoch runs says little about 60-epoch
training. Now: 30 trials per model, each 25 epochs to the end, no pruning. The first 10
trials are random (an unbiased sample of the search space), the rest TPE. The best
settings are retrained for 60 epochs with three seeds. Cost: about 9 hours on the GPU.

Two processes on one GPU were tried for the baselines and dropped: under WSL2 the driver
switches between them, and together they got through about 20% less work than one
process running the jobs in turn.

## 2026-09-30: two baseline seeds

To save time, the baselines use two seeds per model instead of three (decided while seed 1
was running; seed 2 was never started). Final runs keep three seeds.

## 2026-09-30: tuning order

Tuning starts with the strongest models, ordered by mean baseline validation accuracy
(not test): ResNet-18, VGG, DenseNet, simple CNN, CCT, ConvNeXt, ViT. The order changes
nothing about the budget each model gets.

## 2026-09-30: baseline results

Two seeds per model, 60 epochs, defaults (report section 5, `results/baselines.json`).
ResNet-18, VGG and DenseNet lead at 84 to 85% test accuracy and cannot be told apart
(the test set alone has about ±0.9 points of sampling error). The attention models and
ConvNeXt trail by 6 to 13 points and underfit: their training accuracy at the end is only
74 to 81%. The ImageNet-pretrained ResNet-18 gains nothing over training from scratch.
Disgust and fear are the weakest classes for every model.

## 2026-09-30: a crash during tuning

At 22:08 the tuning process died with a segmentation fault in DenseNet's second trial: the
first trial with a different batch size, which makes the compiler rebuild a model whose
layers are recomputed in the backward pass (checkpointing). ResNet-18 and VGG had finished
their 30 trials each. DenseNet now runs without compilation (it only gained 14% from it),
each model's tuning runs in its own process (`scripts/run_all.sh`), a trial cut off by a
crash counts as failed, and the final runs skip models that are not tuned yet.
