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
  lower (correlation 0.24 before, 0.60 after; `results/inspect/framing.png`).

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
training. Now: 30 trials per model, each 25 epochs to the end, no pruning. Trial 0 is the
defaults, trials 1 to 9 are random (an unbiased sample of the search space), the rest TPE. The best
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

## 2026-10-01: stage 2, the network's shape

Stage 1 keeps every architecture fixed, which answers the main question (how much a given
network depends on its training settings). It leaves open how much the shape itself matters.
Stage 2 adds it, for the three strongest baselines (ResNet-18, VGG, DenseNet), as a separate
set of studies so that stage 1 stays comparable across all seven models.

- Three shape settings, the same for each family: width (a multiplier on the channels;
  for DenseNet on the growth rate), depth (blocks per stage, convs per stage, layers per
  block) and the number of stages (how far the image is downsampled). Ranges in
  `scripts/06_tune.py`; from 0.08 to 39 M parameters for ResNet, 0.25 to 23 M for VGG,
  0.03 to 4.4 M for DenseNet. ResNet and VGG are large already, so their width range goes
  mostly down; DenseNet's goes both ways.
- Starts where stage 1 ended: batch size and class weights fixed at the stage 1 best
  (128 and square-root weights for all three), the other training settings searched over
  the same ranges as before. Trial 0 is the stage 1 best with the default shape.
- 40 trials of 25 epochs per model: the start, 11 random, then TPE. Nine settings instead of eight,
  and shape settings interact (with each other, and with the learning rate).
- Final runs: 60 epochs, three seeds, for the best trial and for the smallest network
  within one point of it (both picked on validation), to see if a small network holds up.

Estimated cost: about 10 hours, run after the stage 1 final runs (`scripts/run_shape.sh`).

## 2026-10-01: stage 1 tuning results

All seven studies finished at 05:43 (report section 6, `report/tables/tuning.csv`).

- Two groups: the CNNs with batch norm gain 2 to 3 points (validation macro-F1, 25-epoch
  trials) and 30 to 43% of their trials come within a point of the best. The simple CNN,
  ViT, ConvNeXt and CCT gain 4 to 9 points, with 7 to 10% of trials that close.
- All studies share one sampler seed, so the 9 random trials are the same settings on
  every model: a paired comparison. Nine times the default learning rate, no warmup and
  inverse class weights collapse VGG, ViT and ConvNeXt; ResNet-18 and DenseNet keep training.
- Warmup is the most important setting for four models, the learning rate for ViT and CCT,
  class weighting for ResNet-18. Augmentation, weight decay and batch size matter little.
- With 60 epochs, the defaults beat the best 25-epoch trial for ViT, CCT and DenseNet.
  The trials measure sensitivity under a short budget; the final runs (60 epochs) measure
  the gain from tuning.
- Analysis fix: Optuna's 10 startup trials include the enqueued default, so the random
  phase is trials 1 to 9, not 1 to 10.

## 2026-10-01: stage 1 final runs

Best settings per model, 60 epochs, three seeds, test set (report section 6.1). Tuning
gains 6.6 points of accuracy for ViT, 3.0 to 3.6 for CCT, ConvNeXt and the simple CNN, and
0.1 to 0.4 for VGG, ResNet-18 and DenseNet (within the test set's noise). The gap between
the best and the worst model shrinks from 13 to 7 points; ResNet-18, VGG and DenseNet stay
on top, level with each other. The baselines ran before test predictions were stored, so
default-against-tuned is a difference of means, not a paired test.

## 2026-10-01: stage 2, ResNet-18 done

The best shape (width 1.1, 13.7 M parameters) beats the start (stage 1 best, default
shape) by 0.4 points of validation macro-F1. Re-running the start's exact settings gave
78.3 against 78.7 in stage 1, so 0.4 points is also the run-to-run noise. Width explains
most of the spread across trials (57%), because the range reaches down to tiny networks
that score clearly lower; no network below 2.8 M parameters came within 1.5 points.

Change to the final runs (before any stage 2 final run): the "smallest within one point"
pick is retrained only if it is smaller than the default network. For ResNet-18 that
pick is the default shape itself, which the stage 1 final runs already cover.

## 2026-10-01: stage 2, VGG done

Best shape: width 1.21 and three convolutions per stage instead of two (14.8 M
parameters), 1.4 points of validation macro-F1 above the start. Seven of the eight best
trials use three convolutions per stage. Shape settings explain 44% of the spread, the
training settings 56%. The smallest network within a point of the best (8.6 M) is larger
than the default (7.05 M), so it is not retrained.

## 2026-10-01: DenseNet without recomputation

DenseNet ran at about 2% of the GPU's tensor-core peak (ResNet-18: about 40%), at 120 W of
285 W. A benchmark (stage 2 paused for five minutes) showed that recomputing its layers in
the backward pass, needed when it ran at full resolution, took 40 to 50% of its time and
saved almost no memory with the stride-2 stem (under 2.2 GB without it). Dropped from
trial 25 of the DenseNet shape study on: 1.6 to 1.9 times faster per step, identical
outputs and gradients. Compiling on top saves 0 to 20% per step but costs about 90 s per
run, so DenseNet stays uncompiled. Trial 24 was cut off by the restart and counts as failed.
Running the stage 2 final runs of ResNet-18 in parallel with the DenseNet trials was tried
and dropped: alone each ran at 12.3 s per epoch, together at 25.3 and 35.0 s, so the pair
got through 84% of the work of one process (as for the baselines: WSL2 switches between
the processes instead of running them side by side).

## 2026-10-01: stage 2, DenseNet done

Best shape: growth rate 16 (width 1.31) and 17 layers per block, 1.49 M parameters, 0.3
points above the start: within noise. The learning rate is the most important setting
(33%); the shape settings together explain 26%. The smallest network within a point of the
best has 0.52 M parameters (two thirds of the default) and is retrained in the final runs.
Across all three models: every best trial is larger than the default and keeps its number
of stages, and smaller networks lose steadily (at most a third of the parameters: 1.9 to
4.3 points below the best).

## 2026-10-01: stage 2 final runs

Best shapes, 60 epochs, three seeds, against the stage 1 final runs on the same test
images: +0.2 points of accuracy for each of ResNet-18, VGG and DenseNet, every interval
including zero. VGG's 1.4 points on validation did not carry over. DenseNet's smallest
near-best network (0.52 M) loses 1.2 points of accuracy and 2.5 of macro-F1, a clear
difference. Conclusion: for these three, the published sizes were already right for this
data; stage 2 (about 11 GPU hours) changed no conclusion of stage 1.

## 2026-10-01: long runs, design

Question: do the three strongest models converge, overfit, or generalise late (grokking)
when trained far longer than 60 epochs? `scripts/09_long.py`, measures in `fer/measures.py`.

- Three runs per model: *tuned* (stage 1 best settings, default shape), *larger* (stage 2
  best shape and settings), *memorize* (stage 1 settings without augmentation, label
  smoothing and dropout, weight decay set to lr x weight decay = 1e-3 as in the original
  grokking runs). The memorize run is the setting in which grokking was found: the
  training set can be learned by heart, and weight decay keeps pulling afterwards.
- Constant learning rate after the warmup instead of cosine decay, so a run is not tied to
  a fixed length and can be extended. To compare with the 60-epoch runs, a 10-epoch
  cooldown (learning rate falling linearly to zero) branches off at the end of each block
  of epochs (warmup-stable-decay) and is scored on validation and test.
- Logged every epoch: accuracy and loss on 4,316 fixed training faces without
  augmentation (memorisation), validation scores, the train-validation gap, the spectral
  entropy of the penultimate features (Truong et al. 2026), the absolute weight entropy
  (Golechha 2024), the spectral entropy of the weight matrices (closeness to low rank),
  weight norm, neural collapse (Papyan et al. 2020), and prediction entropy.
- No model selection: the curves show every epoch, the cooldown scores the last one.
- Checkpoints every 5 epochs; `--epochs N` resumes and extends. First block: 100 epochs.

## 2026-10-01: long runs, 100 epochs

All nine runs and their cooldowns done (single seed each). After the cooldown every run
lands within about a point of its 60-epoch counterpart on the test set: the tuned runs
0.2 to 0.6 points below, the larger runs between 0.1 below and 0.6 above. Training longer
does not help by itself. The memorize runs end 2 to 4.7 points below the tuned runs and
do not grok within 100 epochs. At the constant learning rate they fit only 86 to 94% of
the training faces; the step noise keeps them from memorising, and only the cooldown
takes them to 100%. DenseNet was still improving at epoch 100 in all three runs and fits
only 84 to 87% of the training faces at the constant learning rate. In every run the
weights drift towards low rank, most in the memorize runs.

## 2026-10-01: grok run, design

To test grokking where it can actually happen, a run needs a fully memorised training
set first. The VGG memorize run after its cooldown has that (100% of the training faces,
81.5% validation). The grok run continues from there for 1,000 epochs with the same
settings at a tenth of the learning rate (4.3e-4; the cooldown fitted every training face
from 8.7e-4 down) and unchanged weight decay. VGG because it trains fastest (3.1 s per
epoch without augmentation, about an hour for 1,000 epochs). If validation accuracy does
rise, a control run without weight decay is next, to tell grokking from plain fine-tuning.

## 2026-10-01: grok run result

1,000 epochs (110 to 1110) at a tenth of the learning rate, then the cooldown. No
grokking: validation accuracy between spikes peaked at 82.2% in the second cycle and
drifted down to about 80%. After the cooldown the run scores 79.8% on the test set
(macro-F1 70.0), against 81.5% (72.0) where it started. Eleven loss spikes, two of them
near-total collapses (training accuracy down to 35% and 32.5%), each refitting the
training set from a damaged state. Likely reason: with batch norm the weight norm does not
change the function, so weight decay cannot favour a simpler solution; it only raises the
effective step size until training breaks (Lobacheva et al. 2021, the periodic behaviour of
batch norm with weight decay). H14 holds. A fair grokking test needs a network without
batch norm, a small training set and large initial weights (Omnigrok).

## 2026-10-02: grokking on the simple CNN, design

The Omnigrok recipe on faces (`scripts/10_grok.py`): the simple CNN, which has no batch
norm, on 1,001 training faces drawn in class proportion, MSE loss on one-hot targets,
AdamW at lr 1e-3, batch 200, no augmentation, 1e5 steps (20,000 passes over the faces),
full float32. Grid: weights started at 1x or 3x their usual size, times weight decay 0.1,
0.01 and 0. A 3x start scales the outputs by 3^5 = 243; an 8x start, as on MNIST, killed
every ReLU in a test run and left a constant majority-class output. 9 minutes per run.

## 2026-10-02: grokking on the simple CNN, result

Validation accuracy (%). "Half-way" is the step where validation accuracy first covers
half the distance from its lowest point after memorising to its peak.

| Start, weight decay | Memorised at step | Val then | Val at 1e5 steps | Half-way at step |
|---|---|---|---|---|
| 3x, 0.1 | 1,500 | 43.2 | 54.6 | 18,500 |
| 3x, 0.01 | 1,500 | 38.9 | 49.6 | 16,500 |
| 3x, 0 | 1,500 | 39.6 | 45.1 (peak 48.9) | 16,250 |
| 1x, 0.1 | 500 | 52.2 | 57.3 | 9,500 |
| 1x, 0.01 | 500 | 51.2 | 54.6 (peak 55.8) | 8,750 |
| 1x, 0 | 500 | 51.0 | 54.5 | 8,250 |

- Every run rises long after it has memorised the training faces: half-way comes 11 to 19
  times later. The rise is gradual, spread over about a decade of steps, not a jump.
- The 3x start begins 9 to 12 points lower and takes twice as long. With weight decay 0.1
  it closes most of the gap to the 1x start (54.6 against 57.3).
- Weight decay does not set the timing: half-way at 16,250 to 18,500 steps for all three
  3x runs, where Omnigrok expects the time to scale with 1 / weight decay. It sets how far
  the rise goes and whether it lasts. Without it the 3x run falls back from 48.9 to 45.1
  while its weights grow to 8.6x, and its macro-F1 ends below where it started (21.4
  against 22.7).
- The one measure that moves with the rise in every 3x run is NC1 on the training faces:
  it falls about tenfold during the rise (weight decay 0: 0.65 at step 10,000, 0.08 at
  30,000). At the 1x start NC1 is already 0.03 when the faces are memorised. Reading: the
  3x start first fits the training faces without sorting its features by class, a lazy,
  kernel-like fit, and generalises once the features change. This is the lazy-to-rich
  account of grokking (Kumar et al. 2024).
- Weight norm: with weight decay 0.1 the 3x run's norm falls to 1.56x by step 19,250,
  where it is half-way, and then grows again to 2.95x; the 1x run ends at 2.52x. AdamW's
  steps have a fixed size, so the norm likely settles where weight decay balances them,
  whatever the start.
- Representation entropy (Truong et al. 2026) peaks at or after half-way in the 3x runs and
  falls afterwards, so it gives no warning here. Absolute weight entropy falls with weight
  decay and rises without it while validation rises in both: it follows the weight norm.
- Macro-F1 stays at 32 to 34 in the 1x runs, so their late gain is in the common classes.
  The 3x run with weight decay 0.1 goes from 24 to 31.

H15 holds in part. Right: with weight decay 0.1 the 3x start rises late and gradually
towards the 1x level. Wrong: without weight decay it rises too (and falls back), and 0.01
is not slower, only lower.

## 2026-10-02: grokking on VGG without batch norm, design

One long run (the user's choice) to see whether VGG behaves like the simple CNN once batch
norm is gone. `VGG(norm=False)`: the convs get biases instead. With PyTorch's default
initialisation the signal fades through the 10 weight layers until every face gets the
same output, so the start is set by the spread of outputs across the 1,000 training faces
at step 0: alpha 2.8 gives 3.5, the CNN's alpha 3 gives 3.2. No layer starts dead. Weight
decay 0.1, lr 1e-3, batch 200, MSE, 2e5 steps, bf16 with torch.compile (20 ms per step;
with MSE there is no softmax whose precision could matter). It continues the 2,000-step
test from its checkpoint.

## 2026-10-02: grokking on VGG without batch norm, result

Stopped at step 105,000 of 200,000: validation peaked at step 74,250 and every measure had
settled or was repeating. The checkpoint is kept, so the run can be continued.

- Memorised at step 750 at 45.3% validation. Validation rose gradually to a peak of 54.5%,
  half-way at step 11,250 (15 times later). Macro-F1 went from 25.9 to at most 30.1.
- After step 75,000 it slips: 54.3% at 75,000, 52.8% at 100,000, while val loss and NC1
  rise again (NC1 0.0008 at 50,000, 0.0054 at 100,000).
- It ends below the simple CNN on the same faces (peak 57.5% for the CNN's 1x start, 55.2%
  for its 3x start, both with weight decay 0.1). Without batch norm the extra depth does
  not help on 1,000 faces.
- Every 5,000 steps or so the weight norm slowly falls (to about 1.0x to 1.2x) and then
  jumps back by 40% within 250 steps, with validation briefly 1 to 2 points lower. Training
  accuracy stays at 100%. The CNN runs show the same pattern at a tenth of the size. There
  is no batch norm here, so this is most likely the slingshot effect of Adam (Thilak et al.
  2022): once the training set is fitted the gradients and Adam's running estimate of
  their size both become tiny, weight decay erodes the fit until a gradient returns, and
  divided by that tiny estimate the first steps are large. Validation is no higher after a
  jump than before it.
- What keeps changing slowly: relative weight entropy 0.977 at step 10,000 to 0.858 at
  100,000, weight spectral entropy 0.988 to 0.743 (the layers drift towards low rank), both
  slowing down tenfold over the run.

H16 is wrong in size: the late rise is 9 points, not 5, although NC1 was already low before
it started. NC1 falling late is not needed for a late rise; it only came with the larger
rise of the CNN's 3x start.

## 2026-10-02: compression, design

The three larger long-run models after their cooldown, each compression method on its own
and without fine-tuning (`fer/compress.py`, `scripts/11_compress.py`). One knob for the
three factorisations, tau: the share each layer keeps of its output variance (output-based
low rank) or of its squared singular values (Tucker-2, spatial split), swept from 0.999 to
0.8. A layer is only replaced when that saves parameters. The output statistics and the
int8 calibration come from 4,096 training faces; scores are on validation. Timing: median
over repeated runs, one face and 256 faces, GPU in fp32 and CPU with 8 threads, plus one
face on one CPU thread. Other programs share the machine, so each baseline is timed again
at the end of its sweep.

Before the sweep: PyTorch's int8 breaks DenseNet (85.4% to 60.7% on validation; 62.7% with
4,096 calibration faces) but not VGG (86.0% to 86.2%). Keeping only the concatenations in
float recovers most of it (78.1%). PyTorch gives every input of a concatenation one shared
scale, and DenseNet concatenates up to 18 feature groups of very different size, so the
small ones lose their resolution. Both variants are kept.

## 2026-10-02: compression, results without fine-tuning

Baselines (fp32, CPU with 8 threads; the timings drift by up to 12% between the start and
the end of a sweep, since other programs share the machine):

| Model | Size | Val acc | GPU, 1 face | GPU, 256 | CPU, 1 face | CPU, 256 | CPU 1 thread, 1 face |
|---|---|---|---|---|---|---|---|
| VGG | 56.5 MB | 86.0 | 0.77 ms | 34 ms | 6.0 ms | 909 ms | 20.3 ms |
| ResNet-18 | 52.3 MB | 85.6 | 1.20 ms | 53 ms | 6.9 ms | 1,377 ms | 26.0 ms |
| DenseNet | 6.0 MB | 85.4 | 6.5 ms | 58 ms | 8.7 ms | 1,437 ms | 9.8 ms |

The strongest setting of each method that loses at most 0.5 points of validation accuracy
(fewer parameters / fewer MACs / CPU speedup for one face / GPU speedup for one face):

| Method | VGG | ResNet-18 | DenseNet |
|---|---|---|---|
| Output-based low rank | 2.4x / 1.8x / 1.28x / 0.57x | 4.2x / 3.5x / 1.38x / 0.52x | 1.9x / 2.0x / 0.84x / 0.52x |
| Tucker-2 | 1.2x / 1.2x / 1.07x / 0.58x | 2.3x / 2.5x / 1.26x / 0.54x | 1.1x / 1.2x / 0.85x / 0.65x |
| Spatial split | 2.0x / 3.1x / 1.54x / 0.61x | 3.1x / 3.2x / 1.22x / 0.60x | 1.2x / 1.2x / 0.96x / 0.78x |
| int8 (CPU only) | 4.0x smaller, 3.5x faster (256 faces: 2.6x), +0.2 | 4.0x smaller, 2.0x faster (2.7x), +0.3 | 3.1x smaller, but -24.7 (-7.2 with float concatenations) |

- Every factorisation makes a single face slower on the GPU: the extra layers cost more
  launches than the saved arithmetic. On the CPU the speedups stay well below the MAC
  savings, and DenseNet gets slower. int8 is faster than any factorisation.
- VGG's output-based low rank holds up to 99% of the output variance and then collapses
  (98%: 80.7, 95%: 51.8). The cause is neural collapse: at 95% the last conv keeps 7
  directions and the first dense layer 4, below the 6 that seven classes need. Leaving
  these two layers and the classifier whole brings 95% back to 85.2, but they hold most
  of VGG's parameters (9.6 M instead of 2.8 M). One threshold for all layers is the wrong
  way to set ranks. ResNet loses only 0.2 points at 95% (4.2x), likely because its skip
  connections carry the signal past the approximated convs.
- The spatial split holds up far better than the channel factorisations from the
  weights: the weights are close to full rank across channels, but much less so across
  the two directions of the kernel. On VGG it removes the most MACs of all methods (3.1x).
- Everything below 0.5 points still needs the repair fine-tuning that is next.

Verdicts. H17 holds at 99% (VGG 2.4x, ResNet 1.8x, DenseNet 1.2x, no loss) and is wrong at
95% in both directions: VGG collapses, ResNet loses nothing. H18 holds: at equal size
Tucker-2 is always worse than output-based low rank (VGG 5.9 M: 82.0 against 86.0 at
6.1 M). H19 is wrong: the spatial split saves 2 to 3x on VGG and ResNet without loss.
H20 holds for VGG and ResNet (with a larger single-face speedup than predicted for VGG)
and fails for DenseNet. H21 holds.

## 2026-10-02: repair stage, design and pilots

Per-layer ranks instead of one threshold (`fer.compress.sensitivity` and `allocate`,
`scripts/12_repair.py`). For each layer and candidate rank (fractions 1/64 to 3/4 of the
layer's width), only that layer is replaced and the KL divergence from the original
predictions is measured on 2,048 training faces that did not fit the projections. The
ranks with the smallest summed KL that meet a parameter budget follow from a Lagrange
weighting (each layer minimises KL + lambda x weights; lambda by bisection). Then 10 epochs
of repair on the full training set with the model's own settings, the learning rate
falling linearly to zero.

Pilots on the hardest case, VGG with low rank at 8x fewer parameters (1.82 M): before the
repair 83.9% validation accuracy, against 29.6% for one threshold at about the same size
(tau 0.9, 1.72 M). The allocation keeps 38 directions in the last conv and 19 in the first
dense layer, where the threshold kept 7 and 4. After 10 epochs: 85.4% with the learning
rate starting at 0.1x the original, 85.3% at 0.3x, 85.4% at 0.1x with distillation from
the uncompressed model (macro-F1 76.9 against 77.0 without). The grid uses 0.1x without
distillation.

## 2026-10-02: repair stage, results

27 planned repairs, 19 possible: Tucker-2 and the spatial split only touch the 3x3 convs,
so they cannot reach 8x on VGG (its dense head holds 23% of the parameters) or even 2x on
DenseNet (its 1x1 layers hold two thirds). Validation accuracy before and after the
repair, test once at the end (uncompressed: VGG 86.0 / test 85.6, ResNet-18 85.6 / 85.2,
DenseNet 85.4 / 85.0). Speed on one CPU thread, one face, against the uncompressed model:

| Model | Method | Target | Before | After | Test | One thread |
|---|---|---|---|---|---|---|
| VGG | low rank | 2x / 4x / 8x | 86.2 / 85.8 / 83.9 | 85.9 / 85.5 / 84.8 | 85.1 / 84.2 / 84.3 | 1.1x / 1.8x / 2.6x |
| VGG | Tucker-2 | 2x / 4x | 86.3 / 56.9 | 85.4 / 83.1 | 85.1 / 82.3 | 1.3x / 6.3x |
| VGG | spatial | 2x / 4x | 86.0 / 30.1 | 85.8 / 77.0 | 85.0 / 75.4 | 1.2x / 5.8x |
| ResNet-18 | low rank | 2x / 4x / 8x | 85.6 / 86.1 / 83.2 | 86.2 / 85.7 / 84.4 | 85.5 / 85.3 / 83.8 | 1.3x / 2.2x / 3.0x |
| ResNet-18 | Tucker-2 | 2x / 4x / 8x | 85.4 / 85.1 / 84.5 | 86.1 / 85.5 / 84.7 | 85.6 / 85.4 / 84.9 | 1.2x / 2.3x / 3.6x |
| ResNet-18 | spatial | 2x / 4x / 8x | 85.6 / 85.6 / 84.5 | 85.9 / 85.6 / 84.7 | 85.9 / 85.3 / 84.7 | 1.2x / 1.8x / 3.2x |
| DenseNet | low rank | 2x / 4x / 8x | 85.0 / 48.3 / 7.9 | 85.5 / 82.4 / 77.2 | 84.9 / 82.4 / 75.0 | 0.9x / 1.0x / 1.0x |

- Per-layer ranks beat one threshold by far where the threshold failed: VGG low rank at
  about 4x, 85.8 before repair against 51.8; ResNet spatial at 8x, 84.5 against 76.5. Not
  for DenseNet (4x: 48.3 against 50.9): its 106 layers each feed every later layer, so the
  per-layer KL does not add up.
- The allocation can cut the multiply-adds far more than the parameters: VGG Tucker-2 and
  spatial at 4x keep 3.7 M parameters but only 55 M of 1,195 M MACs, cutting the
  high-resolution layers that are cheap in weights. Each cut looked harmless on its own;
  together they broke the network (30.1 and 56.9 before repair), and repair could not
  fully recover them.
- ResNet-18 compresses best: every method stays within a point at 8x on the test set, and
  Tucker-2 at 8x is 3.6x faster on one CPU thread at 0.3 points below the uncompressed
  model. Its skip connections carry the signal past the approximated convs.
- At 2x the repair sometimes costs accuracy (VGG Tucker-2: 86.3 before, 85.4 after): it
  starts at a tenth of the original learning rate, which moves a cooled-down model out of
  its minimum, and 10 epochs do not fully settle it again.
- On the GPU every compressed model is slower for one face (0.5 to 0.75x). DenseNet low
  rank is no faster on the CPU either.

H22 holds for ResNet-18 at every target and for low rank on VGG; it fails for VGG's
Tucker-2 and spatial split at 4x. At 8x it is reversed on ResNet-18: Tucker-2 and the
spatial split stay within a point (0.3 and 0.5 below on the test set), low rank does not
(1.4 below).

## 2026-10-03: int8 on top of the repaired models, and DenseNet with quantisation-aware training

int8 after training (CPU, `scripts/13_quant.py`) on every repaired model. For VGG and
ResNet-18 it costs at most 0.3 points of validation accuracy on top of the factorisation
and multiplies the size reduction by 4. The best combinations (size against the fp32
model; speed on one CPU thread for one face):

| Model | Version | Size | Val | Test | One thread |
|---|---|---|---|---|---|
| ResNet-18 | uncompressed, fp32 | 52.3 MB | 85.6 | 85.2 | 26.0 ms |
| ResNet-18 | Tucker-2 4x + int8 | 3.2 MB (16x) | 85.4 | 85.4 | 3.8x faster |
| ResNet-18 | spatial 8x + int8 | 1.8 MB (29x) | 84.8 | 84.5 | 6.8x faster |
| VGG | uncompressed, fp32 | 56.5 MB | 86.0 | 85.6 | 20.3 ms |
| VGG | low rank 4x + int8 | 3.5 MB (16x) | 85.5 | 84.3 | 3.4x faster |
| VGG | low rank 8x + int8 | 1.9 MB (30x) | 84.6 | 84.6 | 5.2x faster |

DenseNet does not take int8 well. With the concatenations in float, int8 on its repaired
low-rank models loses 7 to 57 points. Quantisation-aware training (5 epochs at 0.05x the
learning rate, observers and batch norm frozen for the last 2) brings the fully quantised
model from 60.7 to 79.7 on validation, still 5.7 points below float; with the
concatenations kept in float it reached 83.8 with simulated int8 during training but 78.4
after conversion, so the conversion around the float concatenations does not match what
the training simulated.

H24 holds for VGG and ResNet-18 (another 4x in size, at most 0.3 points) and fails for
DenseNet. H25 is wrong: 5.7 points below float, not 1.

## 2026-10-03: same-size networks from scratch

The fair baseline (`scripts/14_scratch.py`): for 4x and 8x fewer parameters, the width
whose parameter count comes closest, with the stage 2 depth and stages and the larger
run's settings, trained with the final runs' recipe (60 epochs, one seed). Against the
repaired model with the best validation accuracy at the same target (val / test):

| Model | Target | From scratch | Repaired |
|---|---|---|---|
| VGG | 4x | 85.1 / 83.2 | 85.5 / 84.2 (low rank) |
| VGG | 8x | 85.1 / 84.3 | 84.8 / 84.3 (low rank) |
| ResNet-18 | 4x | 85.4 / 84.5 | 85.7 / 85.3 (low rank) |
| ResNet-18 | 8x | 85.0 / 83.9 | 84.7 / 84.9 (Tucker-2) |
| DenseNet | 4x | 84.6 / 83.9 | 82.4 / 82.4 (low rank) |
| DenseNet | 8x | 84.3 / 83.4 | 77.2 / 75.0 (low rank) |

For VGG and ResNet-18 within a point either way; for DenseNet the network from scratch is
clearly better (its 8x network has 0.22 M parameters instead of 0.19 M, since the growth
rate must be a whole number). H23 mostly holds: compression saves a training run but finds
no small network that training from scratch misses. The planned runs of the study are
complete.

# Part two: beating the references

## 2026-10-04: references and the plan

The study tops out at 85% (ResNet-18), every model from scratch at 48 px grayscale. The
best published RAF-DB models reach 92 to 94%. Part two asks how much of that gap our
models can close, and with what. The order follows a brainstorm on finding a model's
bottleneck: references first, then diagnosis (where do the remaining errors come from),
conventional optimization, a closer look at the architectures, and only then new
mechanisms. Two tracks keep the claims clean: open (public pretrained weights, any input
size) and scratch (our data only, equal budgets).

References with released weights (the two best CLIP prompt methods, MPA-FER 93.74% and
LaCoVL-FER 93.61%, released none):

| Model | RAF-DB | Weights | Parameters |
|---|---|---|---|
| FMAE (Ning et al. 2024) | 93.45 | RAF-DB checkpoint, Hugging Face, CC BY-NC | 304 M (ViT-L/16) |
| POSTER++ (Mao et al. 2023) | 92.21 | RAF-DB checkpoint, Google Drive, MIT | 58 M (44 M trained, the landmark net is frozen) |
| CLIP ViT-B/16 (Radford et al. 2021) | none | OpenAI, through open_clip | 86 M image tower |

Protocol: selection on validation only, test scored once per phase. The RAF-DB references
were trained on RAF-DB's full train set, which holds our RAF validation faces, so their
validation scores on the RAF part are inflated and their fine-tuning runs a fixed number
of epochs. Many RAF-DB papers report the best epoch on the test set; our numbers select
on validation and are therefore conservative. Trained from scratch, the references would
take 8 to 10 GPU hours each; that study waits for overnight runs.

`data/processed/rafdb.npz` (`scripts/17_rafdb.py build`): all 15,339 RAF-DB faces at 48 px
grayscale, the official train (11,050 after our validation share) and test sets (3,068),
nothing removed, as in the papers. Our cleaned dataset kept 3,055 of the test faces.

## 2026-10-04: phase 1, the references and the targets

`scripts/16_references.py`, `scripts/17_rafdb.py`. Reproduction on the official RAF-DB test
set with the released weights, then the same 3,068 faces in grayscale and at our 48 px:

| Model | Paper | Native 100 px color | 100 px gray | 48 px gray |
|---|---|---|---|---|
| FMAE | 93.45 | 93.29 | 92.24 | 91.46 |
| POSTER++ | 92.21 | 92.08 | 91.20 | 89.15 |

Both reproduce within 0.2 points (H26 holds). Their outputs follow RAF-DB's label order
(98 and 100% on training faces with that order, 50% without). Color is worth about one
point to them, the step from 100 to 48 px another 1 to 2.

On our test set (accuracy; macro-F1 for the whole set):

| Model | Applied as released | FER2013 / RAF-DB part | Fine-tuned on ours | FER2013 / RAF-DB part | Macro-F1 |
|---|---|---|---|---|---|
| FMAE | 86.59 | 81.79 / 91.46 | **90.18** | 89.28 / 91.10 | 84.8 |
| POSTER++ | 78.60 | 68.21 / 89.13 | 88.67 | 88.32 / 89.03 | 82.2 |
| CLIP ViT-B/16 | 50.74 zero-shot, 83.18 linear probe | 85.51 / 80.82 (probe) | 87.79 | 88.48 / 87.10 | 81.3 |
| Our ResNet-18 (part one) | | | 85.11 | 86.69 / 83.50 | 77.7 |

Fine-tuning ran a fixed number of epochs (FMAE 4, POSTER++ 10, CLIP 5) with our
augmentation, square-root class weights and label smoothing 0.1, last weights kept.

- H27 is partly wrong. POSTER++ and CLIP fall below our 85% as released, but FMAE beats
  our best model by 1.5 points without ever seeing our data, entirely on the RAF-DB part
  (91.5 against 83.5); on the FER2013 part it loses 4.9 points. The FER2013 part is
  hit hardest for all three, as guessed.
- H28 holds for all three: FMAE 90.2, POSTER++ 88.7, CLIP 87.8. Fine-tuning lifts mainly
  the FER2013 part (FMAE +7.5, POSTER++ +20.1); the RAF-DB part stays where it was.
- Pretraining is worth 3 to 5 points here, but not any pretraining: part one's
  ImageNet ResNet-18 at 96 px reached 84.4, below training from scratch. The three
  references were pretrained on faces (FMAE, POSTER++'s IR-50) or on 400 million web
  images (CLIP), and all run at 224 px with far more compute per face (FMAE 147 faces/s
  against several thousand for ResNet-18 at 48 px).

RAF-DB alone. Our combined-data winners score 83.7 (VGG), 83.6 (ResNet-18) and 83.4
(DenseNet) on the full official test set. Trained on RAF-DB alone in our pipeline: VGG
83.9, ResNet-18 83.5 (3 seeds each), DenseNet 83.4 (seed 0; seeds 1 and 2 still to run).
H29 holds: 9.4 points below the reproduced FMAE. Adding the FER2013 faces neither helps
nor hurts the RAF-DB part.

The targets:

| | Our test set (A) | RAF-DB official test (B) |
|---|---|---|
| Open track | 90.18 (FMAE fine-tuned) | 93.29 (FMAE, native input); 91.46 at 48 px gray |
| Scratch track | 85.11 (ResNet-18, stage 1) | 83.9 (VGG, RAF-DB only) |

The scratch-track targets are our own models for now; the references trained from
scratch (study 3) wait for overnight runs.

## 2026-10-04: phase 2, where the errors come from (first half)

`scripts/18_diagnose.py`, `scripts/19_errors.py`, `scripts/20_probes.py`, on the three family
winners (VGG and DenseNet from the shape stage, ResNet-18 from stage 1).

No defect (H30 holds). With augmentation and every regulariser off, all three fit 256
training faces to 100% (DenseNet after 36 epochs, ResNet-18 after 69, VGG after 168).

Memorisation. Scored without augmentation, the trained winners get 96.5 (ResNet-18), 97.0
(DenseNet) and 98.8% (VGG) of their training faces right, against 85 to 86 on validation,
about the same gap on both sources. They fit what they see; the gap is generalisation.

Shared errors (test set, H34 holds):

| | VGG | ResNet-18 | DenseNet |
|---|---|---|---|
| One run (mean of 3 seeds) | 84.96 | 85.11 | 84.78 |
| Its 3 seeds averaged | 86.33 | 86.66 | 86.46 |

One run of each family averaged: 86.95; all nine runs: **87.76** (FER2013 part 88.7, RAF-DB
part 86.8). At least one family is right on 92.0% of the faces, all three on 76.8%. The
families agree on 87 to 89% of the faces, and their penultimate features are less alike
than two seeds of one family (linear CKA 0.70 to 0.83 across families, 0.86 to 0.91 within).
Averaging buys 2.7 points of the 15% error: a large part is variance between runs, not a
shared limit.

The annotators' view (FER2013 test part, 3,098 faces with 10 FER+ votes each):

| Votes for the label | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|
| Faces | 420 | 393 | 457 | 479 | 553 | 796 |
| ResNet-18 correct | 61.8 | 70.7 | 86.2 | 91.2 | 96.3 | 98.7 |

Two thirds of the FER2013 errors sit on faces with 5 or 6 votes. But measured against the
annotators the model is not at the ceiling. The best possible agreement with one random
annotator, E[max_c p_c], lies between 81.4 (one annotator against the majority of the
others, which the best classifier can only beat) and 83.0 (the plug-in estimate, too high
with ten votes); the bootstrap-corrected estimate is 81.9. ResNet-18 reaches 77.2, the
nine-run ensemble 78.4. A panel of nine annotators beats our model by 4 points at
predicting the tenth. The majorities of two random halves of the annotators agree on 87%
of the faces. The model's uncertainty follows the annotators' only weakly (Spearman 0.28
between the entropies). H33: the first half holds (98.7 and 96.3 at 10 and 9 votes, 61.8
at 5, but 70.7 at 6); the second half is wrong, the model is 4 to 5 points below the
ceiling, not within 3.

Inside the networks (linear probes on 8,000 training faces, scored on validation). The
expression becomes readable mostly between the second and third stage (ResNet-18: 52 after
the stem, 57, 67, 81, 86 after the last stage; VGG and DenseNet alike); the last stage
adds 5 points. The source is readable at 91 to 98% from the first stage on and stays in
the features of ResNet-18 and DenseNet up to the classifier (95 and 97%); VGG's dense head
removes some of it (78%). H37: the second half holds; the first half is wrong, the third
stage already carries most of it.

Occlusion (ResNet-18, recall change on validation when a region is replaced by the mean
face): happy needs the mouth (-30), surprise mouth and eyes (-27, -21), fear eyes and
brows (-28, -18), sad the eyes (-23) and hardly the mouth (-2), angry mouth, eyes and brows
(-20, -14, -13). Disgust, the weakest class (recall 55%), is read from mouth and eyes (-22,
-19) and hardly from the nose (-1), although the nose wrinkle is its defining action unit.

Information (H36 is wrong). ResNet-18 on RAF-DB alone, seed 0, validation / test: 48 px
gray 83.0 / 83.9; 100 px gray with a stride-2 stem (about the same compute) 83.3 / 84.8;
100 px color with a stride-2 stem 83.5 / 82.8; with a stride-1 stem (4x compute) 82.3 / 82.7
gray and 82.9 / 83.1 color. All within the spread between seeds. For a network trained from
scratch neither resolution nor color is the bottleneck; the references themselves lose
only 1.8 (FMAE) and 2.9 (POSTER++) points at 48 px gray. The gap to them is in what the
network has learned, not in what the input holds.

## 2026-10-04: phase 2, where the errors come from (second half)

Sources (ResNet-18 winner, 3 seeds, the same number of optimiser steps for every
training set, selected on the training source's validation share):

| Trained on | FER2013 test part | RAF-DB test part |
|---|---|---|
| FER2013 only | 86.25 | 66.63 |
| RAF-DB only | 67.42 | 84.32 |
| Both (stage 1) | 86.69 | 83.50 |

Across sources a model loses about 18 points. Pooling gains 0.4 on FER2013 and costs 0.8
on RAF-DB (where the RAF-only model also saw each face three times as often). H31: the
second half holds; the first half is wrong for the RAF-DB part. The two datasets are far
apart, which fits the source staying readable in the features.

Label errors (confident learning on 5-fold out-of-fold probabilities, 84.7% correct):
2,574 training faces flagged, 7.6% (FER2013 6.8%, RAF-DB 9.3%), mostly neutral and sad
(868 and 494 flags), on the neutral/sad, happy/neutral and fear/surprise boundaries. The 60
most confident flags of each source, looked at by eye: on FER2013 mostly faces that sit
between two expressions (a wide-eyed open mouth labelled surprise, read as fear), plus a
few cartoons, drawings, and a screaming face labelled happy; on RAF-DB more faces whose
label looks wrong (happy or neutral faces without a smile, read as disgust at 0.95 to 1.00
confidence) and many blurred or half-covered faces. Training without the flagged faces
(3 seeds): validation 85.48 against 85.98, macro-F1 78.1 against 78.9, test 84.95 against
85.11. Removing them costs a little; they are hard faces, not useless ones. H35: the share
is wrong (7.6, not 2 to 4%); the effect is at the edge of the guess (-0.5).

RAF-DB alone, DenseNet seeds 1 and 2: 80.1 and 83.5 (seed 1 ends 3 points below the others,
its validation score too; mean of 3 seeds 82.3).

## 2026-10-04: phase 4 run early, controlled changes to the input

`scripts/22_bottleneck.py perturb`, validation set, our three winners (seed 0) against FMAE
fine-tuned on our data. Accuracy:

| Input | VGG | ResNet-18 | DenseNet | FMAE |
|---|---|---|---|---|
| Intact | 85.9 | 86.0 | 85.2 | 91.5 |
| Mirrored | 85.4 | 85.6 | 84.6 | 92.0 |
| 2x2 tiles shuffled | 48.1 | 57.6 | 54.1 | 79.1 |
| 3x3 tiles shuffled | 32.1 | 38.5 | 34.0 | 56.5 |
| 4x4 tiles shuffled | 28.7 | 30.1 | 25.0 | 44.1 |
| 6x6 tiles shuffled | 20.3 | 27.7 | 20.7 | 24.1 |
| Blur, sigma 1 px | 80.1 | 80.5 | 79.2 | 87.6 |
| Blur, sigma 2 px | 53.0 | 54.5 | 53.1 | 61.0 |
| High-pass (minus sigma 2 blur) | 80.3 | 80.0 | 78.7 | 86.7 |

Two differences stand out. Mirroring changes our models' prediction on 8 to 9% of the
faces (disgust 17 to 22%, fear 14 to 19%), although every one of them trained with random
flips; FMAE's changes on 2.9% (disgust 3.3%). And FMAE still reads 79% of the faces with
their quadrants shuffled, ours 48 to 58%: it can tell the expression from a part on its
own, ours need the parts where they belong. Averaging each face with its mirror image
(flip test-time augmentation) gives the ResNet-18 winner +0.24 on validation and +0.71 on
test (85.82; FER2013 part 87.49, RAF-DB part 84.12).

## 2026-10-05: phase 3, known methods one at a time

`scripts/21_optimize.py`, the ResNet-18 winner with one change, 3 seeds each, mean (seed
spread of the test accuracy 0.1 to 0.6):

| Change | Val | Val macro-F1 | Test | Test macro-F1 | FER2013 / RAF-DB part |
|---|---|---|---|---|---|
| None (stage 1) | 85.98 | 78.85 | 85.11 | 77.66 | 86.69 / 83.50 |
| FER+ vote shares as targets | 85.16 | 78.04 | 84.64 | 77.34 | 84.97 / 84.31 |
| Logit adjustment (tau 1, no class weights) | 84.41 | 75.89 | 82.60 | 73.68 | 85.28 / 79.88 |
| EMA of the weights (0.999) | 85.95 | 79.37 | 85.38 | 77.77 | 87.35 / 83.38 |
| SAM (rho 0.05) | 85.82 | 78.95 | 84.98 | 77.35 | 86.73 / 83.21 |
| Without the flagged faces | 85.48 | 78.11 | 84.95 | | |
| Flip test-time augmentation (no training) | 86.22 | 79.10 | 85.82 | | 87.49 / 84.12 |
| Distilled from our nine-run ensemble (T 4) | 86.82 | 80.20 | 85.28 | 77.60 | 87.24 / 83.28 |
| Distilled from FMAE fine-tuned (T 4), open track | **86.89** | **80.88** | **86.65** | **79.37** | **87.84 / 85.43** |

- H38 is wrong as stated: against the majority labels the vote targets cost 1.7 points on
  the FER2013 part (and gain 0.8 on RAF-DB). They teach the annotators' spread, not the
  majority; whether they predict the annotators better is checked separately.
- H39 roughly holds for FMAE: +1.5 on test, +1.9 on the RAF-DB part, at 48 px and the same
  network. Distilling our own ensemble gains 0.8 on validation but only 0.2 on test.
  The student saw FMAE's outputs on training faces only; FMAE never saw the official
  RAF-DB test faces.
- H40 is mostly wrong: EMA adds half a point of macro-F1 and nothing in accuracy, SAM
  nothing at twice the cost; only the flip test-time augmentation adds 0.7 on test (0.24
  on validation).
- H41 is wrong: logit adjustment loses 1.6 points of accuracy and 3 of macro-F1. It moves
  predictions to the rare classes, and their precision falls faster than their recall
  rises.

So with our own data only, the known methods give at most a few tenths of a point on
test. Only outside knowledge (the FMAE teacher) moves the 48 px network noticeably, which
fits phase 2: the gap is in the learned representation.

## 2026-10-05: phase 2, the learning curve

The ResNet-18 winner on 12.5, 25 and 50% of the training set (drawn per source and
class, as many optimiser steps as the full runs), 3 seeds, validation error:

| Training faces | 4,228 | 8,454 | 16,911 | 33,827 |
|---|---|---|---|---|
| Error | 19.7 | 17.6 | 15.9 | 14.0 |

Each doubling still removes about 1.9 points. A power law err = a n^-b + c fits with
b = 0.16 and a floor c that the data cannot pin down (95% interval 0 to 12.3%); it
predicts 12.6% at twice the data. H32 holds: the curve has not flattened, more faces of
the same kind would still help. Together with phases 1 and 3 this is the error budget:
the remaining errors are not mainly a limit of the labels or the input format, but of
what the network can learn from 34,000 faces. Outside knowledge (FMAE as teacher, or
pretraining) is the lever that works, and part of the error is run-to-run variance (the
nine-run ensemble gains 2.7 points).

## 2026-10-05: phase 4, the readout, and phase 3 seen by the annotators

Readout (`scripts/22_bottleneck.py readout`, ResNet-18 winner, 3 seeds): global average
pooling 85.98 validation / 85.11 test; four learned weightings of the 6x6 grid ("spatial")
85.91 / 85.50; a weight per position ("flatten") 85.83 / 85.23. Shifting the validation
faces by up to 4 px costs none of them anything (the augmentation already shifts by up to
5 px). How the head pools does not matter; whatever ties our networks to the arrangement
of the face (the tile-shuffle test) sits in the convolutional features, not in the head.

Phase 3 judged by the annotators instead of the majority label (FER2013 test faces, mean
of 3 seeds; ECE against the majority label):

| | Agreement with one random annotator | Cross-entropy against the vote shares | ECE |
|---|---|---|---|
| Stage 1 | 76.9 | 0.872 | 0.051 |
| Vote targets | 76.1 | **0.749** | 0.152 |
| Logit adjustment | 76.0 | 1.024 | 0.136 |
| EMA | 77.4 | 0.855 | 0.053 |
| SAM | 77.1 | 0.864 | 0.050 |
| Without flagged faces | 77.2 | 0.945 | 0.044 |
| Distilled from our ensemble | 77.4 | 0.840 | **0.039** |
| Distilled from FMAE | **77.7** | 0.799 | 0.090 |

The vote targets do what they were built for: they predict the annotators' spread best
(14% lower cross-entropy), but their single best guess agrees with an annotator less
often, and against the majority label they look badly calibrated because they spread
their confidence. Distilling our ensemble gives the best calibration. No method gets near
the annotators' ceiling (81.4 to 83.0).

## 2026-10-05: phase 5, the mirror-invariant network

`fer/models/mirror.py`, `scripts/23_mirror.py`. Every filter shares its weights with its
mirror image, batch norm shares its statistics within a pair, downsampling pools before
the conv (a stride-2 3x3 conv would break the symmetry), and the head sees a + b and
|a - b| of every feature pair. A face and its mirror image get the same scores to float
precision (tested). The winner's settings, no tuning of its own, 3 seeds:

| | Parameters | Val | Test | Test macro-F1 | FER2013 / RAF-DB part |
|---|---|---|---|---|---|
| ResNet-18 winner | 11.2 M | 85.98 | 85.11 | 77.66 | 86.69 / 83.50 |
| Winner with flip test-time augmentation | 11.2 M | 86.22 | 85.82 | 78.10 | 87.49 / 84.12 |
| Mirror network, same parameters | 11.2 M | **86.88** | **86.25** | 78.68 | 87.53 / 84.95 |
| Mirror network, same compute | 5.6 M | 86.62 | 85.80 | 78.74 | 87.12 / 84.46 |
| ResNet-18, a quarter of the data | 11.2 M | 82.37 | 81.03 | 70.46 | 84.00 / 78.01 |
| Mirror network, a quarter of the data | 11.2 M | 83.37 | 82.30 | 72.03 | 84.11 / 80.46 |

Validation recall per class (winner / mirror, same parameters): angry 84.7 / 84.5, disgust
57.1 / 54.6, fear 70.5 / 63.2, happy 92.7 / 94.3, neutral 84.0 / 85.5, sad 78.3 / 78.3,
surprise 90.4 / 91.2.

H42 is half right. The accuracy holds: +0.9 on validation and +1.1 on test, more than
averaging the winner over mirror images (+0.24, +0.7), and at the same compute with half
the parameters it still gains 0.6 and 0.7. But the gain is not where the guess put it.
Fear and disgust, whose predictions flipped most often, lose recall; the frequent classes
(happy, neutral, surprise) gain. And with a quarter of the data the gain is about the
same (+1.0), not twice as large. The built-in symmetry makes the network better overall,
not by repairing the classes that mirroring confused, and its macro-F1 gain (+1.0 on test)
is smaller than its accuracy gain suggests on the rare classes.

## 2026-10-05: phase 5, combining

The mirror network (same parameters as ResNet-18) with distillation, the winner's
settings, 3 seeds (`scripts/23_mirror.py combine`):

| | Val | Test | Test macro-F1 | FER2013 / RAF-DB part |
|---|---|---|---|---|
| Mirror network | 86.88 | 86.25 | 78.68 | 87.53 / 84.95 |
| ResNet-18 + FMAE as teacher | 86.89 | 86.65 | 79.37 | 87.84 / 85.43 |
| **Mirror network + FMAE as teacher** (open track) | **88.25** | **87.78** | **81.17** | **88.50 / 87.05** |
| ResNet-18 + our ensemble as teacher | 86.82 | 85.28 | 77.60 | 87.24 / 83.28 |
| Mirror network + our ensemble as teacher (scratch track) | 87.10 | 85.95 | 79.02 | 87.82 / 84.06 |

The two gains add up almost exactly in the open track: mirror +1.14, FMAE teacher +1.54,
both +2.67 over the winner. A 48 px network with ResNet-18's parameter count, trained on
our data with FMAE's outputs on the training faces, reaches 87.8 on test: as good as the
nine-run ensemble of part one's winners (87.76), 2.4 points short of FMAE itself (90.18),
at a small fraction of its compute. H44 holds for the open track (87.8, guessed about
87) and misses slightly for the scratch track (86.0, guessed 86.3).

Self-distillation from our own ensemble lifts validation (+0.8 for ResNet-18, +0.2 for the
mirror network) but not test (+0.2, -0.3). A likely reason, found after the fact: the
ensemble members' checkpoints were chosen on the validation set, so a student that
imitates them inherits their fit to it, and its validation score is optimistic. The FMAE
teacher ran a fixed number of epochs without selection, and its validation and test
gains agree. So validation is not a clean signal for self-distilled students; the
scratch-track winner is the plain mirror network (86.25 on test), with the
self-distilled one (87.10 validation, 85.95 test) reported next to it.

## 2026-10-05: phase 5, masked-face pretraining on our own faces

`fer/pretrain.py`, `scripts/24_masked.py`. The ResNet-18 winner's network first learns to
reconstruct 60% masked 8x8 patches of our training faces (150 epochs, no labels; the
error on masked patches falls from 0.97 to 0.42 of the patch variance), then trains on
the labels with the winner's recipe, 3 seeds: validation 86.56 (+0.58), macro-F1 79.59
(+0.74), test 85.59 (+0.48). (The first fine-tuning attempt died in torch's compile worker
pool, not in our code; it was rerun with single-threaded compilation.)

The controlled input changes on seed 0 of every new model (validation):

| | Intact | Prediction changed when mirrored | 2x2 tiles shuffled | 3x3 tiles shuffled | Blur sigma 2 px |
|---|---|---|---|---|---|
| ResNet-18 winner | 86.0 | 8.0% | 57.6 | 38.5 | 54.5 |
| Masked pretraining | 86.5 | 8.2% | 60.8 | 36.8 | 55.3 |
| Mirror network (same parameters) | 86.5 | 0.0% | 58.2 | 34.6 | 59.1 |
| Mirror network (same compute) | 86.6 | 0.02% | 58.3 | 35.3 | 55.6 |
| ResNet-18 + FMAE as teacher | 86.5 | 6.2% | 59.1 | 37.1 | 56.2 |
| ResNet-18 + our ensemble as teacher | 86.6 | 6.0% | 58.2 | 35.5 | 56.4 |
| FMAE fine-tuned | 91.5 | 2.9% | 79.1 | 56.5 | 61.0 |

H43: the masking objective on our 34,000 faces does not give FMAE's part reading (+3
points with shuffled quadrants, where the objective explanation predicted 10 or more and
FMAE is 21 above our winner); it gives half a point of accuracy. So scale, not the
objective alone, is the likelier source; the test cannot separate scale from FMAE's
architecture (a ViT on patch tokens). The student distilled from FMAE learns its outputs
on intact faces but not its part reading. The mirror network's invariance holds on real
faces (no prediction changes), and it is also the most robust to blur.

## 2026-10-05: phase 6, the candidates against the targets

`scripts/25_final.py`, `results/final_part2.json`. Test accuracy averaged over 3 seeds;
the gain over the ResNet-18 winner with a paired bootstrap 95% interval (the same
resampled test faces for both); the official RAF-DB test set at 48 px grayscale; size,
compute (multiply-accumulates per face) and speed (256 faces on the GPU in fp32, one face
on one CPU thread).

| Model | Track | Test | Gain (95% CI) | Macro-F1 | RAF-DB official, 48 px | Parameters | GMACs | GPU, 256 faces | CPU, 1 face |
|---|---|---|---|---|---|---|---|---|---|
| ResNet-18 winner | scratch | 85.11 | | 77.7 | 83.6 | 11.2 M | 1.25 | 43 ms | 20 ms |
| Masked pretraining | scratch | 85.58 | +0.5 (+0.0, +0.9) | 77.2 | 83.9 | 11.2 M | 1.25 | 43 ms | 21 ms |
| Mirror network, same compute | scratch | 85.80 | +0.7 (+0.2, +1.2) | 78.7 | 84.5 | 5.6 M | 1.25 | 51 ms | 30 ms |
| **Mirror network** | scratch | **86.25** | +1.1 (+0.7, +1.6) | 78.7 | **85.0** | 11.2 M | 2.47 | 84 ms | 62 ms |
| Mirror network + our ensemble as teacher | scratch | 85.95 | +0.9 (+0.3, +1.3) | 79.0 | 84.1 | 11.2 M | 2.47 | 84 ms | 63 ms |
| ResNet-18 + FMAE as teacher | open | 86.65 | +1.5 (+1.1, +2.0) | 79.4 | 85.5 | 11.2 M | 1.25 | 44 ms | 21 ms |
| **Mirror network + FMAE as teacher** | open | **87.78** | +2.7 (+2.1, +3.2) | **81.2** | **87.1** | 11.2 M | 2.47 | 84 ms | 64 ms |
| FMAE fine-tuned (the open target) | open | 90.18 | | 84.8 | 91.1 | 303 M | 61.6 | 665 ms (bf16) | |

The mirror network at the same compute is slower than ResNet-18 in practice (1.2x on the
GPU, 1.5x on one CPU thread) because it builds its mirrored filters on every call; for
deployment they would be built once, leaving a plain conv network of ResNet-18's cost.

Against the targets:

| | Target | Best of part two | Reached |
|---|---|---|---|
| A, scratch (our test set) | 85.11 (ResNet-18 winner) | 86.25 (mirror network) | yes, +1.1 (+0.7, +1.6) |
| A, open (our test set) | 90.18 (FMAE fine-tuned) | 87.78 (mirror network + FMAE as teacher) | no, 2.4 short, at 1/25 of the compute and 1/27 of the parameters |
| B, scratch (RAF-DB official) | 83.9 (VGG on RAF-DB alone) | 85.0 (mirror network) | yes, though trained on the combined data |
| B, open (RAF-DB official) | 93.29 native, 91.46 at 48 px (FMAE) | 87.1 at 48 px | no, 4.4 short at the same input |

So far: from our own data, a built-in mirror symmetry is the one change that clearly
helps (+1.1 on test, interval clear of zero), more than every known method of phase 3;
with FMAE as teacher, the mirror network closes half of the distance to FMAE at 48 px.
The open target stays out of reach for a network of ResNet-18's size trained on 34,000
faces: what FMAE has (part reading, a representation learned from 9 million faces) did
not come from our objective, our input, or our head.

## 2026-10-05: part two parked, and how to pick it up again

Part two stops here for now. Done: phases 1 to 6, hypotheses H26 to H44 with their
verdicts, results in `results/references.json` to `results/final_part2.json`. Open, in
the order I would take them:

1. **The references trained from scratch** (study 3 of phase 1): FMAE's ViT-L, POSTER++
   and CLIP's ViT-B/16 on our training set only, 8 to 10 GPU hours each, overnight.
2. **The mirror network in its deployed form**: mirrored filters built once instead of on
   every call, so it runs at ResNet-18's speed; then its int8 and Tucker-2 versions, as in
   the compression part.
3. **More mechanisms from the phase 5 list** (face-coordinate readout, part tokens with
   pairwise interaction, a learned template with a deformation field, annotator-aware
   output, iterative refinement), each against a parameter- and compute-matched baseline.
4. **Guided evolution** over a block grammar that contains the mirror operators, with
   short proxy runs and successive halving, overnight.
5. **The write-up**: part two in the README, and in the report when it is picked up again.

How the runs work, to start again: every part-two script takes its step as the first
argument (`uv run python scripts/23_mirror.py train`, the steps are listed in each
script's docstring), writes its runs under `runs/` and its summary under `results/`, and
skips runs whose record already says done, so a stopped step resumes where it was. One
GPU job at a time: the desktop shares the GPU, and two jobs at once made the training
stutter. Set `TORCHINDUCTOR_COMPILE_THREADS=1`; a crashed compile worker once killed a
long run. The reference weights (about 3 GB) are fetched by `scripts/16_references.py`
into the ignored `runs/references/weights/`.

The browser demo on the portfolio comes from `scripts/26_web.py` (steps `models`,
`samples`, `thumb`, `site`) and the Rust interpreter in `web/fer-wasm`, which reads the
exported model files (`cargo test --release` checks it against the numpy reference;
`RUSTFLAGS="-C target-feature=+simd128" wasm-pack build --target web --release` builds
it). The seven winners there are the main study's, not part two's.
