#set document(title: "Facial emotion recognition: seven architectures and the value of tuning", author: "Rene-Marcel Lehner")
#set page(paper: "a4", margin: (x: 2.3cm, y: 2.4cm), numbering: "1")
#set text(font: "New Computer Modern", size: 10.5pt, lang: "en")
#set par(justify: true, leading: 0.62em)
#set heading(numbering: "1.1")
#show heading: set block(above: 1.3em, below: 0.7em)
#set figure(gap: 0.8em)
#show figure.caption: set text(size: 9pt)
#set table(stroke: 0.4pt + luma(180), inset: 5pt)
#show table: set par(justify: false)

#align(center)[
  #text(size: 16pt, weight: "bold")[Facial emotion recognition: \ seven architectures and the value of tuning]
  #v(0.4em)
  Rene-Marcel Lehner #h(1em) · #h(1em) 2026
]
#v(1em)

#block(inset: (x: 1.2cm))[
  #set text(size: 9.5pt)
  *Summary.* Seven network architectures, from a plain CNN to a vision transformer, are
  trained from scratch on one cleaned dataset of 44,296 faces (FER2013 with FER+ labels,
  plus RAF-DB), with the same pipeline and budget. Each is first trained with educated
  default settings, then tuned with Optuna under an equal budget. The question: which
  architectures depend on tuning, and by how much? The three CNNs with batch norm (VGG,
  ResNet-18, DenseNet) gain less than half a point and work with most settings. The simple
  CNN, ConvNeXt, CCT and ViT gain 3 to 7 points and work only in a narrow band of
  settings. Tuning halves the gap between the best and the worst architecture but does not
  close it: ResNet-18, VGG and DenseNet stay on top at about 85% test accuracy. A second
  stage that also tunes the width, depth and number of stages of these three finds nothing
  better than their published sizes: smaller networks lose accuracy, larger ones gain at
  most 0.2 points. Three follow-ups on these three models: training longer at a constant
  learning rate does not beat the 60-epoch runs. Tests for grokking find late
  generalisation on small training sets in every setting, but never the sudden jump known
  from algorithmic tasks; with batch norm, weight decay only destabilises training.
  Compression works where the layer outputs are low-dimensional: with ranks chosen per
  layer, a short repair and int8, ResNet-18 becomes 16 times smaller at its original test
  accuracy, or 29 times smaller and 6.8 times faster on one CPU thread at 0.7 points less.
]

= Question

Facial emotion recognition sorts a face into one of seven classes: angry, disgust, fear,
happy, neutral, sad, surprise. Many architectures have been applied to it. Papers usually
report each model after its own, often undisclosed, tuning. That hides two things: how
good an architecture is with sensible default settings, and how much of its final score
is owed to tuning.

This study trains seven architectures with one pipeline on one dataset, first with
defaults chosen before any training, then after the same tuning budget for each. The
guesses made beforehand are listed in @sec-models and checked at the end. Three follow-ups
take the strongest three models further: longer training (@sec-long), grokking
(@sec-grok) and compression (@sec-compress).

*Why a rebuild.* An earlier course project (2024) compared a DenseNet with a CNN on
FER2013. A later review of its code found that neither of its two numbers measured test
performance: a swapped return value trained the DenseNet on the test folder and scored it
there, the hyperparameter search therefore optimised a training loss, and the CNN was
scored on unscaled images. This study starts from scratch, with one split, one data
object, one evaluation path for every model, and a test set that never steers a choice:
every decision is made on the validation set.

= Data <sec-data>

== Sources

- *FER2013* @goodfellow2013: 35,887 grayscale faces of 48×48 pixels from web image search,
  split into training (28,709), public test (3,589) and private test (3,589).
- *FER+* @barsoum2016: the same images, each relabelled by ten annotators, with the extra
  options "unknown" and "not a face".
- *RAF-DB* @li2017raf, basic set: 15,339 aligned colour faces (12,271 train, 3,068 test),
  each label agreed by about 40 annotators. It has far more disgust and fear than FER2013.

== What the raw data looks like

Sample sheets of every source and of every suspicious group decided the cleaning rules.

- Most FER2013 images are real faces; stock-photo watermarks are common and harmless.
- A few hundred images are not usable faces: error pages with text, warning icons, black
  or grey frames, a broken-download icon, faces lying sideways. FER+ marks most of them.
- Agreement varies: 8,120 images are unanimous, 3,875 have four or fewer votes for their
  top class. Low-agreement faces are genuinely ambiguous (grimaces, yawns, hands on faces).
- The original FER2013 label differs from the FER+ majority for about a third of the
  images that are kept. FER2013's single labels are noisy.
- FER2013 contains exact and near copies, many of them across its own training and test
  split. Left in, they let a model score on faces it has already seen.
- RAF-DB faces are aligned and tightly cropped; FER2013 faces are looser. Mixed as they
  are, a model could tell the two sources apart by framing alone.

== Cleaning and unification

#figure(image("figures/dropped.svg", width: 92%), caption: [Images dropped, by reason.]) <fig-dropped>

- *Labels.* FER2013 takes the FER+ majority. An image stays only if at least 5 of its 10
  annotators agree on one of the seven emotions. This removes non-faces, "unknown",
  contempt (not one of the seven), and ambiguous faces.
- *Duplicates.* A 64-bit perceptual hash @zauner2010 compares every image with every other,
  and with every mirrored one. Sheets of pairs at growing distance showed real copies up
  to about 5 bits and mostly different faces beyond. Copies form groups: a group whose
  labels disagree is dropped, otherwise one copy stays, the test copy if there is one.
  1,852 groups (4,424 images) were found; 731 of them crossed the splits.
- *Framing.* FER2013 is cropped to 90% of its size, centred 2 px lower. This crop gives the
  best match between the average faces of the two sources (correlation of their edge
  maps 0.24 before, 0.60 after). RAF-DB is resized to 48×48 and converted to grayscale.
- *Splits.* FER2013's training, public test and private test sets become training,
  validation and test. RAF-DB gives 10% of its training set to validation and keeps its
  test set. No image in training has a near copy in validation or test (checked by a test).

#figure(image("figures/class_counts.svg", width: 92%), caption: [Training images per class and source. Disgust and fear are rare; RAF-DB supplies most of them.]) <fig-counts>

The result has 44,296 images: 33,827 for training, 4,316 for validation and 6,153 for
testing (3,098 from FER2013, 3,055 from RAF-DB). The classes stay imbalanced
(@fig-counts): disgust and fear are about 2% of the training images each.

== Augmentation

Each epoch, every training image is changed at random, with transforms that keep the
expression: mirroring, a small rotation, zoom and shift (up to 15° and 10%), brightness
and contrast, and random erasing of a patch @zhong2020, which imitates hands and hair in
front of the face. One strength parameter scales all of them, so the tuning can choose
how much augmentation each model gets. Augmenting online gives new variants every epoch;
mirroring online has the same effect as storing mirrored copies.

= Models and hypotheses <sec-models>

The architectures form a line, each adding one idea to the one before (@tab-models). All
take a 48×48 grayscale face and are trained from scratch; an ImageNet-pretrained ResNet-18
@he2016 serves as a reference. The architectures stay fixed; tuning only changes how they
are trained.

#figure(
  table(
    columns: (auto, 1fr, auto, auto),
    align: (left, left, right, right),
    table.header([*Model*], [*What it adds*], [*Parameters*], [*GMACs*]),
    [Simple CNN @lecun1998], [Convolutions: local patterns, shared weights], [1.27 M], [0.02],
    [VGG-style @simonyan2015], [Depth, small 3×3 filters, batch norm @ioffe2015], [7.05 M], [0.47],
    [ResNet-18 @he2016], [Skip connections: deep networks stay trainable], [11.2 M], [1.25],
    [DenseNet-BC @huang2017], [Every layer sees all earlier layers: feature reuse], [0.77 M], [0.16],
    [ViT @dosovitskiy2021], [Attention instead of convolution], [3.73 M], [0.53],
    [ConvNeXt @liu2022], [A CNN rebuilt with transformer habits], [3.38 M], [0.10],
    [CCT @hassani2021], [Convolutions make the tokens, attention combines them], [3.88 M], [0.62],
    [ResNet-18, ImageNet], [Pretraining on 1.2 million photos (reference)], [11.2 M], [0.33],
  ),
  caption: [The seven architectures and the reference. GMACs: multiply-adds per face.],
) <tab-models>

Sizes follow published configurations: VGG with four stages of two convolutions, ResNet-18
with the small-image stem, DenseNet-BC with depth 100 and growth rate 12, ViT-Lite-7/4 and
CCT-7/3×2 from @hassani2021, and ConvNeXt at the Atto size @wightman2019.

Guesses written down before any training:

+ *Ranking.* ResNet, VGG and DenseNet end close together at the top; the simple CNN is
  clearly last; ConvNeXt stays below ResNet on this little data; CCT beats ViT by several
  points, and ViT lands near the simple CNN. Convolutions build in that nearby pixels
  belong together and that a pattern means the same wherever it appears. Models without
  that bias (ViT) must learn it from about 34,000 images.
+ *Pretraining* beats every from-scratch model by 2 to 5 points.
+ *Rare classes.* All models do worst on disgust and fear; macro-F1 separates the models
  more than accuracy does.
+ *Efficiency.* DenseNet matches ResNet with far fewer parameters, but trains slower.
+ *Tuning gain.* ViT and ConvNeXt gain most from tuning, ResNet and VGG least, the simple
  CNN little (its limit is capacity).
+ *Spread.* Transformer-like models show the widest spread of results across settings;
  CNNs with batch norm the narrowest.
+ *What matters.* The learning rate matters most for every model; augmentation strength
  comes second for the transformers.

= Training setup

Every model uses the same loop: AdamW @loshchilov2019 with a linear warmup and cosine
decay, 60 epochs, batches of 256, label smoothing 0.1 @szegedy2016, square-root class
weights against the imbalance, and the augmentation above at full strength. The
transformers and ConvNeXt start from a lower learning rate and use stochastic depth
@huang2016sd, as in DeiT @touvron2021 and CCT. The model kept from each run is the epoch
with the best validation macro-F1: accuracy would reward ignoring the rare classes.

Training runs on one RTX 4070 Ti (12 GB). The whole dataset (100 MB) sits on the GPU,
each epoch is augmented there in one call (about 60 ms), and the models run in bfloat16
and compiled. Two findings along the way:

- *Speed does not follow FLOPs.* DenseNet needs half the multiply-adds of ResNet-18 but
  was six times slower at full resolution. Its time goes into copying and re-normalising
  the growing stack of features in every layer; convolutions took 7% of it. A
  memory-efficient implementation @pleiss2017 saved memory, not time. DenseNet therefore
  starts with one stride-2 convolution; its blocks work at 24, 12 and 6 pixels. At that
  size the memory-efficient version is no longer needed; dropping it (during stage 2) made
  DenseNet 1.6 to 1.9 times faster. It still runs at a few percent of the GPU's peak
  arithmetic rate, against about 40% for ResNet-18.
- *Compilation* speeds up the attention models by a third and ResNet and DenseNet by about
  a tenth; for the small CNNs it makes no difference.

= Baselines <sec-baselines>

Every model was trained with its default settings for 60 epochs, with two seeds. @tab-baselines
gives the test scores of the epoch with the best validation macro-F1.

#let baselines = csv("tables/baselines.csv")
#figure(
  text(size: 8pt, table(
    columns: baselines.first().len(),
    align: (left,) + (right,) * (baselines.first().len() - 1),
    table.header(..baselines.first().map(h => [*#h*])),
    ..baselines.slice(1).flatten(),
  )),
  caption: [Baselines on the test set (6,153 faces), mean ± spread over two seeds, in %.
    ECE: expected calibration error (lower is better). Seconds per epoch on one RTX 4070 Ti.],
) <tab-baselines>

- *Three CNNs with batch norm lead, and cannot be told apart.* ResNet-18 (85.0%), VGG and
  DenseNet (84.3% each) are within one point. The test set's own sampling error at this
  accuracy is about ±0.9 points (95%), so none of them clearly wins.
- *Everything built on attention or LayerNorm trails with default settings.* CCT (78.0%)
  and ConvNeXt (76.9%) end level with the simple CNN (78.2%); the plain ViT is last (71.8%).
- *They underfit.* At the end of training the CNNs with batch norm reach 91 to 95% on the
  (augmented) training images; the attention models and ConvNeXt reach only 74 to 81%,
  barely above their test scores. Their defaults, not their capacity, hold them back. This
  is where tuning should help most.
- *Pretraining buys nothing here.* The ImageNet ResNet-18 scores 84.4%, level with the same
  network trained from scratch. It fits the training images best (97.5%) and generalises no
  better; its disgust recall is the lowest of the strong models (48%).
- *Rare classes are the hard part for every model* (@fig-recall): disgust and fear reach
  48 to 63% recall, happy and surprise 80 to 92%.
- *Parameters and speed do not follow each other.* DenseNet matches VGG with 9 times fewer
  parameters and ResNet-18 with 15 times fewer, but is the slowest per epoch.

#figure(image("figures/baseline_curves.svg", width: 100%), caption: [Validation curves of the baselines: mean over two seeds, band from minimum to maximum.]) <fig-curves>

#figure(image("figures/baseline_recall.svg", width: 92%), caption: [Recall per class on the test set (%), mean over two seeds.]) <fig-recall>

= Tuning and sensitivity <sec-tuning>

Each model gets 30 Optuna @akiba2019 trials with the TPE sampler @bergstra2011. The first
trial uses the defaults, the next 9 are random: they sample the search space without bias
and give the spread across settings. All studies use the same random seed, so these 9
trials try the same settings on every model (the learning rate relative to the model's
default), which makes them a paired comparison. Every trial runs the same shortened schedule of 25 epochs to the end, without
pruning, so all trials can be compared. The search ranges are educated guesses centred on
each model's defaults: the learning rate from a tenth to ten times the default, weight
decay from 0.0005 to 0.5, batch size 128, 256 or 512, label smoothing up to 0.2,
augmentation strength up to 1.5, three class weightings, up to 10 warmup epochs, and
dropout or stochastic depth. The default settings are the first trial of every study. The
best settings are retrained for the full 60 epochs with three seeds and scored once on the
test set.

#let tuning = csv("tables/tuning.csv")
#figure(
  text(size: 7.5pt, table(
    columns: tuning.first().len(),
    align: (left,) + (right,) * (tuning.first().len() - 2) + (left,),
    table.header(..tuning.first().map(h => [*#h*])),
    ..tuning.slice(1).flatten(),
  )),
  caption: [Tuning, 30 trials of 25 epochs per model, validation macro-F1 in %. Default and
    Best: the default trial and the best trial. Default, 60 ep.: the baselines on
    validation, for comparison. Random: the 9 random trials. Within 1 pt: share of all
    trials within one point of the best. Most important: the top two settings by fANOVA
    @hutter2014.],
) <tab-tuning>

- *Two groups.* The three CNNs with batch norm gain 2 to 3 points from tuning, and 30 to
  43% of their trials land within a point of the best: many settings work. The simple
  CNN, ViT, ConvNeXt and CCT gain 4 to 9 points, and only 7 to 10% of their trials come
  that close (@tab-tuning, @fig-trials).
- *The same settings break some models and not others.* One of the paired random trials
  combines nine times the default learning rate, no warmup and inverse-frequency class
  weights. VGG (1.6%), ViT (4.5%) and ConvNeXt (5.1%) collapse: they end up predicting
  only one to three of the rarest classes, which these weights favour most. CCT (24.5%) and
  the simple CNN (32.6%) barely learn. ResNet-18 (70.3%) and DenseNet (65.7%) keep
  training. They are the only models in which every layer also has a direct path to
  the output (skip or dense connections); VGG has batch norm but no such path.
- *How training starts matters most.* Warmup is the most important setting for four models,
  the learning rate for ViT and CCT, class weighting for ResNet-18 (@fig-importance).
  Warmup and learning rate act together: a high learning rate needs a warmup, and for five
  models the worst trial is the one above. Augmentation, weight decay and batch size
  explain less than 7% each, for every model. For batch size this is a weak estimate: TPE
  settled on 128 for all seven models within a few trials, so 256 and 512 together were
  tried only 5 or 6 times per model.
- *The best settings are alike.* All seven picked batches of 128 and 4 to 9 warmup epochs.
  The CNNs and ConvNeXt picked about four times their default learning rate (0.0036 to
  0.0043); ViT and CCT stayed near theirs (0.0004 and 0.0005). Inverse-frequency class
  weights hurt every model; the best trials of ViT and CCT use no class weights, those of
  the CNNs square-root weights. Six of the seven picked lighter augmentation than the
  default (strength 0.3 to 0.6 instead of 1).
- *A 25-epoch gain is not a 60-epoch gain.* The defaults were chosen for 60 epochs. With 60
  epochs they beat the best 25-epoch trial for ViT, CCT and DenseNet, and tie it for
  ResNet-18 (@tab-tuning, column "Default, 60 ep."). Much of ViT's gain of 8 points makes
  up for the short schedule. The trial results measure how sensitive each model is under
  a fixed short budget; the gain from tuning itself comes from the final runs, where the
  tuned settings and the defaults both get 60 epochs.

#figure(image("figures/tuning_trials.svg", width: 100%), caption: [Every trial of every study. Pale dots: random trials, solid dots: TPE trials, bar: the default settings. Dots near zero: trials that collapsed.]) <fig-trials>

#figure(image("figures/tuning_importance.svg", width: 100%), caption: [fANOVA importance of each setting, in % of the variance it explains across the 30 trials of a study.]) <fig-importance>

== The tuned models on the test set

Each study's best settings were retrained for 60 epochs with three seeds and scored once
on the test set (@tab-final, @fig-final).

#let final = csv("tables/final.csv")
#figure(
  text(size: 8pt, table(
    columns: final.first().len(),
    align: (left,) + (right,) * (final.first().len() - 1),
    table.header(..final.first().map(h => [*#h*])),
    ..final.slice(1).flatten(),
  )),
  caption: [Default settings (baselines, two seeds) against tuned settings (three seeds), both
    60 epochs, test set, in %. The interval is a bootstrap over the test images.],
) <tab-final>

- *Tuning pays where the defaults were wrong.* ViT gains 6.6 points of accuracy, CCT 3.6,
  ConvNeXt 3.2 and the simple CNN 3.0. The three CNNs with batch norm gain 0.1 to 0.4
  points, less than the test set's sampling error: for them, the defaults chosen before
  training were as good as 30 trials of tuning.
- *Tuning halves the gap between architectures, but does not close it.* With defaults, 13
  points separate the best model from the worst; after tuning, 7 (85.1 against 78.4).
  ResNet-18, VGG and DenseNet stay on top and cannot be told apart (@tab-gaps). CCT and
  the simple CNN follow 3 points behind, then ConvNeXt and ViT.
- *The short trials still found good long settings.* For ViT and CCT, the 60-epoch
  defaults beat every 25-epoch trial on validation. Yet the best trial's settings, trained
  for 60 epochs, beat the defaults by 3.6 to 6.6 points on the test set: the settings that
  won after 25 epochs were also better after 60.

#let gaps = csv("tables/final_gaps.csv")
#figure(
  text(size: 8pt, table(
    columns: gaps.first().len(),
    align: (left, left, right, right, left),
    table.header(..gaps.first().map(h => [*#h*])),
    ..gaps.slice(1).flatten(),
  )),
  caption: [Neighbours in the tuned ranking: the accuracy gap and its 95% interval from a
    paired bootstrap over the test images. Clear: the interval excludes zero.],
) <tab-gaps>

#figure(image("figures/final_gain.svg", width: 100%), caption: [Test scores with default settings (open) and tuned settings (filled), 60 epochs each.]) <fig-final>

= The network's shape <sec-shape>

Stage 1 keeps every architecture at its published size. Stage 2 asks what the shape adds,
for the three strongest: ResNet-18, VGG and DenseNet. Each gets a new study of 40 trials
in which three shape settings join the training settings:

- *Width:* a multiplier on the channels of every layer, from 0.25 to 1.5 (for DenseNet on
  the growth rate, from 0.5 to 2).
- *Depth:* blocks per stage for ResNet-18 and convolutions per stage for VGG (1 to 3),
  layers per dense block for DenseNet (6 to 20).
- *Stages:* how often the image is halved, 3 or 4 (DenseNet: 2 or 3 dense blocks).

This covers 0.08 to 39 M parameters for ResNet, 0.25 to 23 M for VGG and 0.03 to 4.4 M for
DenseNet. Batch size and class weights stay at the stage 1 best (all three chose 128 and
square-root weights); the other training settings are searched over the same ranges as
before. The first trial is the stage 1 best with the default shape, so the gain over it is
what the shape adds. Eleven random trials follow, then TPE. Guesses written before the
first trial:

8. *Shape against settings.* The shape settings together matter less than the training
  settings. The three models reach the same accuracy at 0.8 to 11 M parameters, so size
  does not seem to be what limits them.
9. *Gain.* The best shape improves on the start by less than a point.
10. *Smaller networks.* For each model, a network with at most a quarter of the default's
  parameters comes within a point of the best.

#let shape = csv("tables/shape.csv")
#figure(
  text(size: 7.5pt, table(
    columns: shape.first().len(),
    align: (left,) + (right,) * 6 + (left, left),
    table.header(..shape.first().map(h => [*#h*])),
    ..shape.slice(1).flatten(),
  )),
  caption: [Stage 2, 40 trials of 25 epochs per model, validation macro-F1 in %. Start: the
    stage 1 best with the default shape. Shape share: the combined fANOVA importance of
    width, depth and stages. Depth counts blocks (ResNet-18), convolutions (VGG) or layers
    (DenseNet) per stage.],
) <tab-shape>

- *The published sizes are close to the best.* For ResNet-18 and DenseNet the best shape
  beats the start by 0.4 and 0.3 points. That is within the noise: rerunning the stage 1
  best settings gave 0.1 to 0.7 points less than the same settings had in stage 1. Only
  VGG gains clearly, 1.4 points, with three convolutions per stage instead of two and 20%
  more width; seven of its eight best trials use the deeper layout.
- *Smaller costs accuracy, larger barely pays.* Every model's best trial is larger than
  its default (1.2 to 2.1 times the parameters) and keeps its number of stages. Networks
  with at most two thirds of the default's parameters score 1.1 to 1.9 points below the
  best, at most a third 1.9 to 4.3 points, at most a tenth 3.8 to 8 points
  (@fig-shape-size).
- *How much the shape matters depends on the range.* The shape settings explain 71% of
  the spread for ResNet-18 (width alone 57%), 44% for VGG and 26% for DenseNet
  (@fig-shape-importance). For ResNet-18 that share comes from the narrow end of the range,
  which reaches down to 0.08 M parameters, where scores fall by up to 12 points. Among the
  good networks, the shape changes little.

#figure(image("figures/shape_size.svg", width: 100%), caption: [Every stage 2 trial: validation macro-F1 against the number of parameters (log scale). Ring: the start, the stage 1 best with the default shape.]) <fig-shape-size>

#figure(image("figures/shape_importance.svg", width: 100%), caption: [fANOVA importance in stage 2, in % of the variance explained across the 40 trials of a study.]) <fig-shape-importance>

The best trial of each study, and DenseNet's smallest network within a point of the best
(0.52 M parameters), were retrained for 60 epochs with three seeds. @tab-shape-final
compares them with the stage 1 final runs on the same test images. For ResNet-18 and VGG
no network smaller than the default came within a point of the best, so they have no
second pick.

#let shapefinal = csv("tables/shape_final.csv")
#figure(
  text(size: 8pt, table(
    columns: shapefinal.first().len(),
    align: (left, left) + (right,) * 5,
    table.header(..shapefinal.first().map(h => [*#h*])),
    ..shapefinal.slice(1).flatten(),
  )),
  caption: [Stage 2 against stage 1 on the test set, three seeds each, in %. The interval is a
    paired bootstrap over the test images for the accuracy difference.],
) <tab-shape-final>

- *The shape adds nothing measurable.* All three best shapes score 0.2 points above their
  stage 1 counterparts, with intervals that include zero. VGG's lead of 1.4 points on
  validation was mostly the luck of picking the best of 40 noisy trials.
- *The smaller DenseNet does not hold up.* With two thirds of the parameters it loses 1.2
  points of accuracy and 2.5 of macro-F1 on the test set, a clear difference, although it
  was within a point of the best on validation.
- *For these three models the published sizes were already right.* Stage 2 cost about 11
  GPU hours and changed no conclusion of stage 1.

= Long training <sec-long>

The final runs train for 60 epochs with a learning rate that decays to zero, so their
length is fixed in advance and leaves open whether the models had converged. The three
strongest models were trained further, in three runs each, for 100 epochs at a constant
learning rate after the warmup:

- *Tuned:* the stage 1 best settings at the default shape.
- *Larger:* the stage 2 best shape and settings.
- *Memorize:* the stage 1 settings without augmentation, label smoothing and dropout, and
  with the weight decay set so that learning rate times weight decay is $10^(-3)$ per
  step, as in the original grokking runs @power2022. A network that can learn its
  training set by heart, to see whether it starts to generalise late.

A constant learning rate leaves the runs open-ended. To see what a run would reach if it
ended at a given epoch, a branch of 10 epochs with the learning rate falling linearly to
zero (a cooldown, as in warmup-stable-decay schedules @hagele2024) starts from its
checkpoint and is scored on validation and test; the main run never sees it. Besides loss
and accuracy, every epoch records the spectral entropy of each weight matrix (1 when all
its directions carry equal weight, lower when it is closer to low rank) and neural
collapse (NC1 @papyan2020: the spread of the penultimate features within a class against
the spread between classes, on training faces). Guesses written before the runs:

11. *Convergence.* Validation accuracy levels off within 100 epochs while the clean
  training accuracy keeps rising. After the cooldown the runs land within a point of the
  60-epoch final runs: training longer does not help by itself.
12. *No grokking yet.* The memorize runs learn more than 95% of the training faces within
  100 epochs; their validation accuracy falls behind and does not recover.
13. *Compressibility.* The weights' spectral entropy falls during long training, most in
  the memorize runs.

#let longt = csv("tables/long.csv")
#figure(
  text(size: 7.5pt, table(
    columns: longt.first().len(),
    align: (left, left) + (right,) * 6,
    table.header(..longt.first().map(h => [*#h*])),
    ..longt.slice(1).flatten(),
  )),
  caption: [Long runs, one seed each, in %. Clean train: 4,316 fixed training faces without
    augmentation. Test after a 10-epoch cooldown from epoch 100. Last column: the mean of the
    60-epoch final runs (three seeds) with the same shape and settings.],
) <tab-long>

#figure(image("figures/long_curves.svg", width: 100%), caption: [The long runs per epoch, at a constant learning rate (the cooldowns are not shown). Bottom: the spectral entropy of the weights, averaged over the conv and dense layers.]) <fig-long>

- *Longer alone does not help.* After the cooldown the tuned runs score 0.2 to 0.6 points
  below their 60-epoch counterparts, the larger runs between 0.1 below and 0.6 above, all
  within the noise of one seed (@tab-long).
- *Validation rises slowly, the fit to the training set faster.* After epoch 40, VGG's
  validation accuracy gains less than a point; ResNet-18 and DenseNet gain 1 to 2 points
  and are still improving at epoch 100 (@fig-long). The clean training accuracy keeps
  rising, to 95 to 99% for VGG and ResNet-18 and 86 to 89% for DenseNet.
- *The memorize runs do not memorise at a constant learning rate.* They fit 86 to 94% of
  the clean training faces; the noise of the steps keeps them from the rest, and only the
  cooldown takes them to 100%. Their validation accuracy peaks early (epoch 18 for VGG)
  and then falls; after the cooldown they score 1.5 to 4.4 points below the tuned runs.
- *The weights drift towards low rank,* in every run and most in the memorize runs (VGG:
  0.996 to 0.901).

= Grokking <sec-grok>

Grokking is generalisation long after a network has learned its training set by heart
@power2022. It was found on small algorithmic tasks and, with large initial weights, on
MNIST @liu2023omnigrok. Two tests here.

*VGG with batch norm.* VGG's memorize run after its cooldown (every training face learned,
81.5% validation accuracy) trained on for 1,000 epochs at a tenth of the learning rate and
the same weight decay. The guess:

14. *No grokking.* Validation accuracy rises by less than a point: the network already
  generalises when it has memorised 34,000 faces, so little is left to find late.

#figure(image("figures/grok_vgg_bn.svg", width: 100%), caption: [VGG with batch norm, continued from its memorised state for 1,000 epochs: validation accuracy and training loss (log scale) per epoch.]) <fig-grok-vgg>

There was no grokking (@fig-grok-vgg). Validation accuracy between the loss spikes peaked
at 82.2% and drifted down to about 80%; after a cooldown the run scores 79.8% on the test
set, against 81.5% where it started. Eleven loss spikes occurred, two of them near-total
collapses (training accuracy down to 35% and 32.5%). With batch norm, the size of the conv
weights does not change what the network computes, so weight decay cannot favour a
simpler solution. It only shrinks the weights, which raises the effective step size until
training breaks @vanlaarhoven2017 @lobacheva2021.

*Without batch norm, on 1,000 faces.* The setting of @liu2023omnigrok: the simple CNN,
which has no batch norm, on 1,001 training faces drawn in class proportion, MSE loss on
one-hot targets, AdamW at a learning rate of $10^(-3)$, no augmentation, $10^5$ steps of
200 faces in float32. The weights start at 1 or 3 times their usual size (3 times scales
the outputs by $3^5 = 243$; the factor 8 used on MNIST killed every ReLU), each with
weight decay 0.1, 0.01 and 0. A last run trains VGG without batch norm at a start that
gives the same spread of outputs across faces as the CNN's 3x start (2.8x, weight decay
0.1). The guesses:

15. *Grokking on faces.* With weight decay 0.1 the validation accuracy of the 3x start
  rises long after the training faces are memorised, towards the 1x start's level, and
  gradually, as on MNIST. With 0.01 the rise comes later; without weight decay validation
  accuracy stays low.
16. *VGG without batch norm.* A late rise of about 5 points (its NC1 was already low after
  2,000 steps of a test run).

#let grokt = csv("tables/grok.csv")
#figure(
  text(size: 7.5pt, table(
    columns: grokt.first().len(),
    align: (left,) + (right,) * 7,
    table.header(..grokt.first().map(h => [*#h*])),
    ..grokt.slice(1).flatten(),
  )),
  caption: [Grokking on 1,000 training faces, validation accuracy in %, one seed each, up to
    $10^5$ steps. Half-way: the step at which validation accuracy first covers half the
    distance from its lowest point after memorising to its peak.],
) <tab-grok>

#figure(image("figures/grok_curves.svg", width: 100%), caption: [Validation accuracy, NC1 on the training faces (log scale) and the weight norm relative to the usual start, against training steps (log scale).]) <fig-grok>

- *Every run generalises late, but gradually.* Validation accuracy rises in all seven runs
  long after the training faces are memorised; half of the rise comes 11 to 19 times
  later (@tab-grok). The rise spreads over about a decade of steps, with no jump.
- *Large starting weights delay it.* The 3x start begins 9 to 12 points lower and reaches
  half-way twice as late as the 1x start. With weight decay 0.1 it closes most of the gap
  (54.6% against 57.3% at the end).
- *Weight decay sets the size of the rise, not its timing.* All three 3x runs reach
  half-way between 16,250 and 18,500 steps, where @liu2023omnigrok would expect the time
  to grow with 1 / weight decay. Without weight decay the gain fades again while the
  weights grow to 8.6 times their usual size.
- *Neural collapse moves with the rise.* In every 3x run NC1 falls about tenfold during
  the rise; at the 1x start it is already low when the faces are memorised (@fig-grok).
  This fits the account of grokking as a change from lazy to rich learning @kumar2024: the
  large start first fits the training faces with its random features and generalises
  once the features themselves change. The measures proposed as signals of grokking do
  not lead here: the spectral entropy of the penultimate features @truong2026 peaks at or
  after the half-way point, and the absolute weight entropy @golechha2024 follows the
  weight norm, falling with weight decay and rising without it.
- *VGG without batch norm behaves alike and ends lower.* It memorises at 45.3%, peaks at
  54.5% after 74,250 steps and then slips, below the simple CNN on the same faces. Every
  5,000 steps or so its weight norm falls and jumps back by 40% within 250 steps, the
  slingshot pattern of Adam on a fitted training set @thilak2022; validation is no higher
  after a jump than before it.

On faces, generalisation comes late in every setting tried, but never as the sudden jump
known from the algorithmic tasks.

= Compression <sec-compress>

Can the three strongest networks be made much smaller or faster without losing accuracy?
Four methods, applied to the larger long-run models after their cooldown (VGG 14.8 M
parameters, ResNet-18 13.7 M, DenseNet 1.5 M):

- *Output-based low rank* @zhang2016. A layer's outputs on 4,096 training faces are
  projected onto their main directions, after scaling each channel to unit variance (what
  the following batch norm sees). The layer then computes only $r$ channels, and a 1×1
  convolution maps them back to the full width.
- *Tucker-2* @kim2016, from the weights alone: a 3×3 convolution becomes a 1×1 convolution
  into fewer channels, a smaller 3×3 core, and a 1×1 convolution out.
- *Spatial split* @jaderberg2014: a 3×3 convolution becomes a 3×1 convolution into $R$
  channels and a 1×3 convolution, from the singular value decomposition of the weights
  reshaped to $(C_"in" dot 3) times (C_"out" dot 3)$ @tai2016.
- *int8* @jacob2018: weights and activations as 8-bit integers after training, calibrated
  on training faces. PyTorch runs this on the CPU only.

*Why it should work.* A convolution with $C_"in"$ input and $C_"out"$ output channels and
$k times k$ kernels has $C_"out" C_"in" k^2$ weights; at rank $r$ the factorised layer has
$r (C_"in" k^2 + C_"out")$, about $r slash C_"out"$ of the original, and the same share of
the multiply-adds. The best rank-$r$ version loses exactly the energy beyond the $r$-th
direction (Eckart-Young). Measured before any compression: the weights are close to full
rank (keeping 95% of their squared singular values saves only 1.2 to 1.7 times the
parameters), while the layer outputs are low-dimensional (95% of their variance fits into
4 to 5.5 times fewer parameters for VGG and ResNet-18, 2 times for DenseNet). In VGG's
last convolution, 7 of 616 directions hold 95% of the output variance: neural collapse
@papyan2020 squeezes the deep features of 7 classes towards at most 6 dimensions.

*Procedure.* First a sweep with one threshold for all layers (the share of variance or of
squared singular values each layer keeps) and no fine-tuning. Then ranks per layer: every
layer alone at each candidate rank, scored by the KL divergence from the original
predictions on 2,048 other training faces; the ranks with the smallest summed KL that meet
a budget of 2, 4 or 8 times fewer parameters; then 10 epochs of repair on the training set
with the model's own settings, the learning rate falling from a tenth of the original to
zero (a pilot found no gain from a higher rate or from distillation @hinton2015). Then
int8 on top of the repaired models, quantisation-aware training for DenseNet, and networks
of the same size trained from scratch, the fair baseline @liu2019rethinking. Speed is the
median over repeated runs in fp32, on the GPU and on the CPU with 8 threads, for one face
and for 256. Guesses written before the respective runs:

17. *Output-based low rank.* At 99% of the output variance VGG and ResNet-18 lose less
  than a point with about 2 times fewer parameters; at 95% several points. DenseNet saves
  at most 1.3 times at 99%.
18. *Tucker-2.* At the same size it loses more accuracy than output-based low rank.
19. *Spatial split.* Under 2 times fewer parameters before accuracy drops.
20. *int8.* 4 times smaller with under half a point lost; 1.5 to 3 times faster on the CPU
  for 256 faces, less for one.
21. *Speed.* No factorisation makes one face faster on the GPU; on the CPU the speedup
  stays below the saving in multiply-adds.
22. *Repair.* Within a point of the uncompressed model at 2 and 4 times fewer parameters
  for every method on VGG and ResNet-18; at 8 times only low rank stays within a point.
23. *The fair baseline.* A network of the same size trained from scratch is no worse than
  the repaired one.
24. *Both together.* int8 on top of a repaired model gives another 4 times in size and
  costs less than half a point more.
25. *DenseNet in int8.* Quantisation-aware training brings it to within a point of the
  float model.

#let cthr = csv("tables/compress_threshold.csv")
#figure(
  text(size: 7.5pt, table(
    columns: cthr.first().len(),
    align: (left, left) + (right,) * 6,
    table.header(..cthr.first().map(h => [*#h*])),
    ..cthr.slice(1).flatten(),
  )),
  caption: [One threshold for all layers, no fine-tuning: the strongest setting of each
    method within half a point of the uncompressed validation accuracy. Speed for one face:
    how many times faster than the uncompressed model, fp32, on the CPU (8 threads) and the
    GPU.],
) <tab-compress-threshold>

#let crep = csv("tables/compress_repair.csv")
#figure(
  text(size: 7pt, table(
    columns: crep.first().len(),
    align: (left, left, left) + (right,) * 7,
    table.header(..crep.first().map(h => [*#h*])),
    ..crep.slice(1).flatten(),
  )),
  caption: [Ranks chosen per layer for 2, 4 and 8 times fewer parameters, validation
    accuracy before and after 10 epochs of repair, test accuracy after, in %. Missing
    targets were out of reach: Tucker-2 and the spatial split only touch the 3×3
    convolutions.],
) <tab-compress-repair>

#let cint = csv("tables/compress_int8.csv")
#figure(
  text(size: 7pt, table(
    columns: cint.first().len(),
    align: (left, left) + (right,) * 5,
    table.header(..cint.first().map(h => [*#h*])),
    ..cint.slice(1).flatten(),
  )),
  caption: [int8 on the CPU (8 threads), alone and on top of the repaired models, in %.],
) <tab-compress-int8>

#figure(image("figures/compress_tradeoff.svg", width: 100%), caption: [Validation accuracy against size. Dashed: one threshold for all layers, no fine-tuning. Dots: ranks per layer, after repair. Crosses: int8 versions.]) <fig-compress>

- *The outputs are compressible, the weights much less.* With one threshold and no
  fine-tuning, output-based low rank keeps the accuracy at 2.4 times fewer parameters for
  VGG and 4.2 for ResNet-18; Tucker-2, which only sees the weights, at 1.2 and 2.3
  (@tab-compress-threshold). The spatial split does better than expected (2 and 3.1
  times): the weights are close to full rank across channels, but not across the two
  directions of the kernel.
- *One threshold fails where the features have collapsed.* VGG's low rank breaks down
  below 99% of the variance (51.8% validation accuracy at 95%): the threshold leaves 7
  directions in the last convolution and 4 in the first dense layer, too few to separate
  7 classes. Ranks chosen per layer from their measured effect avoid this: at 4 times
  fewer parameters VGG keeps 85.8% before any repair.
- *ResNet-18 compresses best.* After repair every method stays within a point of the
  uncompressed model at 8 times fewer parameters (test 83.8 to 84.9% against 85.2%,
  @tab-compress-repair); its skip connections carry the signal past the approximated
  convolutions. VGG holds at 4 and 8 times only with low rank (test 84.2 and 84.3%
  against 85.6%). DenseNet holds 2 times with low rank; its 1×1 layers, two thirds of its
  parameters, are out of reach of the methods for 3×3 kernels, and the per-layer KL does
  not add up over its 106 densely connected layers.
- *Repair recovers most of a deep cut, but not all.* VGG's Tucker-2 and spatial split at 4
  times cut the high-resolution layers, which are cheap in parameters, until only 55 M of
  1,195 M multiply-adds remained; before repair they scored 57% and 30%, after it 83% and
  77%. At 2 times the repair can cost half a point: restarting training moves a
  cooled-down model out of its minimum.
- *int8 is the simplest win, and it stacks.* After training it makes VGG and ResNet-18 4
  times smaller at no cost in accuracy and 2 to 3.5 times faster on the CPU. On top of a
  factorised model it costs at most 0.3 points more (@tab-compress-int8): ResNet-18 with
  Tucker-2 at 4 times plus int8 is 16 times smaller (3.2 MB) and as accurate as the
  original on the test set (85.4% against 85.2%); with the spatial split at 8 times it is
  29 times smaller (1.8 MB), 0.7 points less accurate and 6.8 times faster on one CPU
  thread. DenseNet breaks in int8 (60.7%): PyTorch gives all inputs of a concatenation one
  shared scale, and DenseNet concatenates up to 18 feature groups of very different size.
  Quantisation-aware training recovers it only to 79.7%.
- *A network trained small from scratch is as good.* For VGG and ResNet-18 the repaired
  models and networks of the same size trained from scratch with the final runs' recipe
  (60 epochs, one seed) lie within a point of each other: at 4 times fewer parameters the
  networks from scratch trail by 0.3 to 1.0 points, at 8 times they lead on validation and
  are level or a point behind on the test set (@tab-compress-scratch). For DenseNet the network trained
  from scratch is clearly better: 83.9% against 82.4% at 4 times fewer parameters, 83.4%
  against 75.0% at 8 times (with 17% more parameters than the target, since its growth
  rate must be a whole number). Factorising a trained network saves a new training run;
  it does not find small networks that training from scratch would miss.

#let cscr = csv("tables/compress_scratch.csv")
#figure(
  text(size: 7.5pt, table(
    columns: cscr.first().len(),
    align: (left, left, right, right, right, left, right, right),
    table.header(..cscr.first().map(h => [*#h*])),
    ..cscr.slice(1).flatten(),
  )),
  caption: [Networks of the same size trained from scratch against the repaired models (the
    method with the best validation accuracy at each target), accuracy in %.],
) <tab-compress-scratch>

- *Speed depends on the hardware.* On the GPU every factorised model is slower for one
  face (0.5 to 0.75 times the speed): more, thinner layers cost more launches than they
  save. On one CPU thread the factorised models at 8 times are 2.6 to 3.6 times faster,
  below their saving in multiply-adds (3.6 to 5.6 times); DenseNet gets no faster.

= Hypotheses revisited

+ *Ranking: mostly right.* ResNet-18, VGG and DenseNet lead within a point of each other,
  and ConvNeXt stays below ResNet. CCT beats ViT by 6 points. Two parts were wrong: the
  simple CNN is not last (it ties CCT and beats ConvNeXt and ViT), and ViT lands 6 points
  below the simple CNN, not near it. After tuning the top three are unchanged, and CCT
  and the simple CNN still tie.
+ *Pretraining: wrong.* The ImageNet ResNet-18 scores 84.4%, the same network from scratch
  85.0%. With 34,000 training faces, pretraining on photos adds nothing measurable.
+ *Rare classes: right.* Disgust and fear are the weakest classes for every model. Macro-F1
  separates the models only slightly more than accuracy (15 against 13 points between the
  best and the worst).
+ *Efficiency: right.* DenseNet matches VGG and ResNet-18 with 9 and 15 times fewer
  parameters and is the slowest per epoch, even after its stem was changed for speed.
+ *Tuning gain: mostly right.* ViT gains most (6.6 points), ResNet-18 and VGG least (0.1
  and 0.4), CCT lies in between (3.6). Three misses: ConvNeXt gains no more than CCT; the
  simple CNN gains 3.0 points, so its limit was its settings, not its capacity; and
  DenseNet gains as little as ResNet-18 and VGG.
+ *Spread: partly right.* The CNNs with batch norm have the narrowest spread (IQR 6.3 to
  7.4 points; 30 to 43% of trials within a point of the best). The widest spread is not a
  transformer's but ConvNeXt's (13.9), then the simple CNN's (10.4) and CCT's (9.5). ViT's
  random trials spread moderately (6.8) but all lie far below its best (median 43.8
  against 59.9).
+ *What matters: mostly wrong.* The learning rate is the most important setting only for
  ViT and CCT. Warmup leads for four models, class weighting for ResNet-18. Augmentation
  is never second; for the transformers it explains 3 to 5%.
8. *Shape against settings: right for two of three.* The shape settings explain less than
  the training settings for VGG (44%) and DenseNet (26%), but more for ResNet-18 (71%),
  driven by the narrowest networks in its range.
9. *Gain: right on the test set.* The best shapes gain 0.2 points of test accuracy, within
  the noise. On validation, VGG's gain looked larger (1.4 points).
10. *Smaller networks: wrong.* No network with at most a quarter of the default's
  parameters came within a point of the best; the closest were 1.9 (ResNet-18) to 7.7
  points (VGG) below it.
11. *Convergence: mostly right.* After the cooldown every long run lands within 0.6 points
  of its 60-epoch counterpart. Validation did not fully level off: ResNet-18 and DenseNet
  still gained 1 to 2 points between epochs 40 and 100.
12. *No grokking yet: partly right.* No memorize run recovered, but none memorised more
  than 94% of the training faces at the constant learning rate; only the cooldown took
  them to 100%.
13. *Compressibility: right.* The weights' spectral entropy fell in all nine runs, most in
  the memorize runs.
14. *No grokking (VGG with batch norm): right.* Validation drifted down by about 2 points,
  and training broke down in eleven loss spikes.
15. *Grokking on faces: partly right.* With weight decay 0.1 the 3x start rises late and
  gradually towards the 1x level. Wrong: without weight decay it rises too and then falls
  back, and with 0.01 the rise is not later, only smaller.
16. *VGG without batch norm: wrong in size.* The late rise is 9 points, not 5, although
  NC1 was already low; a late fall of NC1 is not needed for a late rise.
17. *Output-based low rank: right at 99%, wrong at 95%.* At 99% of the output variance no
  model lost accuracy (2.4, 1.8 and 1.2 times fewer parameters). At 95% the guess failed
  both ways: VGG collapsed to 51.8%, ResNet-18 lost nothing at 4.2 times fewer.
18. *Tucker-2: right without fine-tuning.* At equal size it was always worse than
  output-based low rank (VGG at about 6 M parameters: 82.0% against 86.0%). With ranks
  per layer and repair the order turns on ResNet-18 at 8 times (test 84.9% against 83.8%).
19. *Spatial split: wrong.* It saved 2 to 3 times without loss, more than Tucker-2.
20. *int8: right for VGG and ResNet-18.* 4 times smaller, no loss, 2.6 to 2.7 times faster
  for 256 faces, and for one face more than guessed for VGG (3.5 times). DenseNet lost 25
  points.
21. *Speed: right.* No factorised model was faster for one face on the GPU, and none sped up
  on the CPU as much as its multiply-adds fell.
22. *Repair: partly right.* ResNet-18 stayed within a point with every method at every
  target, VGG only with low rank. At 8 times the guess was reversed on ResNet-18: Tucker-2
  and the spatial split stayed within a point, low rank did not (1.2 points below on
  validation).
23. *The fair baseline: mostly right.* For VGG and ResNet-18 the networks trained from
  scratch come within a point of the repaired ones (0.3 to 1.0 points behind at 4 times,
  ahead on validation at 8 times); for DenseNet they are clearly better (1.5 and 8.4
  points on the test set).
24. *Both together: right for VGG and ResNet-18* (another 4 times in size, at most 0.3
  points), wrong for DenseNet (7 points or more).
25. *DenseNet in int8: wrong.* Quantisation-aware training reached 79.7%, 5.7 points below
  the float model.

= Limitations

- *Short trials.* Trials run 25 epochs, the final runs 60. Settings that are best after 25
  epochs favour fast learners and light regularisation; for ViT and CCT the 60-epoch
  defaults beat every trial. A longer trial schedule would cost about twice the GPU time.
- *Search ranges are guesses.* Each range is centred on a default chosen before training.
  A model whose default sits far from its best looks more sensitive than one whose default
  happens to fit. The learning rate range (a tenth to ten times the default) contained the
  best value for every model.
- *Small samples.* 30 trials over eight settings, 9 of them random. fANOVA importances from
  30 trials are rough, and weak for settings that TPE stopped varying early (batch size).
  One random seed for the paired trials means one draw of 9 settings.
- *Seeds and test set.* Two seeds for the baselines, three for the final runs. The test set
  alone has about ±0.9 points of sampling error, so differences of a point or less between
  models are not resolved. The baselines did not keep their test predictions (added to the
  pipeline later), so the gain from tuning is a difference of means, not a paired test.
- *One task, one resolution.* 48×48 grayscale faces from two datasets with filtered labels
  (at least 5 of 10 FER+ annotators agree). Ambiguous faces were removed, so the test set
  is easier than faces in the wild.
- *Fixed sizes.* Stage 1 keeps the published sizes (0.8 to 11 M parameters); stage 2
  varies the shape only for the three strongest. TPE chases accuracy, so mid-sized
  networks were sampled less than large ones, and the size curve rests on few trials there.
- *One seed after stage 2.* The long runs, the grokking runs and every compressed model
  were trained once. Repeating the same repair gave results half a point apart, so
  differences below that are not resolved.
- *Timings on a shared machine.* Medians of repeated runs in PyTorch's eager mode, with
  other programs running; the same model timed at the start and end of a sweep differed
  by up to 12%. Compiled or exported models (ONNX, TensorRT) would be faster and could
  rank the methods differently, and int8 was timed on the CPU only.

#bibliography("refs.bib", style: "ieee")
