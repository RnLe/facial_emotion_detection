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
  close it: ResNet-18, VGG and DenseNet stay on top at about 85% test accuracy.
]

= Question

Facial emotion recognition sorts a face into one of seven classes: angry, disgust, fear,
happy, neutral, sad, surprise. Many architectures have been applied to it. Papers usually
report each model after its own, often undisclosed, tuning. That hides two things: how
good an architecture is with sensible default settings, and how much of its final score
is owed to tuning.

This study trains seven architectures with one pipeline on one dataset, first with
defaults chosen before any training, then after the same tuning budget for each. The
guesses made beforehand are listed in @sec-models and checked at the end.

*Why a rebuild.* An earlier course project (2024) compared a DenseNet with a CNN on
FER2013. A later review of its code found that neither of its two numbers measured test
performance: a swapped return value trained the DenseNet on the test folder and scored it
there, the hyperparameter search therefore optimised a training loss, and the CNN was
scored on unscaled images. This study starts from scratch, with one split, one data
object, one evaluation path for every model, and a test set used once.

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
  starts with one stride-2 convolution; its blocks work at 24, 12 and 6 pixels.
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
- *Fixed sizes.* The architectures keep their published sizes (0.8 to 11 M parameters).
  Stage 2 varies the shape for the three strongest.

#bibliography("refs.bib", style: "ieee")
