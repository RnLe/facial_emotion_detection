#set document(title: "Facial emotion recognition: seven architectures and the value of tuning", author: "Rene-Marcel Lehner")
#set page(paper: "a4", margin: (x: 2.3cm, y: 2.4cm), numbering: "1")
#set text(font: "New Computer Modern", size: 10.5pt, lang: "en")
#set par(justify: true, leading: 0.62em)
#set heading(numbering: "1.1")
#show heading: set block(above: 1.3em, below: 0.7em)
#set figure(gap: 0.8em)
#show figure.caption: set text(size: 9pt)
#set table(stroke: 0.4pt + luma(180), inset: 5pt)

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
  architectures depend on tuning, and by how much?
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

_To be written when the baseline runs are complete._

= Tuning and sensitivity <sec-tuning>

Each model gets 40 Optuna @akiba2019 trials with the TPE sampler @bergstra2011 (10 random
trials first) and successive halving @jamieson2016, which stops weak trials after 6 and 18
of 60 epochs. The search ranges are educated guesses centred on each model's defaults: the
learning rate from a tenth to ten times the default, weight decay from 0.0005 to 0.5,
batch size 128, 256 or 512, label smoothing up to 0.2, augmentation strength up to 1.5,
three class weightings, up to 10 warmup epochs, and dropout or stochastic depth. The
default settings are the first trial of every study. The best settings are retrained with
three seeds and scored once on the test set.

_Results to be written when the tuning is complete._

= Hypotheses revisited

_To be written._

= Limitations

_To be written._

#bibliography("refs.bib", style: "ieee")
