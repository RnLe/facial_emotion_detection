# Hypotheses

Written on 2026-09-30, before any model was trained. The report compares the results with
these guesses; they are not changed afterwards.

## The models, and what each step adds

Seven architectures, each a natural next step from the one before, all trained from
scratch on 48x48 grayscale faces. Sizes follow a published configuration where one exists.

| # | Model | What it adds | Configuration | Source |
|---|---|---|---|---|
| 1 | Simple CNN | Convolutions: local patterns, shared weights | 3 conv + pool stages (32, 64, 128), dense head | LeCun et al. 1998 (the idea) |
| 2 | VGG-style | Depth, small 3x3 filters, batch norm | 4 stages of two 3x3 convs (64 to 512), BN, dropout | Simonyan and Zisserman 2015; Barsoum et al. 2016 (VGG-13 on FER+) |
| 3 | ResNet-18 | Skip connections: very deep nets still train | 4 stages of 2 residual blocks (64 to 512), 3x3 stem, no max pool | He et al. 2016 (CIFAR stem) |
| 4 | DenseNet-BC | Every layer sees all earlier ones: feature reuse, few parameters | depth 100, growth 12, compression 0.5, stride-2 stem* | Huang et al. 2017 (CIFAR setting) |
| 5 | ViT | Attention instead of convolution: every patch looks at every other | 4x4 patches, 7 layers, 4 heads, width 256 (ViT-Lite-7/4) | Dosovitskiy et al. 2021; Hassani et al. 2021 |
| 6 | ConvNeXt | A CNN redesigned with transformer habits: big depthwise kernels, LayerNorm, GELU | ConvNeXt-Atto (2, 2, 6, 2 blocks; 40 to 320 channels), 2x2 stem for small images | Liu et al. 2022; timm (Atto size) |
| 7 | Hybrid (CCT) | Convolutions to make the tokens, attention to combine them | CCT-7/3x2: two conv layers, then 7 transformer layers | Hassani et al. 2021 |
| R | ResNet-18, ImageNet weights | Pretraining on 1.2 million photos | torchvision weights, input scaled to 96x96 | He et al. 2016 |

\* Changed on 2026-09-30 after timing tests, before any real training run: at full 48 px
resolution one DenseNet epoch took six times as long as ResNet-18's (see the journal).

The reference (R) is trained with the same budget but not tuned; it shows what pretraining
is worth next to everything a from-scratch architecture can do.

## Guesses before training

Test accuracy on the combined test set (FER+ and RAF-DB), with default settings:

- **H1, ranking.** ResNet-18, VGG and DenseNet end close together at the top of the
  from-scratch models (about 80 to 84%). The simple CNN is clearly last (about 70 to 74%).
  ConvNeXt lands below ResNet from scratch on this little data. The hybrid beats the pure
  ViT by several points, and the pure ViT lands near the simple CNN.
  *Why:* convolutions build in two facts about images (nearby pixels belong together, a
  pattern means the same wherever it appears). With about 34,000 training images, models
  that must learn these facts themselves (ViT) are at a disadvantage; CCT gets them back
  from its convolutional tokenizer.
- **H2, the reference.** The pretrained ResNet-18 beats every from-scratch model, by 2 to 5
  points, because it already knows edges, textures and face parts.
- **H3, rare classes.** Every model does worst on disgust and fear (the rarest classes,
  and the ones people confuse with anger and surprise). Macro-F1 separates the models
  more than accuracy does.
- **H4, efficiency.** DenseNet reaches ResNet-like accuracy with about 14 times fewer
  parameters, but trains slower per image (all those concatenations).

## Guesses about tuning

Each model gets the same tuning budget, with search ranges centred on its default.

- **H5, who gains most.** ViT and ConvNeXt gain most from tuning (several points); ResNet and
  VGG gain least (about a point); DenseNet and the hybrid are in between. The simple CNN
  gains little, because its limit is capacity, not settings.
  *Why:* batch norm and skip connections make CNNs forgiving; transformers and ConvNeXt
  (AdamW, LayerNorm, no batch norm) are known to need careful learning rate, weight decay
  and regularisation.
- **H6, spread.** Transformer-like models show the widest spread of results across trials
  (many poor settings), CNNs with batch norm the narrowest.
- **H7, what matters.** Learning rate is the most important setting for every model.
  Augmentation strength comes second for the transformers and matters little for the
  simple CNN.

## Guesses about the network's shape (stage 2)

Written on 2026-10-01, before any stage 2 trial. Stage 1 results for ResNet-18 and VGG
were known by then (tuning gains of 2 to 3 points of validation macro-F1).

Stage 2 tunes ResNet-18, VGG and DenseNet again, now with their width, depth and number of
stages in the search, starting from their stage 1 best.

- **H8, shape against settings.** For each of the three, the shape settings together
  matter less than the training settings (lower combined importance).
  *Why:* the three models reach the same accuracy at very different sizes (0.8 to 11 M
  parameters), so size is not what limits them.
- **H9, gain.** The best shape improves on the stage 1 best by less than one point of
  validation macro-F1, and after retraining the difference on the test set is within noise.
- **H10, smaller networks.** For each family, a network with at most a quarter of the
  default's parameters comes within one point of the best.
  *Why:* the default sizes come from CIFAR and ImageNet configurations; 34,000 faces of
  48x48 pixels should not need 11 M parameters.

## Guesses about long training (long runs)

Written on 2026-10-01, before any long run. Three runs each for ResNet-18, VGG and
DenseNet (tuned settings; the larger stage 2 shape; a "memorize" run without augmentation,
label smoothing or dropout and with stronger weight decay), 100 epochs with a constant
learning rate, then a 10-epoch cooldown.

- **H11, convergence.** Validation accuracy levels off within 100 epochs while accuracy on
  the clean training faces keeps rising. After the cooldown, the tuned and larger runs
  land within a point of the 60-epoch final runs: training longer does not help by itself.
- **H12, grokking.** The memorize runs learn the training faces almost completely (above
  95%) within 100 epochs, and their validation accuracy falls behind and does not recover
  within 100 epochs. The representation entropy and the weight entropy decline steadily,
  without the sudden drop that precedes grokking.
- **H13, compressibility.** The spectral entropy of the weights falls during long
  training, most in the memorize runs: the layers move towards low rank, which would make
  them easier to compress.

## Guess about grokking (grok run)

Written on 2026-10-01, after the 100-epoch long runs, before the grok run. VGG's memorize
run, after its cooldown (every training face learned, 81.5% validation accuracy), trained
on for 1,000 epochs at a tenth of the learning rate with the same weight decay.

- **H14, no grokking.** Training accuracy stays at 100% and validation accuracy rises by
  less than a point. The weight norm falls under weight decay without changing what the
  network predicts. *Why:* grokking was found where memorising comes with near-chance test
  accuracy on small training sets. Here the network already generalises (81.5%) by the
  time it has memorised 34,000 faces, so there is little left to discover late.

## Guess about grokking on the simple CNN (grok grid)

Written on 2026-10-01, after short test runs (2,000 steps) and before the full runs. The
Omnigrok recipe on faces: the simple CNN (no batch norm), 1,000 training faces, weights
started 3 times larger than usual, MSE loss, AdamW, 1e5 steps. In the test runs the 3x
start memorised the training faces by step 1,750 at 38% validation accuracy, against 52%
for the normal start.

- **H15, grokking on faces.** With weight decay 0.1, validation accuracy of the 3x start
  rises long after the training faces are memorised, towards the normal start's level,
  while the weight norm falls. With 0.01 the rise is slower and may not finish within
  1e5 steps; without weight decay validation accuracy stays low. The rise is gradual,
  as on MNIST, not the sudden jump of the algorithmic tasks.

## Guess about grokking on VGG without batch norm

Written on 2026-10-02, after a 2,000-step test and before the long run. VGG without batch
norm, alpha 2.8 (the same output spread across faces at step 0 as the CNN's alpha 3),
weight decay 0.1, otherwise the CNN recipe, 2e5 steps, bf16. In the test it memorised the
training faces by step 750 at 45.3% validation, and NC1 was already down to 0.03 by step
2,000; the CNN's 3x start kept NC1 at 0.5 to 0.9 for 10,000 steps.

- **H16, a small late rise.** Validation accuracy rises after memorising, gradually, by
  about 5 points, like the CNN's 1x start, not the 11 points of its 3x start. *Why:* in
  the CNN the big late rise came with the late fall of NC1, which here happened before
  memorising was complete, so the features are already sorted by class.

## Guesses about compression

Written on 2026-10-02, before any compressed model was evaluated. The three larger long-run
models after their cooldown (VGG 14.8 M parameters, ResNet-18 13.7 M, DenseNet 1.5 M). Each
method on its own, swept over its strength, scored on validation without fine-tuning.
Measured so far: the weights are close to full rank (keeping 95% of their squared singular
values saves 1.2 to 1.7x), while the layer outputs are low-dimensional (95% of the output
variance fits into 4 to 5.5x fewer parameters for VGG and ResNet, 2x for DenseNet).

- **H17, output-based low rank.** At 99% of the output variance VGG and ResNet lose less
  than 1 point of validation accuracy with about 2x fewer parameters and MACs; at 95% (4 to
  5.5x fewer parameters) they lose several points. DenseNet saves at most 1.3x at 99%.
- **H18, Tucker-2 from the weights.** At the same number of parameters it loses more
  accuracy than output-based low rank, because the weights are close to full rank.
- **H19, spatial split.** Also from the weights alone: under 2x fewer parameters before
  accuracy drops.
- **H20, int8.** About 4x smaller with under 0.5 points lost for all three models. On the
  CPU 1.5 to 3x faster for 256 faces, less for a single face.
- **H21, speed.** Splitting layers does not make a single face faster on the GPU (more
  layers to launch one after another); on the CPU the speedup stays below the MAC reduction.

## Guesses about the repair stage

Written on 2026-10-02 after three pilot repairs of VGG with low rank at 8x fewer parameters
(per-layer ranks: 83.9% before, 85.4% after 10 epochs, against 86.0% uncompressed) and
before the full grid, the int8 follow-ups and the networks trained from scratch.

- **H22, repair.** With per-layer ranks and 10 repair epochs every method stays within 1
  point of the uncompressed validation accuracy at 2x and 4x fewer parameters on VGG and
  ResNet. At 8x low rank stays within 1 point; Tucker-2 and the spatial split lose more,
  since they only touch the 3x3 convs and must cut those harder.
- **H23, the fair baseline.** A network of the same size trained from scratch with the
  same recipe as the stage finals is no worse than the repaired compressed one (Liu et al.
  2019 found this for pruning).
- **H24, both together.** int8 on top of a repaired factorised model gives another 4x in
  size and loses less than 0.5 points more.
- **H25, DenseNet in int8.** A few epochs of quantisation-aware training bring DenseNet's
  int8 version to within 1 point of the float model, with the concatenations quantised.
