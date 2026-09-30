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
