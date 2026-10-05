from .cnn import VGG, SimpleCNN
from .convnext import ConvNeXt
from .densenet import DenseNetBC
from .mirror import MirrorResNet18
from .resnet import PretrainedResNet18, ResNet18
from .transformer import CCT, ViT

# In the order of the study: each one a step on from the one before.
MODELS = {
    "cnn": SimpleCNN,
    "vgg": VGG,
    "resnet": ResNet18,
    "densenet": DenseNetBC,
    "vit": ViT,
    "convnext": ConvNeXt,
    "cct": CCT,
    "resnet_pretrained": PretrainedResNet18,
    "mirror": MirrorResNet18,  # part two: mirror-invariant ResNet-18
}

NAMES = {
    "cnn": "Simple CNN",
    "vgg": "VGG-style",
    "resnet": "ResNet-18",
    "densenet": "DenseNet-BC",
    "vit": "ViT",
    "convnext": "ConvNeXt",
    "cct": "Hybrid (CCT)",
    "resnet_pretrained": "ResNet-18, ImageNet",
    "mirror": "Mirror ResNet-18",
}


def build(name, **kwargs):
    return MODELS[name](**kwargs)
