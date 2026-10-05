"""The published models used as references, loaded from their released weights.

Each loader returns the model and its input spec: size, normalisation, and the order of
its seven outputs. Both RAF-DB references were trained with torchvision's ImageFolder on
RAF-DB's label folders, so their outputs follow RAF-DB's label order; `to_ours` reorders
logits into CLASSES order.

- FMAE (Ning et al. 2024): ViT-L/16 pretrained as a masked autoencoder on 9 million faces,
  fine-tuned on RAF-DB. Hugging Face forever208/FMAE-IAT. Weights CC BY-NC 4.0.
- POSTER++ (Mao et al. 2023): IR-50 face backbone with cross-attention from facial
  landmark features. github.com/Talented-Q/POSTER_V2, MIT. Model code vendored in poster/.
- CLIP ViT-B/16 (Radford et al. 2021), OpenAI weights through open_clip.
"""
import pickle
import types
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn

from ..data import RAF_TO_CLASS

WEIGHTS = Path("runs/references/weights")
IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
CLIP = ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
# output k is RAF-DB label k + 1 (folders 1..7 in sorted order)
RAF_ORDER = [RAF_TO_CLASS[k + 1] for k in range(7)]


@dataclass
class Spec:
    size: int
    mean: tuple
    std: tuple
    order: list | None  # output k is our class order[k]; None when already in CLASSES order


def to_ours(logits, order):
    """Reorder a model's outputs into CLASSES order."""
    if order is None:
        return logits
    out = torch.empty_like(logits)
    out[:, order] = logits
    return out


class _Stub:
    """Stands in for training-script classes pickled into a checkpoint (meters, loggers)."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (AttributeError, ModuleNotFoundError):
            return _Stub


_lenient = types.SimpleNamespace(Unpickler=_Unpickler, load=pickle.load, __name__="lenient_pickle")


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False, pickle_module=_lenient)


def fmae(checkpoint="fmae_rafdb_model.pth", drop_path=0.1):
    """ViT-L/16 with mean pooling over the patch tokens and a norm before the head, as in
    FMAE's fine-tuning (models_vit.py with global_pool)."""
    from timm.models.vision_transformer import VisionTransformer

    model = VisionTransformer(
        img_size=224, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=7, global_pool="avg", drop_path_rate=drop_path,
    )
    state = load_checkpoint(WEIGHTS / checkpoint)
    state = state.get("model", state)
    # FMAE's classifier is AU_head; timm's own 1000-class head stays unused in its checkpoints
    if "AU_head.weight" in state:
        state = {k: v for k, v in state.items() if not k.startswith("head.")}
        state["head.weight"], state["head.bias"] = state.pop("AU_head.weight"), state.pop("AU_head.bias")
    model.load_state_dict(state, strict=True)
    return model, Spec(224, *IMAGENET, RAF_ORDER)


def poster(checkpoint="poster_rafdb.pth"):
    from .poster import pyramid_trans_expr2

    model = pyramid_trans_expr2(img_size=224, num_classes=7)
    state = load_checkpoint(WEIGHTS / checkpoint)
    state = {k.removeprefix("module."): v for k, v in state["state_dict"].items()}
    model.load_state_dict(state, strict=True)
    return model, Spec(224, *IMAGENET, RAF_ORDER)


class ClipClassifier(nn.Module):
    """CLIP's image tower with a linear head. The head starts from the text embeddings
    of the class prompts (zero-shot) or from a linear probe."""

    def __init__(self, visual, head_weight, scale=100.0):
        super().__init__()
        self.visual = visual
        self.head = nn.Linear(head_weight.shape[1], head_weight.shape[0], bias=True)
        with torch.no_grad():
            self.head.weight.copy_(head_weight * scale)
            self.head.bias.zero_()

    def features(self, x):
        f = self.visual(x)
        return f / f.norm(dim=-1, keepdim=True)

    def forward(self, x):
        return self.head(self.features(x))


ADJECTIVES = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]  # CLASSES order
TEMPLATES = [
    "a photo of a {} face.",
    "a photo of a person looking {}.",
    "a close-up photo of a {} person.",
    "a face with a {} expression.",
    "a grayscale photo of a {} face.",
]


def clip():
    """CLIP ViT-B/16 as a zero-shot classifier: the head holds the mean text embedding
    of each class over a few prompt templates."""
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    with torch.no_grad():
        rows = []
        for adj in ADJECTIVES:
            t = model.encode_text(tokenizer([tpl.format(adj) for tpl in TEMPLATES]))
            t = t / t.norm(dim=-1, keepdim=True)
            t = t.mean(0)
            rows.append(t / t.norm())
        head = torch.stack(rows)
    return ClipClassifier(model.visual, head), Spec(224, *CLIP, None)


LOADERS = {"fmae": fmae, "poster": poster, "clip": clip}


def load(name, **kwargs):
    return LOADERS[name](**kwargs)
