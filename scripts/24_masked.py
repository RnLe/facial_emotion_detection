"""Masked-face pretraining on our own faces, then the usual training (part two, phase 5,
H43: is FMAE's part reading a matter of its objective or of its 9 million faces?).

pretrain: the ResNet-18 winner's network learns to reconstruct masked patches of our
    training faces (fer/pretrain.py), 150 epochs, no labels.
finetune: the winner's recipe from the pretrained weights, 3 seeds.
"""
import argparse
import json

import torch

from fer.pretrain import masked_pretrain
from fer.runs import RUNS, add_to_plan, is_done, record_path
from fer.train import GPUData, run

ENCODER = RUNS / "masked" / "resnet" / "encoder.pt"


def base_config(seed):
    cfg = json.loads(record_path("final", "resnet", seed).read_text())["config"]
    return {k: v for k, v in cfg.items() if k not in ("stage", "iteration")}


def pretrain():
    if is_done("masked", "resnet", "pretrain"):
        return
    cfg = {"model": "resnet", "epochs": 150, "batch_size": 256, "lr": 1e-3, "weight_decay": 0.05, "warmup_epochs": 10,
           "mask_ratio": 0.6, "augment": 1.0, "seed": 0}
    add_to_plan([{"stage": "masked", "model": "resnet", "iteration": "pretrain", "epochs": cfg["epochs"]}])
    weights = masked_pretrain(cfg, GPUData(), record=record_path("masked", "resnet", "pretrain"))
    ENCODER.parent.mkdir(parents=True, exist_ok=True)
    torch.save(weights, ENCODER)


def finetune(seeds):
    data = GPUData()
    for seed in seeds:
        name = f"finetune_{seed}"
        rec = record_path("masked", "resnet", name)
        if not is_done("masked", "resnet", name):
            add_to_plan([{"stage": "masked", "model": "resnet", "iteration": name, "epochs": 60}])
            run({**base_config(seed), "seed": seed, "init": str(ENCODER), "stage": "masked", "iteration": name}, data=data, test=True, record=rec)
        r = json.loads(rec.read_text())
        print(name, f"val {r['val']['accuracy']:.4f}  F1 {r['val']['macro_f1']:.4f}  test {r['test']['accuracy']:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["pretrain", "finetune"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    pretrain() if a.step == "pretrain" else finetune(a.seeds)
