#!/bin/sh
# The whole training schedule, one process per step, so that a crash stops only its own
# model. Every step skips what is already done, so this can simply be run again.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
$PY -u scripts/05_baselines.py >> runs/baseline.log 2>&1
for m in resnet vgg densenet cnn cct convnext vit; do  # best baselines first
    $PY -u scripts/06_tune.py --models "$m" >> runs/tune.log 2>&1
done
$PY -u scripts/07_final.py >> runs/final.log 2>&1
