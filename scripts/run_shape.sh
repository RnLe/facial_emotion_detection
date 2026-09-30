#!/bin/sh
# Stage 2: the network's shape joins the search, for the three strongest baselines. Each
# model in its own process, as in run_all.sh.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
for m in resnet vgg densenet; do
    $PY -u scripts/06_tune.py --stage shape --models "$m" >> runs/shape.log 2>&1
done
$PY -u scripts/07_final.py --stage shape >> runs/shape_final.log 2>&1
