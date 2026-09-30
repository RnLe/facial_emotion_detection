# Facial emotion recognition

Seven network architectures, from a plain CNN to a vision transformer, trained from scratch
on one cleaned dataset (FER2013 with FER+ labels, plus RAF-DB). The question: how much does
each architecture gain from hyperparameter tuning?

Work in progress. Notes on every step are in [docs/journal.md](docs/journal.md).

## Setup

```bash
uv sync
uv run python scripts/01_download.py       # needs a Kaggle token in ~/.kaggle
uv run python scripts/02_inspect.py        # sample sheets for looking at the raw data
uv run python scripts/03_build_dataset.py  # cleaning and unification -> data/processed/dataset.npz
uv run pytest
```

The datasets are not part of this repository. RAF-DB is for non-commercial research only.
