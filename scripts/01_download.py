"""Download the raw data into data/raw.

FER2013: the original challenge CSV (48x48 grayscale, with its Training /
PublicTest / PrivateTest split). FER+: Microsoft's relabelling of the same
images, ten votes per image. RAF-DB: the basic-emotion part (aligned faces).
Needs a Kaggle token (~/.kaggle) and git.
"""
import subprocess
from pathlib import Path

RAW = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)


def kaggle(ref, dest):
    if (RAW / dest).exists():
        print(f"{dest}: already there")
        return
    subprocess.run(["kaggle", "datasets", "download", "-d", ref, "-p", str(RAW / dest), "--unzip"], check=True)


kaggle("deadskull7/fer2013", "fer2013")
kaggle("shuvoalok/raf-db-dataset", "rafdb")

if not (RAW / "FERPlus").exists():
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/microsoft/FERPlus", str(RAW / "FERPlus")], check=True)
