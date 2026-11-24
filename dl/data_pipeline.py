"""Task 2 data pipeline: ingestion -> cleaning -> split -> transformation -> augmentation.

Run once to build the cache:   python -m dl.data_pipeline
Every step is a small function so it can be unit-tested and reported on separately.
"""
from __future__ import annotations

import hashlib
import json
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

DATA_ROOT = Path("data/pv/raw/color")
CACHE = Path("data/cache")
IMG_SIZE = 128
CROP = "Tomato"          # crop subset used (10 classes); set to None for all 38 classes
SEED = 42


# ------------------------------------------------------------------ 1. ingestion
def ingest(root=DATA_ROOT, crop=CROP) -> pd.DataFrame:
    """Scan class folders (folder name = label) into a table of path + label."""
    rows = []
    for d in sorted(Path(root).iterdir()):
        if d.is_dir() and (crop is None or d.name.startswith(crop + "___")):
            rows += [(str(p), d.name.split("___", 1)[1].strip()) for p in d.iterdir()
                     if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]      # case-insensitive extensions
    df = pd.DataFrame(rows, columns=["path", "label"])
    return df


# ------------------------------------------------------------------ 2. cleaning
def _inspect(path: str) -> dict:
    """Per-image quality statistics (runs in worker processes)."""
    raw = Path(path).read_bytes()
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return dict(path=path, ok=False)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (17, 16), interpolation=cv2.INTER_AREA)
    dhash = (small[:, 1:] > small[:, :-1]).astype(np.uint8).ravel()          # 256-bit difference hash
    return dict(path=path, ok=True, md5=hashlib.md5(raw).hexdigest(),
                dhash=np.packbits(dhash).tobytes().hex(),
                blur=float(cv2.Laplacian(gray, cv2.CV_64F).var()),           # focus measure
                mean=float(gray.mean()), std=float(gray.std()))


def clean(df: pd.DataFrame, workers=2):
    """Drop unreadable, duplicate, blurred and degenerate images. Returns (clean_df, report).

    Thresholds are data-driven (robust z-score on log Laplacian variance) so they adapt to the
    dataset instead of being magic numbers.
    """
    with Pool(workers) as p:
        stats = pd.DataFrame(p.map(_inspect, df.path.tolist(), chunksize=64))
    d = df.merge(stats, on="path")
    report = {"n_raw": len(d)}

    unreadable = ~d.ok.fillna(False).astype(bool)
    report["unreadable"] = int(unreadable.sum())
    d = d[~unreadable]

    dup_md5 = d.duplicated("md5", keep="first")                      # byte-identical files
    report["exact_duplicates"] = int(dup_md5.sum())
    d = d[~dup_md5]
    dup_hash = d.duplicated(["dhash", "label"], keep="first")        # visually identical within a class
    report["near_duplicates"] = int(dup_hash.sum())
    d = d[~dup_hash]

    lb = np.log1p(d.blur.values)
    med = np.median(lb)
    mad = 1.4826 * np.median(np.abs(lb - med))
    blur_cut = med - 3.5 * mad
    blurred = lb < blur_cut
    report["blur_threshold_laplacian_var"] = float(np.expm1(blur_cut))
    report["blurred"] = int(blurred.sum())
    d = d[~blurred]

    degenerate = (d["mean"] < 15) | (d["mean"] > 240) | (d["std"] < 8)   # nearly black/white/flat
    report["degenerate"] = int(degenerate.sum())
    d = d[~degenerate]

    report["n_clean"] = len(d)
    report["class_counts"] = d.label.value_counts().to_dict()
    return d[["path", "label"]].reset_index(drop=True), report


# ------------------------------------------------------------------ 3. split
def split(df: pd.DataFrame, val=0.15, test=0.15, seed=SEED):
    """Stratified train/val/test split (keeps class proportions in every subset)."""
    train, rest = train_test_split(df, test_size=val + test, stratify=df.label, random_state=seed)
    v, t = train_test_split(rest, test_size=test / (val + test), stratify=rest.label, random_state=seed)
    return train.reset_index(drop=True), v.reset_index(drop=True), t.reset_index(drop=True)


# ------------------------------------------------------------------ 4. transformation
def _load_resized(path: str, size=IMG_SIZE) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)   # INTER_AREA: anti-aliased downscale
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def build_arrays(df: pd.DataFrame, classes: list[str], workers=2):
    X = np.empty((len(df), IMG_SIZE, IMG_SIZE, 3), np.uint8)      # preallocate: avoids a 2x peak from np.stack
    with Pool(workers) as p:
        for i, arr in enumerate(p.imap(_load_resized, df.path.tolist(), chunksize=64)):
            X[i] = arr
    y = df.label.map({c: i for i, c in enumerate(classes)}).values.astype(np.int64)
    return X, y


def channel_stats(X: np.ndarray, chunk=1000):
    """Per-channel mean/std in [0,1], accumulated in chunks (a float64 copy of X would need ~5 GB)."""
    n = s1 = s2 = 0.0
    for i in range(0, len(X), chunk):
        c = X[i:i + chunk].astype(np.float64) / 255.0
        s1 = s1 + c.sum(axis=(0, 1, 2))
        s2 = s2 + (c ** 2).sum(axis=(0, 1, 2))
        n += c.shape[0] * c.shape[1] * c.shape[2]
    mean = s1 / n
    return mean, np.sqrt(s2 / n - mean ** 2)


def build_cache():
    CACHE.mkdir(parents=True, exist_ok=True)
    df = ingest()
    df, report = clean(df)
    classes = sorted(df.label.unique())
    tr, va, te = split(df)
    out = {}
    for name, part in (("train", tr), ("val", va), ("test", te)):
        X, y = build_arrays(part, classes)
        out[f"X_{name}"], out[f"y_{name}"] = X, y
        out[f"paths_{name}"] = np.array(part.path.tolist())
    mean, std = channel_stats(out["X_train"])                      # statistics from TRAIN only (no leakage)
    np.savez(CACHE / "tomato_128.npz", classes=np.array(classes), mean=mean, std=std, **out)
    report.update(classes=classes, n_train=len(tr), n_val=len(va), n_test=len(te),
                  mean=mean.round(4).tolist(), std=std.round(4).tolist(), img_size=IMG_SIZE)
    (CACHE / "cleaning_report.json").write_text(json.dumps(report, indent=2))
    return report


def load_cache():
    d = np.load(CACHE / "tomato_128.npz", allow_pickle=False)
    return {k: d[k] for k in d.files}


# ------------------------------------------------------------------ 5. augmentation (batch, on tensors)
def to_tensor(X: np.ndarray, mean, std) -> torch.Tensor:
    t = torch.from_numpy(X).permute(0, 3, 1, 2).float() / 255.0
    return (t - torch.tensor(mean).view(1, 3, 1, 1).float()) / torch.tensor(std).view(1, 3, 1, 1).float()


def augment_batch(x: torch.Tensor, rot_deg=30, scale=(0.85, 1.15), noise_std=0.05) -> torch.Tensor:
    """Random flips, rotation, scale, brightness/contrast jitter and Gaussian noise (train only).

    Implemented with one affine_grid/grid_sample call for the whole batch, which is far faster
    on CPU than per-image PIL transforms. Leaf orientation is arbitrary, so flips/rotations
    keep the label; colour jitter is mild because colour is diagnostic for disease.
    """
    n = x.size(0)
    ang = (torch.rand(n) * 2 - 1) * rot_deg * np.pi / 180
    sc = torch.empty(n).uniform_(*scale)
    flip = (torch.rand(n) < 0.5).float() * 2 - 1
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos * flip, -sin, torch.zeros(n)], 1),
                         torch.stack([sin * flip, cos, torch.zeros(n)], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="reflection", align_corners=False)
    vflip = torch.rand(n) < 0.5
    x[vflip] = x[vflip].flip(2)
    x = x * torch.empty(n, 1, 1, 1).uniform_(0.85, 1.15) + torch.empty(n, 1, 1, 1).uniform_(-0.15, 0.15)
    return x + torch.randn_like(x) * noise_std


if __name__ == "__main__":
    rep = build_cache()
    print(json.dumps({k: v for k, v in rep.items() if k != "class_counts"}, indent=2))
    print(rep["class_counts"])
