"""Train the leaf-disease CNN.   python -m dl.train --epochs 14

Saves the best checkpoint (by validation macro-F1) to models/tomato_cnn.pt together with everything
the app needs for inference (class names, normalisation statistics, input size, architecture).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from dl.data_pipeline import augment_batch, load_cache, to_tensor
from dl.model import build_model, n_params

MODELS = Path("models")


def class_weights(y: np.ndarray, k: int, power=0.5) -> torch.Tensor:
    """(N / (K * n_c)) ** power. power=1 is full inverse-frequency; 0.5 tempers it, because
    full weighting makes the rarest class (373 images) dominate the gradient and add noise."""
    counts = np.bincount(y, minlength=k).astype(np.float64)
    return torch.tensor((len(y) / (k * counts)) ** power, dtype=torch.float32)


@torch.no_grad()
def predict(model, X, mean, std, bs=256):
    model.eval()
    logits = [model(to_tensor(X[i:i + bs], mean, std)) for i in range(0, len(X), bs)]
    return torch.cat(logits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--arch", default="plantcnn")
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="models/tomato_cnn.pt")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(2)
    d = load_cache()
    classes, mean, std = [str(c) for c in d["classes"]], d["mean"], d["std"]
    Xtr, ytr, Xva, yva = d["X_train"], d["y_train"], d["X_val"], d["y_val"]
    k = len(classes)

    model = build_model(k, a.arch, a.pretrained)
    print(f"{a.arch}: {n_params(model):,} parameters | train {len(Xtr)} val {len(Xva)}", flush=True)
    crit = nn.CrossEntropyLoss(weight=class_weights(ytr, k), label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = int(np.ceil(len(Xtr) / a.bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, epochs=a.epochs, steps_per_epoch=steps)

    hist, best = [], -1.0
    Path(a.out).parent.mkdir(exist_ok=True)
    for ep in range(a.epochs):
        model.train()
        t0, perm, tl, correct = time.time(), np.random.permutation(len(Xtr)), 0.0, 0
        for i in range(0, len(perm), a.bs):
            idx = np.sort(perm[i:i + a.bs])
            x = augment_batch(to_tensor(Xtr[idx], mean, std))
            y = torch.from_numpy(ytr[idx])
            out = model(x)
            loss = crit(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tl += loss.item() * len(idx)
            correct += (out.argmax(1) == y).sum().item()
        logits = predict(model, Xva, mean, std)
        vloss = nn.functional.cross_entropy(logits, torch.from_numpy(yva)).item()
        pred = logits.argmax(1).numpy()
        vacc, vf1 = float((pred == yva).mean()), float(f1_score(yva, pred, average="macro"))
        hist.append(dict(epoch=ep + 1, train_loss=tl / len(Xtr), train_acc=correct / len(Xtr),
                         val_loss=vloss, val_acc=vacc, val_macro_f1=vf1, seconds=time.time() - t0))
        print(json.dumps({k_: round(v, 4) for k_, v in hist[-1].items()}), flush=True)
        if vf1 > best:
            best = vf1
            torch.save(dict(state_dict=model.state_dict(), classes=classes, mean=mean, std=std,
                            img_size=int(d["X_train"].shape[1]), arch=a.arch, epoch=ep + 1, val_macro_f1=vf1,
                            hparams=vars(a)), a.out)
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/training_history.json").write_text(json.dumps(hist, indent=2))
    print("best val macro-F1: %.4f" % best)


if __name__ == "__main__":
    main()
