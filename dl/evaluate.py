"""Evaluate the saved model on the held-out test set.   python -m dl.evaluate"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from dl.data_pipeline import load_cache
from dl.inference import load_model
from dl.train import predict

OUT = Path("outputs")


def main():
    torch.set_num_threads(2)
    model, ck = load_model()
    d = load_cache()
    classes = ck["classes"]
    logits = predict(model, d["X_test"], ck["mean"], ck["std"])
    y, pred = d["y_test"], logits.argmax(1).numpy()
    short = [c.replace("Tomato_Yellow_Leaf_Curl_Virus", "YLCV").replace("Spider_mites Two-spotted_spider_mite", "Spider_mites")
             .replace("Tomato_mosaic_virus", "Mosaic_virus").replace("_", " ") for c in classes]

    res = dict(test_accuracy=float(accuracy_score(y, pred)), test_macro_f1=float(f1_score(y, pred, average="macro")),
               test_weighted_f1=float(f1_score(y, pred, average="weighted")), n_test=int(len(y)),
               best_epoch=ck["epoch"],
               per_class=classification_report(y, pred, target_names=short, output_dict=True, digits=4))
    print(classification_report(y, pred, target_names=short, digits=4))
    print("test accuracy %.4f | macro-F1 %.4f" % (res["test_accuracy"], res["test_macro_f1"]))

    # robustness: accuracy when Gaussian noise (sigma in 0..1 of normalised units) is added to test inputs
    rob = {}
    X = torch.cat([torch.from_numpy(d["X_test"][i:i + 256]) for i in range(0, 1024, 256)])
    for s in (0.0, 0.1, 0.2, 0.4):
        from dl.data_pipeline import to_tensor
        xt = to_tensor(X.numpy(), ck["mean"], ck["std"])
        xt = xt + torch.randn_like(xt) * s
        with torch.no_grad():
            rob[str(s)] = float((model.eval()(xt).argmax(1).numpy() == y[:len(xt)]).mean())
    res["noise_robustness_acc_first1024"] = rob
    print("noise robustness:", rob)

    cm = confusion_matrix(y, pred)
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(short)), short, rotation=60, ha="right")
    ax.set_yticks(range(len(short)), short)
    for i in range(len(short)):
        for j in range(len(short)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7, color="white" if cmn[i, j] > .5 else "black")
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("Test confusion matrix (counts; colour = row-normalised)")
    fig.colorbar(im); fig.tight_layout(); fig.savefig(OUT / "t2_confusion.png", dpi=120); plt.close(fig)

    h = json.loads((OUT / "training_history.json").read_text())
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    e = [r["epoch"] for r in h]
    axs[0].plot(e, [r["train_loss"] for r in h], label="train (augmented)"); axs[0].plot(e, [r["val_loss"] for r in h], label="val")
    axs[0].set_xlabel("epoch"); axs[0].set_ylabel("loss"); axs[0].legend()
    axs[1].plot(e, [r["train_acc"] for r in h], label="train acc"); axs[1].plot(e, [r["val_acc"] for r in h], label="val acc")
    axs[1].plot(e, [r["val_macro_f1"] for r in h], label="val macro-F1"); axs[1].set_xlabel("epoch"); axs[1].legend()
    fig.tight_layout(); fig.savefig(OUT / "t2_curves.png", dpi=120); plt.close(fig)

    (OUT / "task2_metrics.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
