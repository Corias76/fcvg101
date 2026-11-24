"""Extra figures for the report: Grad-CAM examples and sample/augmentation grids."""
import cv2, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, torch
from dl.data_pipeline import augment_batch, load_cache, to_tensor
from dl.inference import grad_cam, predict

d = load_cache(); classes = [str(c) for c in d["classes"]]
rng = np.random.default_rng(1)
# --- Grad-CAM on test images of 4 classes
pick = [np.flatnonzero(d["y_test"] == classes.index(c))[0] for c in ("Late_blight", "Leaf_Mold", "Tomato_mosaic_virus", "healthy")]
fig, axs = plt.subplots(2, 4, figsize=(12, 6.2))
for j, i in enumerate(pick):
    bgr = cv2.imread(str(d["paths_test"][i]))
    ov, _, lab = grad_cam(bgr)
    top, _ = predict(bgr)
    axs[0, j].imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)); axs[0, j].set_title("true: " + classes[d["y_test"][i]].replace("_", " "), fontsize=9)
    axs[1, j].imshow(cv2.cvtColor(ov, cv2.COLOR_BGR2RGB)); axs[1, j].set_title(f"pred: {top[0][0].replace('_',' ')} ({top[0][1]:.0%})", fontsize=9)
    axs[0, j].axis("off"); axs[1, j].axis("off")
fig.tight_layout(); fig.savefig("outputs/t2_gradcam.png", dpi=100); plt.close(fig)
# --- augmentation examples
x = to_tensor(d["X_train"][:1], d["mean"], d["std"]).repeat(8, 1, 1, 1)
xa = augment_batch(x)
den = lambda t: np.clip(t.permute(1, 2, 0).numpy() * d["std"] + d["mean"], 0, 1)
fig, axs = plt.subplots(1, 5, figsize=(12, 2.8))
axs[0].imshow(den(x[0])); axs[0].set_title("original (128px)")
for k in range(4):
    axs[k + 1].imshow(den(xa[k])); axs[k + 1].set_title(f"augmented {k+1}")
for a in axs: a.axis("off")
fig.tight_layout(); fig.savefig("outputs/t2_augment.png", dpi=100); plt.close(fig)
# --- class distribution
import json
rep = json.load(open("data/cache/cleaning_report.json"))
fig, ax = plt.subplots(figsize=(8, 3.4))
names = [k.replace("Tomato_Yellow_Leaf_Curl_Virus", "YLCV").replace("Spider_mites Two-spotted_spider_mite", "Spider mites").replace("Tomato_mosaic_virus", "Mosaic virus").replace("_", " ") for k in rep["class_counts"]]
ax.barh(names[::-1], list(rep["class_counts"].values())[::-1]); ax.set_xlabel("images after cleaning")
fig.tight_layout(); fig.savefig("outputs/t2_classdist.png", dpi=100); plt.close(fig)
print("ok")
