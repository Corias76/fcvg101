"""Inference helpers shared by evaluation and the app: load checkpoint, preprocess, predict, Grad-CAM."""
from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from dl.data_pipeline import to_tensor
from dl.model import build_model


@lru_cache(maxsize=1)
def load_model(path="models/tomato_cnn.pt"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(len(ck["classes"]), ck["arch"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


def preprocess(bgr: np.ndarray, ck) -> tuple[torch.Tensor, np.ndarray]:
    """BGR image -> normalised 1x3xHxW tensor, using exactly the training-time resize + statistics."""
    s = ck["img_size"]
    rgb = cv2.cvtColor(cv2.resize(bgr, (s, s), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    return to_tensor(rgb[None], ck["mean"], ck["std"]), rgb


def predict(bgr, path="models/tomato_cnn.pt", topk=3):
    model, ck = load_model(path)
    x, _ = preprocess(bgr, ck)
    with torch.no_grad():
        p = F.softmax(model(x), 1)[0]
    conf, idx = p.topk(topk)
    return [(ck["classes"][i], float(c)) for c, i in zip(conf, idx)], p.numpy()


def grad_cam(bgr, path="models/tomato_cnn.pt", target=None):
    """Grad-CAM (Selvaraju et al.): weight the last conv feature maps by the mean gradient of the
    class score, ReLU, normalise, upsample. Shows which leaf regions drove the prediction."""
    model, ck = load_model(path)
    x, rgb = preprocess(bgr, ck)
    acts, grads = [], []
    h1 = model.cam_layer.register_forward_hook(lambda m, i, o: acts.append(o))
    h2 = model.cam_layer.register_full_backward_hook(lambda m, gi, go: grads.append(go[0]))
    try:
        model.zero_grad()
        out = model(x)
        t = int(out.argmax(1)) if target is None else target
        out[0, t].backward()
    finally:
        h1.remove()
        h2.remove()
    w = grads[0].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * acts[0]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=rgb.shape[:2], mode="bilinear", align_corners=False)[0, 0].detach().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), 0.55, heat, 0.45, 0)
    return overlay, cam, ck["classes"][t]
