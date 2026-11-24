"""Tool wrappers: each runs a Task 1 / Task 2 module and returns images + text for the UI (rubric 3.2).

Every tool has the same signature  tool(image_bgr, **options) -> ToolResult  so the app can call
`TOOLS[route.intent]` uniformly - this is the 'tool use' interface.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from cv_engine import calibration as cal
from cv_engine import features as feat
from cv_engine import filtering as flt

MODEL_PATH = "models/tomato_cnn.pt"
CALIB_DIR = "data/calib"


@dataclass
class ToolResult:
    images: list = field(default_factory=list)    # [(title, RGB or gray ndarray)]
    text: str = ""
    metrics: dict = field(default_factory=dict)


def _rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr.ndim == 3 else bgr


def tool_calibrate(image, calib_paths=None, pattern=(9, 6), **_) -> ToolResult:
    paths = calib_paths or cal.glob_images(CALIB_DIR)
    res = cal.calibrate(paths, pattern)
    K = res.K
    und = cal.undistort(image, res.K, res.dist, alpha=1.0)
    text = (f"Calibrated from {len(res.used_images)} chessboard views.\n"
            f"fx={K[0,0]:.1f}, fy={K[1,1]:.1f}, cx={K[0,2]:.1f}, cy={K[1,2]:.1f}\n"
            f"distortion (k1,k2,p1,p2,k3) = {np.round(res.dist, 4).tolist()}\n"
            f"RMS reprojection error = {res.rms:.3f} px")
    return ToolResult([("Input", _rgb(image)), ("Undistorted", _rgb(und)),
                       ("Detected corners (calibration view 1)", _rgb(cal.draw_corners(cv2.imread(str(res.used_images[0])), pattern)))],
                      text, dict(K=K.round(2).tolist(), rms=res.rms))


def tool_filter(image, denoiser="median", fft_mode="lowpass", noise="salt_pepper", **_) -> ToolResult:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    noisy = flt.add_salt_pepper(gray, 0.06) if noise == "salt_pepper" else flt.add_gaussian_noise(gray, 25)
    den = {"median": flt.median_filter, "gaussian": flt.gaussian_filter, "box": flt.box_filter,
           "bilateral": flt.bilateral_filter}[denoiser](noisy)
    mag, _ = flt.sobel_gradients(gray)
    fft = flt.fft_filter(gray, 25, fft_mode)
    psnr = flt.denoise_comparison(gray, noisy)
    text = (f"Added {noise.replace('_', ' ')} noise, then denoised with {denoiser}.\n"
            f"PSNR noisy {psnr['noisy']:.1f} dB → {flt.psnr(gray, den):.1f} dB.\n"
            f"Edges: Sobel magnitude and Canny. Frequency: Gaussian {fft_mode} (σ=25).")
    return ToolResult([("Grayscale", gray), ("Noisy", noisy), (f"Denoised ({denoiser})", den),
                       ("Sobel magnitude", flt.normalise_uint8(mag)), ("Canny edges", flt.canny_edges(gray)),
                       ("log |FFT|", flt.normalise_uint8(flt.fft_spectrum(gray))),
                       (f"Gaussian {fft_mode}", flt.normalise_uint8(fft))], text,
                      dict(psnr=psnr))


def tool_features(image, image2=None, method="sift", harris=False, **_) -> ToolResult:
    big = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC) if max(image.shape[:2]) < 400 else image
    h, w = big.shape[:2]
    note = "second view supplied by user"
    if image2 is None:
        image2, H_gt = feat.synthetic_second_view(big)
        note = "no second image uploaded → simulated a second view with a known homography"
    else:
        image2, H_gt = cv2.resize(image2, (w, h)), None
    g1, g2 = (cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in (big, image2))
    r = feat.match_pair(g1, g2, method)
    n_in = int(r["inliers"].sum())
    text = (f"{method.upper()}: {len(r['kp1'])}/{len(r['kp2'])} keypoints, {len(r['matches'])} matches after ratio test (0.75), "
            f"{n_in} RANSAC inliers ({n_in / max(len(r['matches']), 1):.0%}).\n{note}.")
    metrics = dict(matches=len(r["matches"]), inliers=n_in)
    if H_gt is not None and r["H"] is not None:
        metrics["homography_error_px"] = feat.homography_error(r["H"], H_gt, w, h)
        text += f"\nHomography error vs ground truth: {metrics['homography_error_px']:.2f} px."
    imgs = [("Matches (inliers)", _rgb(feat.draw_matches(big, image2, r)))]
    if harris:
        pts, _ = feat.harris_corners(g1)
        imgs.append((f"Harris from scratch ({len(pts)} corners)", _rgb(feat.draw_keypoints(big, pts))))
    return ToolResult(imgs, text, metrics)


def tool_classify(image, explain=False, **_) -> ToolResult:
    from dl.inference import grad_cam, predict
    if not Path(MODEL_PATH).exists():
        return ToolResult(text=f"Trained model not found at {MODEL_PATH}. Run `python -m dl.train` first.")
    top, probs = predict(image, MODEL_PATH, topk=3)
    label, conf = top[0]
    healthy = label == "healthy"
    text = (f"Prediction: **{label.replace('_', ' ')}** ({conf:.1%} confidence) - "
            f"{'leaf looks healthy' if healthy else 'disease detected'}.\n"
            + "\n".join(f"{i + 1}. {n.replace('_', ' ')}: {c:.1%}" for i, (n, c) in enumerate(top)))
    if conf < 0.6:
        text += "\n⚠ Low confidence: the leaf may be outside the 10 trained tomato classes - retake the photo."
    imgs = [("Input", _rgb(image))]
    if explain:
        overlay, _, _ = grad_cam(image, MODEL_PATH)
        imgs.append(("Grad-CAM (regions driving the prediction)", _rgb(overlay)))
    return ToolResult(imgs, text, dict(top3=top))


def tool_explain(image, **kw) -> ToolResult:
    return tool_classify(image, explain=True, **kw)


TOOLS = {"calibrate": tool_calibrate, "filter": tool_filter, "features": tool_features,
         "classify": tool_classify, "explain": tool_explain}
