"""Task 1 demo: runs the whole classical pipeline and saves figures + numbers to outputs/.

Usage:  python run_task1.py [--leaf path/to/leaf.jpg]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cv_engine import calibration as cal
from cv_engine import features as feat
from cv_engine import filtering as flt

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
DEFAULT_LEAF = next(Path("data/pv/raw/color/Tomato___Late_blight").glob("*.JPG"), None)


def rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr.ndim == 3 else bgr


def grid(images, titles, path, cols=3, size=4.2):
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * size, rows * size))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, im, t in zip(np.atleast_1d(axes).ravel(), images, titles):
        ax.imshow(im, cmap="gray" if im.ndim == 2 else None)
        ax.set_title(t, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------ 1.1 calibration
def run_calibration():
    paths = cal.glob_images("data/calib")
    res = cal.calibrate(paths, pattern=(9, 6), square_size=1.0)
    cal.save_calibration(res, "models/calibration.npz")
    print("Calibration images used:", len(res.used_images), "/", len(paths))
    print("K =\n", np.round(res.K, 2))
    print("dist (k1,k2,p1,p2,k3) =", np.round(res.dist, 5))
    print("RMS reprojection error (px): %.3f" % res.rms)

    # pinhole model check: explicit projection vs cv2.projectPoints on view 0
    obj, imgp, _, _ = cal.find_corners(res.used_images[:1])
    R, _ = cv2.Rodrigues(res.rvecs[0])
    mine = cal.project_pinhole(res.K, R, res.tvecs[0].ravel(), obj[0])
    ref, _ = cv2.projectPoints(obj[0], res.rvecs[0], res.tvecs[0], res.K, np.zeros(5))
    pinhole_diff = float(np.abs(mine - ref.reshape(-1, 2)).max())
    print("max |explicit pinhole - cv2.projectPoints| (px, no distortion): %.2e" % pinhole_diff)

    img = cv2.imread(str(res.used_images[0]))
    und = cal.undistort(img, res.K, res.dist, alpha=1.0)
    grid([rgb(cal.draw_corners(img)), rgb(img), rgb(und)],
         ["Detected corners", "Original", "Undistorted (alpha=1)"], OUT / "t1_calibration.png")

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(range(len(res.per_image_error)), res.per_image_error)
    ax.axhline(res.rms, color="r", ls="--", label=f"overall RMS {res.rms:.3f}px")
    ax.set_xlabel("calibration image"); ax.set_ylabel("mean reprojection error (px)"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "t1_reproj_error.png", dpi=110); plt.close(fig)
    return dict(K=res.K.round(3).tolist(), dist=res.dist.round(6).tolist(), rms=res.rms,
                n_images=len(res.used_images), pinhole_vs_cv2_px=pinhole_diff)


# ------------------------------------------------------------------ 1.2 filtering
def run_filtering(leaf):
    gray = cv2.cvtColor(leaf, cv2.COLOR_BGR2GRAY)

    # from-scratch correlation vs cv2.filter2D
    k = np.ones((5, 5), np.float32) / 25
    diff = float(np.abs(flt.convolve2d_naive(gray, k) - cv2.filter2D(gray.astype(np.float32), -1, k,
                 borderType=cv2.BORDER_CONSTANT)).max())
    print("max |naive conv - cv2.filter2D|: %.2e" % diff)

    g_noisy = flt.add_gaussian_noise(gray, 25)
    sp_noisy = flt.add_salt_pepper(gray, 0.06)
    psnr_g = flt.denoise_comparison(gray, g_noisy)
    psnr_sp = flt.denoise_comparison(gray, sp_noisy)
    print("PSNR (dB) Gaussian noise:", {k: round(v, 2) for k, v in psnr_g.items()})
    print("PSNR (dB) Salt&pepper   :", {k: round(v, 2) for k, v in psnr_sp.items()})
    grid([g_noisy, flt.box_filter(g_noisy), flt.gaussian_filter(g_noisy), flt.median_filter(g_noisy),
          sp_noisy, flt.box_filter(sp_noisy), flt.gaussian_filter(sp_noisy), flt.median_filter(sp_noisy)],
         ["Gaussian noise", "box", "gaussian", "median", "Salt&pepper", "box", "gaussian", "median"],
         OUT / "t1_denoise.png", cols=4, size=3.2)

    mag, _ = flt.sobel_gradients(gray)
    canny = flt.canny_edges(gray)
    grid([gray, flt.normalise_uint8(mag), canny], ["Grayscale", "Sobel magnitude", "Canny"], OUT / "t1_edges.png")

    low = flt.fft_filter(gray, 25, "lowpass")
    high = flt.fft_filter(gray, 25, "highpass")
    grid([gray, flt.normalise_uint8(flt.fft_spectrum(gray)), flt.normalise_uint8(low), flt.normalise_uint8(high)],
         ["Grayscale", "log |FFT| spectrum", "Gaussian low-pass (σ=25)", "Gaussian high-pass (σ=25)"],
         OUT / "t1_fft.png", cols=4, size=3.2)
    return dict(naive_conv_max_diff=diff, psnr_gaussian_noise=psnr_g, psnr_salt_pepper=psnr_sp)


# ------------------------------------------------------------------ 1.3 features
def run_features(leaf):
    big = cv2.resize(leaf, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)   # more texture for keypoints
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    warped, H_gt = feat.synthetic_second_view(big)
    gray2 = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    pts_s, _ = feat.harris_corners(gray)
    R_cv = cv2.cornerHarris(np.float32(gray), 2, 3, 0.05)
    pts_cv = np.argwhere(R_cv > 0.01 * R_cv.max())[:, ::-1]
    print("Harris from scratch: %d corners | cv2.cornerHarris above 1%% max: %d px" % (len(pts_s), len(pts_cv)))
    grid([rgb(feat.draw_keypoints(big, pts_s))], ["Harris (from scratch)"], OUT / "t1_harris.png", cols=1, size=6)

    results = {}
    for method in ("sift", "orb"):
        r = feat.match_pair(gray, gray2, method)
        n_m, n_in = len(r["matches"]), int(r["inliers"].sum())
        err = feat.homography_error(r["H"], H_gt, w, h)
        results[method] = dict(kp1=len(r["kp1"]), kp2=len(r["kp2"]), ratio_matches=n_m, ransac_inliers=n_in,
                               inlier_rate=n_in / max(n_m, 1), homography_error_px=err)
        print(method.upper(), results[method])
        cv2.imwrite(str(OUT / f"t1_matches_{method}.jpg"), feat.draw_matches(big, warped, r))

    # effect of ratio threshold (justifies 0.75)
    sweep = {}
    for ratio in (0.6, 0.7, 0.75, 0.8, 0.9, 1.0):
        r = feat.match_pair(gray, gray2, "sift", ratio=ratio)
        sweep[ratio] = (len(r["matches"]), int(r["inliers"].sum()))
    print("SIFT ratio sweep (matches, inliers):", sweep)
    results["sift_ratio_sweep"] = {str(k): v for k, v in sweep.items()}
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leaf", default=str(DEFAULT_LEAF))
    args = ap.parse_args()
    leaf = cv2.imread(args.leaf)
    assert leaf is not None, f"cannot read {args.leaf}"
    summary = dict(calibration=run_calibration(), filtering=run_filtering(leaf), features=run_features(leaf))
    (OUT / "task1_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print("saved outputs/task1_summary.json")


if __name__ == "__main__":
    main()
