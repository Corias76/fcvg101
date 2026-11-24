"""Interest points and two-view matching (rubric 1.3).

Includes a from-scratch Harris detector (NumPy/SciPy) alongside OpenCV's SIFT/ORB,
and a matching routine using Lowe's ratio test followed by RANSAC.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage


# ----------------------------------------------------------------- Harris from scratch
def harris_response(gray, sigma=1.5, k=0.05):
    """Harris corner response R = det(M) - k * trace(M)^2.

    M is the structure tensor: image-gradient products smoothed by a Gaussian window.
    Derivatives use Sobel on a float image; k=0.05 is within the usual 0.04-0.06 range.
    """
    g = gray.astype(np.float32) / 255.0
    Ix = ndimage.sobel(g, axis=1)
    Iy = ndimage.sobel(g, axis=0)
    Sxx = ndimage.gaussian_filter(Ix * Ix, sigma)
    Syy = ndimage.gaussian_filter(Iy * Iy, sigma)
    Sxy = ndimage.gaussian_filter(Ix * Iy, sigma)
    det = Sxx * Syy - Sxy ** 2
    tr = Sxx + Syy
    return det - k * tr ** 2


def harris_corners(gray, sigma=1.5, k=0.05, rel_thresh=0.01, nms_size=7, max_pts=500):
    """Harris keypoints: threshold relative to max response + non-maximum suppression."""
    R = harris_response(gray, sigma, k)
    local_max = (R == ndimage.maximum_filter(R, size=nms_size))
    mask = local_max & (R > rel_thresh * R.max())
    ys, xs = np.nonzero(mask)
    order = np.argsort(-R[ys, xs])[:max_pts]
    return np.stack([xs[order], ys[order]], axis=1).astype(np.float32), R


# ----------------------------------------------------------------- detectors / matching
def detect_and_describe(gray, method="sift", n_features=1500):
    """Return keypoints and descriptors for 'sift' or 'orb'."""
    if method == "sift":
        det = cv2.SIFT_create(nfeatures=n_features)
    elif method == "orb":
        det = cv2.ORB_create(nfeatures=n_features)
    else:
        raise ValueError("method must be 'sift' or 'orb'")
    return det.detectAndCompute(gray, None)


def match_ratio_test(des1, des2, method="sift", ratio=0.75):
    """k-NN matching (k=2) filtered by Lowe's ratio test.

    A match is kept only if its best distance is clearly smaller than the second best,
    which rejects ambiguous matches (e.g. repeated textures such as leaf veins).
    SIFT descriptors are float vectors (L2); ORB descriptors are binary (Hamming).
    """
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return []
    norm = cv2.NORM_L2 if method == "sift" else cv2.NORM_HAMMING
    bf = cv2.BFMatcher(norm)
    good = []
    for pair in bf.knnMatch(des1, des2, k=2):
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
            good.append(pair[0])
    return good


def estimate_homography(kp1, kp2, matches, reproj_thresh=3.0):
    """RANSAC homography from matches; returns H and boolean inlier mask."""
    if len(matches) < 4:
        return None, np.zeros(len(matches), bool)
    src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, reproj_thresh)
    return H, (mask.ravel().astype(bool) if mask is not None else np.zeros(len(matches), bool))


def match_pair(gray1, gray2, method="sift", ratio=0.75, n_features=1500):
    """Full pipeline: detect -> describe -> ratio-test match -> RANSAC. Returns a dict."""
    kp1, d1 = detect_and_describe(gray1, method, n_features)
    kp2, d2 = detect_and_describe(gray2, method, n_features)
    matches = match_ratio_test(d1, d2, method, ratio)
    H, inl = estimate_homography(kp1, kp2, matches)
    return dict(kp1=kp1, kp2=kp2, matches=matches, H=H, inliers=inl)


def draw_matches(img1, img2, res, only_inliers=True, max_draw=80):
    ms = [m for m, ok in zip(res["matches"], res["inliers"]) if ok or not only_inliers][:max_draw]
    return cv2.drawMatches(img1, res["kp1"], img2, res["kp2"], ms, None,
                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


def draw_keypoints(img, pts):
    vis = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for x, y in pts:
        cv2.circle(vis, (int(x), int(y)), 3, (0, 0, 255), 1)
    return vis


# ----------------------------------------------------------------- evaluation with known geometry
def synthetic_second_view(img, angle=12, scale=0.9, shear=0.0, persp=(2e-4, -1e-4), seed=0):
    """Warp an image by a known homography to simulate a second camera view.

    Returns warped image and the ground-truth 3x3 homography H_gt (img -> warped),
    which lets us measure matching accuracy quantitatively.
    """
    h, w = img.shape[:2]
    c = (w / 2, h / 2)
    A = np.vstack([cv2.getRotationMatrix2D(c, angle, scale), [0, 0, 1]])
    A[0, 1] += shear
    P = np.eye(3)
    P[2, 0], P[2, 1] = persp
    H = P @ A
    return cv2.warpPerspective(img, H, (w, h)), H


def homography_error(H_est, H_gt, w, h, n=50):
    """Mean pixel distance between H_est and H_gt applied to a grid of image points."""
    if H_est is None:
        return float("nan")
    xs, ys = np.meshgrid(np.linspace(0, w - 1, n), np.linspace(0, h - 1, n))
    pts = np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float32).reshape(-1, 1, 2)
    a = cv2.perspectiveTransform(pts, H_est.astype(np.float64))
    b = cv2.perspectiveTransform(pts, H_gt.astype(np.float64))
    return float(np.mean(np.linalg.norm(a - b, axis=2)))
