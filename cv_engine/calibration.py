"""Image formation and camera calibration (Requirement a, rubric 1.1).

Pinhole model:  x ~ K [R | t] X   with   K = [[fx, s, cx], [0, fy, cy], [0, 0, 1]].
Lens distortion (Brown-Conrady) is modelled by dist = (k1, k2, p1, p2, k3).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class CalibrationResult:
    K: np.ndarray                      # 3x3 intrinsic matrix
    dist: np.ndarray                   # distortion coefficients (k1, k2, p1, p2, k3)
    rms: float                         # overall RMS reprojection error (pixels)
    per_image_error: list[float]       # mean reprojection error per used image
    used_images: list[str]
    image_size: tuple[int, int]        # (width, height)
    rvecs: list = field(default_factory=list, repr=False)
    tvecs: list = field(default_factory=list, repr=False)


def find_corners(image_paths, pattern=(9, 6), square_size=1.0):
    """Detect and sub-pixel refine inner chessboard corners.

    pattern is the number of INNER corners (cols, rows). Sub-pixel refinement
    (cornerSubPix) matters because calibration accuracy is limited by corner
    localisation noise, and integer-pixel corners add ~0.3 px of error.
    Returns object points, image points, image size, and the list of images used.
    """
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * square_size
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    obj_pts, img_pts, used, size = [], [], [], None
    for p in map(str, image_paths):
        gray = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        size = gray.shape[::-1]
        ok, corners = cv2.findChessboardCorners(gray, pattern)
        if not ok:
            continue
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        obj_pts.append(objp)
        img_pts.append(corners)
        used.append(p)
    return obj_pts, img_pts, size, used


def calibrate(image_paths, pattern=(9, 6), square_size=1.0) -> CalibrationResult:
    """Zhang-style calibration with cv2.calibrateCamera from >=3 views."""
    obj_pts, img_pts, size, used = find_corners(image_paths, pattern, square_size)
    if len(used) < 3:
        raise ValueError(f"Need >= 3 views with detectable chessboard, got {len(used)}")
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
    errs = []
    for o, i, r, t in zip(obj_pts, img_pts, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, r, t, K, dist)
        diff = i.reshape(-1, 2) - proj.reshape(-1, 2)   # reshape: OpenCV versions differ in array layout
        errs.append(float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))))
    return CalibrationResult(K, dist.ravel(), float(rms), errs, used, size, list(rvecs), list(tvecs))


def project_pinhole(K, R, t, X):
    """Project Nx3 world points with the ideal pinhole model (no distortion).

    Written explicitly (instead of cv2.projectPoints) to demonstrate the model:
    X_cam = R X + t ;  x = K X_cam ;  (u, v) = (x/z, y/z).
    """
    Xc = (R @ X.T + t.reshape(3, 1)).T
    x = (K @ Xc.T).T
    return x[:, :2] / x[:, 2:3]


def undistort(img, K, dist, alpha=0.0):
    """Remove lens distortion. alpha=0 crops to valid pixels, alpha=1 keeps all."""
    h, w = img.shape[:2]
    newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha, (w, h))
    out = cv2.undistort(img, K, dist, None, newK)
    if alpha == 0:
        x, y, rw, rh = roi
        if rw > 0 and rh > 0:
            out = out[y:y + rh, x:x + rw]
    return out


def draw_corners(img, pattern=(9, 6)):
    """Return a BGR copy of img with detected chessboard corners drawn (or unchanged)."""
    vis = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    vis = vis.copy()
    gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
    ok, corners = cv2.findChessboardCorners(gray, pattern)
    if ok:
        cv2.drawChessboardCorners(vis, pattern, corners, ok)
    return vis


def save_calibration(res: CalibrationResult, path):
    np.savez(path, K=res.K, dist=res.dist, rms=res.rms, size=res.image_size)


def load_calibration(path):
    d = np.load(path)
    return d["K"], d["dist"]


def glob_images(folder):
    return sorted(p for p in Path(folder).glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
