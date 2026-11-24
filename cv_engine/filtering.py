"""Linear filtering, denoising, gradients and frequency analysis (rubric 1.2)."""
from __future__ import annotations

import cv2
import numpy as np


# ----------------------------------------------------------------- noise & metrics
def add_gaussian_noise(img, sigma=20, seed=0):
    rng = np.random.default_rng(seed)
    out = img.astype(np.float32) + rng.normal(0, sigma, img.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def add_salt_pepper(img, amount=0.04, seed=0):
    rng = np.random.default_rng(seed)
    out = img.copy()
    m = rng.random(img.shape[:2])
    out[m < amount / 2] = 0
    out[m > 1 - amount / 2] = 255
    return out


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


# ----------------------------------------------------------------- spatial filters
def convolve2d_naive(img, kernel):
    """Correlation implemented from scratch with NumPy (zero-padded, 'same' size).

    Vectorised over the kernel footprint (loop over kernel taps, not pixels), so it
    stays fast while remaining an explicit statement of the definition
    g(x,y) = sum_{u,v} k(u,v) f(x+u, y+v). Verified against cv2.filter2D in tests.
    """
    img = img.astype(np.float32)
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode="constant")
    out = np.zeros_like(img)
    for i in range(kh):
        for j in range(kw):
            out += kernel[i, j] * padded[i:i + img.shape[0], j:j + img.shape[1]]
    return out


def box_filter(img, k=5):
    return cv2.blur(img, (k, k))


def gaussian_filter(img, k=5, sigma=0):
    return cv2.GaussianBlur(img, (k, k), sigma)


def median_filter(img, k=5):
    return cv2.medianBlur(img, k)


def bilateral_filter(img, d=9, sigma_color=50, sigma_space=50):
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


def denoise_comparison(clean, noisy):
    """PSNR of each denoiser against the clean image."""
    return {
        "noisy": psnr(clean, noisy),
        "box 5x5": psnr(clean, box_filter(noisy, 5)),
        "gaussian 5x5": psnr(clean, gaussian_filter(noisy, 5)),
        "median 5x5": psnr(clean, median_filter(noisy, 5)),
        "bilateral": psnr(clean, bilateral_filter(noisy)),
    }


# ----------------------------------------------------------------- gradients & edges
def sobel_gradients(gray, ksize=3):
    """Return gradient magnitude and orientation (radians) from Sobel derivatives."""
    g = cv2.GaussianBlur(gray, (5, 5), 1.0).astype(np.float32)   # suppress noise before differentiating
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=ksize)
    return np.hypot(gx, gy), np.arctan2(gy, gx)


def canny_edges(gray, low=None, high=None, pct=90, ratio=0.4):
    """Canny edges. If thresholds are not given they are derived from the image's own gradients:
    high = the `pct`-th percentile of the Sobel magnitude, low = ratio * high.

    (A median-intensity rule, common in tutorials, ties thresholds to brightness instead of contrast and
    returned an almost empty edge map on low-contrast leaf images, so gradient statistics are used.)
    """
    blur = cv2.GaussianBlur(gray, (5, 5), 1.4)
    if low is None or high is None:
        gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
        high = float(np.percentile(np.hypot(gx, gy), pct))
        low = ratio * high
    return cv2.Canny(blur, low, high, L2gradient=True)


# ----------------------------------------------------------------- frequency domain
def fft_spectrum(gray):
    """Centred log-magnitude spectrum for visualisation."""
    F = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
    return np.log1p(np.abs(F))


def fft_filter(gray, cutoff=30, mode="lowpass"):
    """Gaussian low/high-pass filtering in the Fourier domain.

    A Gaussian transfer function is used instead of an ideal (brick-wall) mask because
    an ideal mask has a sinc impulse response, which produces ringing artefacts.
    """
    h, w = gray.shape
    F = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
    v, u = np.ogrid[-h // 2:h - h // 2, -w // 2:w - w // 2]
    H = np.exp(-(u ** 2 + v ** 2) / (2.0 * cutoff ** 2))
    if mode == "highpass":
        H = 1.0 - H
    out = np.real(np.fft.ifft2(np.fft.ifftshift(F * H)))
    return out if mode == "highpass" else np.clip(out, 0, 255)


def normalise_uint8(a):
    a = a.astype(np.float32)
    a = (a - a.min()) / (a.max() - a.min() + 1e-8)
    return (a * 255).astype(np.uint8)
