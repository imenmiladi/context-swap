"""Shadow extraction.

1. Estimate the original background colour (flat median, or a fitted plane per channel).
2. shadow = clip(1 - luminance / background_luminance, 0, 1): 0 = background, 1 = black.
3. Hue test: a real shadow darkens the background WITHOUT changing its hue. Pixels whose
   R/G/B ratios to the background differ too much (a wooden stool on a white wall) are
   not shadows and are rejected.
4. Smooth (masked Gaussian: the subject's pixels never leak in) and soft-threshold noise.
5. Extend the map into the semi-transparent edge band by copying the NEAREST background
   pixel, so the shadow reaches the subject's edge however wide that band is.
6. Keep only shadow regions attached to the subject.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

from .utils import luminance

BG_MODELS = ("uniform", "gradient")


def _background_samples(alpha: np.ndarray, border_frac: float = 0.03) -> np.ndarray:
    h, w = alpha.shape
    b = max(2, int(round(min(h, w) * border_frac)))
    m = np.zeros((h, w), bool)
    m[:b, :] = m[-b:, :] = True
    m[:, :b] = m[:, -b:] = True
    m &= alpha < 0.05
    if m.sum() < 200:
        m = alpha < 0.05
    return m


def _fit_plane(xn, yn, vals):
    """Robust fit  v ~ a*x + b*y + c  (outliers are rejected)."""
    A = np.stack([xn, yn, np.ones_like(xn)], axis=1)
    keep = np.ones(len(vals), bool)
    coef = np.zeros(3)
    for _ in range(3):
        coef, *_ = np.linalg.lstsq(A[keep], vals[keep], rcond=None)
        res = vals - A @ coef
        mad = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep]))) + 1e-3
        new_keep = np.abs(res) < 2.5 * mad
        if new_keep.sum() < 50:
            break
        keep = new_keep
    return coef


def estimate_background(rgb, alpha, model="uniform"):
    """Per-pixel RGB of the original background (H, W, 3) float32, or None."""
    if model not in BG_MODELS:
        raise ValueError(f"bg model must be one of {BG_MODELS}")
    h, w = alpha.shape
    ys, xs = np.nonzero(_background_samples(alpha))
    if len(ys) < 50:
        return None
    vals = rgb[ys, xs].astype(np.float32)
    if model == "uniform":
        med = np.median(vals, axis=0)
        return np.broadcast_to(med, (h, w, 3)).astype(np.float32).copy()

    if len(ys) > 50_000:
        idx = np.random.default_rng(0).choice(len(ys), 50_000, replace=False)
        ys, xs, vals = ys[idx], xs[idx], vals[idx]
    xn, yn = xs / max(w - 1, 1), ys / max(h - 1, 1)
    gx, gy = np.meshgrid(np.linspace(0, 1, w, dtype=np.float32), np.linspace(0, 1, h, dtype=np.float32))
    out = np.empty((h, w, 3), np.float32)
    for c in range(3):
        coef = _fit_plane(xn, yn, vals[:, c])
        out[..., c] = coef[0] * gx + coef[1] * gy + coef[2]
    return np.clip(out, 0, 255)


def estimate_background_luminance(lum_or_rgb, alpha, model="uniform"):
    bg = estimate_background(lum_or_rgb, alpha, model)
    return None if bg is None else luminance(bg)


def extract_shadow_map(rgb, alpha, bg_model="uniform", threshold=0.04, smooth=2.0, hue_tolerance=0.25):
    """Return (shadow_map in [0, 1], estimated background luminance or None).

    hue_tolerance: max allowed spread of the per-channel ratio pixel/background
    (0.25 = 25 %). Set to 0 to disable the hue test.
    """
    h, w = alpha.shape
    rgbf = rgb.astype(np.float32)
    bg = estimate_background(rgbf, alpha, bg_model)
    if bg is None:
        return np.zeros((h, w), np.float32), None

    lum = luminance(rgbf)
    bg_lum = np.maximum(luminance(bg), 20.0)
    s = np.clip(1.0 - lum / bg_lum, 0.0, 1.0)

    # hue test: a real shadow scales R, G and B by about the same factor
    if hue_tolerance and hue_tolerance > 0:
        ratio = rgbf / np.maximum(bg, 10.0)
        spread = (ratio.max(-1) - ratio.min(-1)) / np.maximum(ratio.mean(-1), 1e-3)
        gate = np.clip(2.0 - spread / hue_tolerance, 0.0, 1.0)  # 1 below tol, 0 above 2*tol
        s = s * gate

    # masked Gaussian: only background pixels contribute
    valid = alpha < 0.05
    sigma = max(float(smooth), 1.0)
    vf = valid.astype(np.float32)
    num = cv2.GaussianBlur(s * vf, (0, 0), sigma)
    den = cv2.GaussianBlur(vf, (0, 0), sigma)
    s = np.where(valid & (den > 1e-6), num / np.maximum(den, 1e-6), 0.0).astype(np.float32)

    # soft threshold to kill sensor noise / texture
    s = np.clip((s - threshold) / (1.0 - threshold), 0.0, 1.0)

    # extend into the semi-transparent band: copy the nearest background pixel
    band_px = max(16, int(0.03 * max(h, w)))
    dist, (iy, ix) = ndimage.distance_transform_edt(~valid, return_indices=True)
    s = np.where(valid, s, np.where(dist <= band_px, s[iy, ix], 0.0)).astype(np.float32)

    # keep only shadow regions attached to the subject
    binary = (s > 0).astype(np.uint8)
    touch = cv2.dilate((alpha >= 0.5).astype(np.uint8), np.ones((7, 7), np.uint8))
    _, labels = cv2.connectedComponents(binary, connectivity=8)
    keep = np.unique(labels[(touch > 0) & (binary > 0)])
    keep = keep[keep > 0]
    s = s * np.isin(labels, keep)
    s[alpha > 0.98] = 0.0

    return s.astype(np.float32), float(np.median(bg_lum))