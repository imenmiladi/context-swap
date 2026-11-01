"""Tiny synthetic scene with a KNOWN alpha, so the pipeline can be tested without any model."""
import cv2
import numpy as np


def make_scene(w=480, h=360, gradient=0.0, bg_lum=225, with_shadow=True, seed=0):
    """Red ellipse on a grey backdrop, with a soft cast shadow on the lower right.

    gradient: left->right luminance change of the backdrop (in grey levels).
    Returns (rgb uint8, true_alpha float32, shadow_region bool).
    """
    rng = np.random.default_rng(seed)
    xs = np.linspace(-0.5, 0.5, w, dtype=np.float32)[None, :].repeat(h, 0)
    bg = np.full((h, w), float(bg_lum), np.float32) + gradient * xs
    rgb = np.repeat(bg[..., None], 3, axis=2)

    cx, cy, rx, ry = w // 2 - 20, h // 2 - 20, 70, 95
    subj = np.zeros((h, w), np.float32)
    cv2.ellipse(subj, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1, cv2.LINE_AA)

    shadow_region = np.zeros((h, w), bool)
    if with_shadow:
        sh = np.zeros((h, w), np.float32)
        cv2.ellipse(sh, (cx + 55, cy + ry - 10), (95, 28), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
        sh = cv2.GaussianBlur(sh, (0, 0), 6) * 0.45  # up to 45 % darker
        rgb *= (1 - sh)[..., None]
        shadow_region = (sh > 0.25) & (subj < 0.01)

    colour = np.array([200, 30, 40], np.float32)
    rgb = rgb * (1 - subj[..., None]) + colour * subj[..., None]
    rgb += rng.normal(0, 1.0, rgb.shape)  # sensor-like noise
    return np.clip(rgb, 0, 255).astype(np.uint8), subj, shadow_region
