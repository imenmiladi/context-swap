"""End-to-end pipeline, split in small pure stages so the UI can cache each of them.

    load_image -> compute_alpha (segment.py)          [slow, depends on the image only]
               -> refine_alpha                         [fast]
               -> estimate_foreground (decontaminate)  [medium]
               -> extract_shadow_map (shadow.py)       [fast]
               -> composite                            [instant, depends on the colour]
"""
from __future__ import annotations

import io
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from .segment import compute_alpha
from .shadow import extract_shadow_map
from .utils import hex_to_rgb


@dataclass
class RenderParams:
    bg_hex: str = "#FFFFFF"
    keep_shadows: bool = False
    shadow_strength: float = 0.8      # 0 = no shadow, 1 = shadow can reach black
    bg_model: str = "uniform"         # "uniform" | "gradient": model of the ORIGINAL background
    shadow_threshold: float = 0.04    # ignore darkening below 4 % (noise / texture)
    shadow_smooth: float = 2.0        # px, Gaussian sigma applied to the shadow map
    shadow_hue_tolerance: float = 0.25  # reject 'shadows' whose hue differs from the backdrop (0 = off)
    decontaminate: bool = True        # remove old-background colour bleeding at the edges
    edge_shrink: int = 0              # px, erode the matte to cut halos
    feather: float = 0.0              # px, blur the matte for softer edges


@dataclass
class Result:
    image: Image.Image
    alpha: np.ndarray
    foreground: np.ndarray            # uint8 (H, W, 3), decontaminated subject colours
    shadow_map: np.ndarray | None
    timings: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- I/O
def load_image(source, max_side: int = 2048) -> Image.Image:
    """Load a path / bytes / PIL image as RGB, honouring EXIF rotation.

    Transparent inputs are flattened on white. Images larger than `max_side`
    (longest side) are downscaled; max_side=0 disables the cap.
    """
    if isinstance(source, Image.Image):
        img = source
    elif isinstance(source, (bytes, bytearray)):
        img = Image.open(io.BytesIO(source))
    else:
        img = Image.open(Path(source))
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        base = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        base.alpha_composite(rgba)
        img = base
    img = img.convert("RGB")
    if max_side and max(img.size) > max_side:
        s = max_side / max(img.size)
        img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
    return img


# ------------------------------------------------------------------- stages
def refine_alpha(alpha: np.ndarray, edge_shrink: int = 0, feather: float = 0.0) -> np.ndarray:
    a = alpha.astype(np.float32)
    if edge_shrink > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * edge_shrink + 1, 2 * edge_shrink + 1))
        a = cv2.erode(a, k)
    if feather > 0:
        a = cv2.GaussianBlur(a, (0, 0), feather)
    return np.clip(a, 0.0, 1.0)


def estimate_foreground(rgb: np.ndarray, alpha: np.ndarray, decontaminate: bool = True) -> np.ndarray:
    """Foreground colours (float32, 0-255). With decontamination, semi-transparent edge pixels
    get the colour of the subject instead of a mix with the old background (no halo)."""
    if not decontaminate:
        return rgb.astype(np.float32)
    try:
        from pymatting import estimate_foreground_ml
    except ImportError:
        warnings.warn("pymatting not installed: skipping edge decontamination")
        return rgb.astype(np.float32)
    fg = estimate_foreground_ml(rgb.astype(np.float64) / 255.0, alpha.astype(np.float64))
    return (np.clip(fg, 0.0, 1.0) * 255.0).astype(np.float32)


def composite(
    foreground: np.ndarray,
    alpha: np.ndarray,
    bg_hex: str,
    shadow_map: np.ndarray | None = None,
    shadow_strength: float = 0.8,
) -> np.ndarray:
    """Alpha-composite the subject over a solid colour; the shadow multiplies that colour."""
    bg = np.array(hex_to_rgb(bg_hex), np.float32)
    layer = np.broadcast_to(bg, foreground.shape).copy()
    if shadow_map is not None:
        layer *= (1.0 - shadow_strength * shadow_map)[..., None]  # multiply: shadow keeps the new hue
    a = alpha[..., None]
    out = foreground * a + layer * (1.0 - a)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ one-shot API
def render(image: Image.Image, alpha: np.ndarray, params: RenderParams) -> Result:
    """Run every stage after segmentation."""
    rgb = np.asarray(image.convert("RGB"))
    if alpha.shape != rgb.shape[:2]:
        alpha = cv2.resize(alpha, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    timings = {}

    t = time.perf_counter()
    a = refine_alpha(alpha, params.edge_shrink, params.feather)
    fg = estimate_foreground(rgb, a, params.decontaminate)
    timings["foreground"] = time.perf_counter() - t

    shadow = None
    if params.keep_shadows:
        t = time.perf_counter()
        shadow, _ = extract_shadow_map(
        rgb, alpha, params.bg_model, params.shadow_threshold, params.shadow_smooth, params.shadow_hue_tolerance)
        timings["shadow"] = time.perf_counter() - t

    out = composite(fg, a, params.bg_hex, shadow, params.shadow_strength)
    return Result(Image.fromarray(out), a, np.clip(fg, 0, 255).astype(np.uint8), shadow, timings)


def process(source, params: RenderParams, engine: str = "birefnet", max_side: int = 2048) -> Result:
    """Load -> segment -> render. Convenience wrapper used by the CLI."""
    image = load_image(source, max_side)
    t = time.perf_counter()
    alpha = compute_alpha(image, engine)
    seg_time = time.perf_counter() - t
    result = render(image, alpha, params)
    result.timings["segmentation"] = seg_time
    return result
