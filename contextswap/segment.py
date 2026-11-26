"""Subject segmentation -> soft alpha matte in [0, 1].

Engines
-------
birefnet       BiRefNet (ZhengPeng7/BiRefNet), best quality, default.
birefnet-lite  BiRefNet_lite, lighter, better suited to CPU.
grabcut        Classical OpenCV GrabCut. No download, no torch. Much weaker:
               used for tests, quick demos and as a fallback.

Heavy imports (torch / transformers) are done lazily so that the rest of the
package works without them.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

ENGINES = {
    "birefnet": "ZhengPeng7/BiRefNet",
    "birefnet-lite": "ZhengPeng7/BiRefNet_lite",
    "grabcut": None,
}

_MODEL_CACHE: dict[str, "BiRefNetSegmenter"] = {}


class BiRefNetSegmenter:
    """Thin wrapper around the BiRefNet checkpoint from the Hugging Face hub."""

    INPUT_SIZE = (1024, 1024)  # resolution the model was trained at
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # ImageNet statistics
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, model_id: str, device: str | None = None):
        import torch
        from transformers import AutoModelForImageSegmentation

        self.torch = torch
        # MPS is avoided on purpose: BiRefNet uses deformable convs not supported there.
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForImageSegmentation.from_pretrained(model_id, trust_remote_code=True)
        self.model.to(self.device).eval()

    def predict(self, image: Image.Image) -> np.ndarray:
        torch = self.torch
        w, h = image.size
        small = image.convert("RGB").resize(self.INPUT_SIZE, Image.BILINEAR)
        arr = (np.asarray(small, dtype=np.float32) / 255.0 - self.MEAN) / self.STD
        x = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.model(x)[-1].sigmoid().float().cpu()  # last output = final mask
        pred = torch.nn.functional.interpolate(pred, size=(h, w), mode="bilinear", align_corners=False)
        return pred[0, 0].numpy().clip(0.0, 1.0)


def _grabcut_alpha(image: Image.Image) -> np.ndarray:
    import cv2

    arr = np.asarray(image.convert("RGB"))[..., ::-1].copy()  # RGB -> BGR
    h, w = arr.shape[:2]
    scale = min(1.0, 640 / max(h, w))  # GrabCut is slow, run it on a small copy
    small = cv2.resize(arr, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    mask = np.zeros((sh, sw), np.uint8)
    mx, my = int(sw * 0.05), int(sh * 0.05)  # assume the subject lies inside a 5% margin
    rect = (mx, my, sw - 2 * mx, sh - 2 * my)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(small, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1.0, 0.0).astype(np.float32)
    fg = cv2.GaussianBlur(fg, (0, 0), 1.5)
    return cv2.resize(fg, (w, h), interpolation=cv2.INTER_LINEAR).clip(0.0, 1.0)


def compute_alpha(image: Image.Image, engine: str = "birefnet") -> np.ndarray:
    """Return a float32 alpha matte (H, W) in [0, 1] for the given PIL image."""
    if engine not in ENGINES:
        raise ValueError(f"Unknown engine {engine!r}. Choose from {list(ENGINES)}")
    if engine == "grabcut":
        return _grabcut_alpha(image)
    if engine not in _MODEL_CACHE:
        _MODEL_CACHE[engine] = BiRefNetSegmenter(ENGINES[engine])
    return _MODEL_CACHE[engine].predict(image)
